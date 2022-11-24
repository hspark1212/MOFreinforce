import os
import json
import copy
import random

import torch
from tqdm import tqdm

from utils.metrics import Metrics

from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup, get_constant_schedule

from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

import numpy as np
import selfies as sf

topo_to_cn = json.load(open("assets/final_topo_cn.json"))
mc_to_cn = json.load(open("assets/mc_cn.json"))


class Reinforce(LightningModule):
    def __init__(self, agent, predictors, config):
        super(Reinforce, self).__init__()

        self.agent = copy.deepcopy(agent)
        # freeze only encoder
        for param in self.agent.transformer.encoder.parameters():
            param.requires_grad = False

        self.freeze_agent = copy.deepcopy(agent)
        # freeze agent
        for param in self.freeze_agent.parameters():
            param.requires_grad = False

        self.predictors = predictors

        # reward
        self.threshold = config["threshold"]
        self.reward_max = config["reward_max"]

        # reinforce
        self.lr = config["lr"]
        self.decay_ol = config["decay_ol"]
        self.scheduler = config["scheduler"]
        self.decay_topo = config["decay_topo"]
        self.early_stop = config["early_stop"]
        self.ratio_exploit = config["ratio_exploit"]
        self.ratio_mask_mc = config["ratio_mask_mc"]

        # load model
        if config["load_path"] != "":
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

    def forward(self, src, max_len=128):
        """
        based on evaluate() in Generator
        :param src: encoded_input of generator [B=1, seq_len]
        :return:
        """

        vocab_to_idx = self.agent.vocab_to_idx

        # mask_mc of encoded_src
        if torch.randn(1) < self.ratio_mask_mc:
            src[:, 0].fill_(0)  # 0 is pad_idx

        src_mask = self.agent.transformer.make_src_mask(src)

        enc_src = self.agent.transformer.encoder(src, src_mask)  # [B=1, seq_len, hid_dim]

        # get target
        trg_indexes = [vocab_to_idx["[SOS]"]]

        list_prob = []

        for i in range(max_len):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(src.device)
            trg_mask = self.agent.transformer.make_trg_mask(trg_tensor)  # [B=1, 1, seq_len, seq_len]

            output = self.agent.transformer.decoder(
                trg_tensor, enc_src, trg_mask, src_mask)  # [B=1, seq_len, vocab_dim]

            with torch.no_grad():
                freeze_output = self.freeze_agent.transformer.decoder(
                    trg_tensor, enc_src, trg_mask, src_mask)  # [B=1, seq_len, vocab_dim]
                if torch.rand(1)[0] < self.ratio_exploit:  # and self.training: ### beta
                    output = freeze_output

            if "output_ol" in output.keys():
                out = output["output_ol"]
                prob = out.softmax(dim=-1)[0, -1]
                m = torch.multinomial(prob, 1).item()
            elif "output_mc" in output.keys():
                out = output["output_mc"]
                prob = out.softmax(dim=-1)[0]
                m = torch.multinomial(prob, 1).item()
            else:
                out = output["output_topo"]
                prob = out.softmax(dim=-1)[0]
                m = torch.multinomial(prob, 1).item()

            list_prob.append(prob[m])
            trg_indexes.append(m)

            if i >= 2 and m == vocab_to_idx["[EOS]"]:
                break

        # get topo
        topo_idx = trg_indexes[1]
        topo = self.agent.idx_to_topo[topo_idx]
        # get mc
        mc_idx = trg_indexes[2]
        mc = self.agent.idx_to_mc[mc_idx]
        # get ol
        ol_idx = trg_indexes[3:]
        ol_tokens = [self.agent.idx_to_vocab[idx] for idx in ol_idx]

        # check matching connection points
        topo_cn = topo_to_cn.get(topo, [0])  # if topo is [PAD] then topo_cn = [0]
        if len(topo_cn) == 1:
            topo_cn.append(2)
        mc_cn = mc_to_cn.get(mc, -1)  # if mc is [PAD] then mc_cn = -1
        ol_cn = ol_idx.count(self.agent.vocab_to_idx["[*]"])

        # convert to selfies and smiles
        try:
            gen_sf = "".join(ol_tokens[:-1])  # remove EOS token
            gen_sm = sf.decoder(gen_sf)
            new_gen_sf = sf.encoder(gen_sm)  # to check valid SELFIES
            assert ol_tokens[-1] == "[EOS]", Exception("The last token is not [EOS]")
            assert new_gen_sf == gen_sf, Exception("SELFIES error")
            assert set(topo_cn) == set([mc_cn, ol_cn]), Exception("connection points error")
        except Exception as e:
            print(e)
            # print(f"The failed gen_sf : {gen_sf}")
            # print(f"The failed gen_sm : {gen_sm}")
            gen_sf = None
            gen_sm = None

        ret = {
            "topo": topo,
            "mc": mc,
            "topo_idx": topo_idx,
            "mc_idx": mc_idx,
            "ol_idx": ol_idx,
            "gen_sf": gen_sf,
            "gen_sm": gen_sm,
            "list_prob": list_prob,
        }
        return ret

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        max_steps = self.trainer.estimated_stepping_batches
        print(f"mas_steps : {max_steps}")

        if self.scheduler == "warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=int(self.trainer.max_epochs * 0.1),
                num_training_steps=self.trainer.max_epochs,
            )
        elif self.scheduler == "constant":
            scheduler = get_constant_schedule(
                optimizer=optimizer,
            )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.agent.train()

        rl_loss = 0
        total_reward = 0
        num_fail = 0

        for enc_input in batch["encoded_input"]:
            src = enc_input.unsqueeze(0)
            out = self(src)
            if out["gen_sm"] is None:
                num_fail += 1
                continue

            # get reward from predictor
            with torch.no_grad():
                rewards, preds = self.get_reward(output_generator=out)
                reward = sum(rewards)

            total_reward += reward

            # Reinforce algorithm
            discounted_reward = reward
            tmp_loss = []

            # topology, metal cluster (the probability of metal cluster is almost zero, because it is used as input)
            for prob in out["list_prob"][:2]:
                rl_loss -= torch.log(prob) * reward * self.decay_topo
                tmp_loss.append(-torch.log(prob).item() * reward)
            # organic linker (discounter_reward is applied only for organic linkers)
            for i, prob in enumerate(out["list_prob"][2:][::-1]):
                rl_loss -= torch.log(prob) * discounted_reward
                discounted_reward = discounted_reward * self.decay_ol

                tmp_loss.append(-torch.log(prob).item() * discounted_reward)
            print(rewards, preds, out["topo"], out["mc"], out["gen_sm"], out["gen_sf"], src[0, 1].item())
            print(rl_loss, tmp_loss[:2], sum(tmp_loss[2:]))

        n_batch = batch["encoded_input"].shape[0]
        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch
        num_fail = torch.tensor(num_fail) / n_batch

        self.log("train/rl_loss", rl_loss)
        self.log("train/reward", total_reward, prog_bar=True)
        self.log("train/num_fail", num_fail, prog_bar=True)

        return rl_loss

    def validation_step(self, batch, batch_idx):
        return batch

    def validation_epoch_end(self, batches):
        list_src = torch.concat([b["encoded_input"] for b in batches], dim=0)
        metrics = Metrics(self.agent.vocab_to_idx, self.agent.idx_to_vocab)
        split = "val"

        metrics = self.update_metrics(list_src, metrics, split)
        if metrics.get_mean(metrics.scaffold) < self.early_stop:
            raise Exception(f"EarlyStopping for scaffold : scaffold accuracy is less than {self.early_stop}")

    def test_step(self, batch, batch_idx):
        return batch

    def test_epoch_end(self, batches):
        list_src = torch.concat([b["encoded_input"] for b in batches], dim=0)
        metrics = Metrics(self.agent.vocab_to_idx, self.agent.idx_to_vocab)
        split = "test"

        metrics = self.update_metrics(list_src, metrics, split)
        # save results
        ret = {
            "rewards": metrics.rewards,
            "preds": metrics.preds,
            "gen_sms": metrics.gen_ol,
            "gen_mcs": metrics.gen_mc,
            "gen_topos": metrics.gen_topo,
        }
        path = os.path.join(self.logger.save_dir, f"results_{self.logger.name}.json")
        json.dump(ret, open(path, "w"))

    def get_reward(self,
                   output_generator,
                   sos_idx=1,
                   pad_idx=0,
                   max_len=128,
                   ):
        # create the input of predictor
        device = self.predictors[0].device
        batch = {}
        batch["mc"] = torch.LongTensor([output_generator["mc_idx"]]).to(device)
        batch["topo"] = torch.LongTensor([output_generator["topo_idx"]]).to(device)
        batch["ol"] = torch.LongTensor([[sos_idx] +  # add sos_idx
                                        output_generator["ol_idx"][:max_len - 4] +
                                        [pad_idx] * (max_len - 4 - len(output_generator["ol_idx"]))]).to(
            device)  # add pad_idx

        preds = []
        rewards = []
        for i, predictor in enumerate(self.predictors):
            infer = predictor.infer(batch)
            p = predictor.regression_head(infer["cls_feats"])
            p = predictor.normalizer.decode(p)
            p = p.detach().item()
            preds.append(p)

            # get reward
            if self.threshold:
                if abs(p) >= abs(self.reward_max[i]):
                    r = 1
                else:
                    r = 0
                rewards.append(r)
            else:
                rewards.append(p / self.reward_max[i])

        if self.threshold:
            if all(rewards):
                rewards = [1] * len(rewards)
            else:
                rewards = [0] * len(rewards)
        return rewards, preds

    def update_metrics(self, list_src, metrics, split):
        for src in tqdm(list_src):
            out = self(src.unsqueeze(0))

            if out["gen_sm"] is None:
                metrics.num_fail.append(1)
                continue
            else:
                metrics.num_fail.append(0)

            with torch.no_grad():
                rewards, preds = self.get_reward(output_generator=out)

            metrics.update(out, src, rewards=rewards, preds=preds)

        self.log(f"{split}/conn_match", metrics.get_mean(metrics.conn_match))
        self.log(f"{split}/unique_ol", len(set(metrics.gen_ol)) / len(metrics.gen_ol))
        self.log(f"{split}/unique_topo_mc", len(set(zip(metrics.gen_topo, metrics.gen_mc))) / len(metrics.gen_topo))
        self.log(f"{split}/scaffold", metrics.get_mean(metrics.scaffold))

        self.log(f"{split}/total_reward", metrics.get_mean(metrics.rewards))
        for i, reward in enumerate(zip(*metrics.rewards)):
            self.log(f"{split}/reward_{i}", metrics.get_mean(reward))
        for i, pred in enumerate(zip(*metrics.preds)):
            self.log(f"{split}/target_{i}", metrics.get_mean(pred))
        self.log(f"{split}/num_fail", metrics.get_mean(metrics.num_fail))

        # add image to log
        # gen_ol with frags (32 images)
        for i in range(32):
            idx = random.Random(i).choice(range(len(metrics.gen_ol)))
            ol = metrics.gen_ol[idx]
            frags = metrics.input_frags[idx]
            imgs = []
            for s in [ol] + frags:
                m = Chem.MolFromSmiles(s)
                if not m:
                    continue
                img = MolToImage(m)
                img = np.array(img)
                img = torch.tensor(img)
                imgs.append(img)
            imgs = np.stack(imgs, axis=0)
            self.logger.experiment.add_image(f"{split}/{i}", imgs, self.global_step, dataformats="NHWC")

        # total gen_ol
        imgs = []
        for i, m in enumerate(metrics.gen_ol[:32]):
            try:
                m = Chem.MolFromSmiles(m)
                img = MolToImage(m)
                img = np.array(img)
                img = torch.tensor(img)
                imgs.append(img)
            except Exception as e:
                print(e)
        imgs = np.stack(imgs, axis=0)
        self.logger.experiment.add_image(f"{split}/gen_ol/", imgs, self.global_step, dataformats="NHWC")
        return metrics
