import os
import json
import copy
from tqdm import tqdm

import torch

from utils.metrics import metrics_generator

from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup, get_constant_schedule

from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

import numpy as np
import selfies as sf


class Reinforce(LightningModule):
    def __init__(self, agent, predictor, config):
        super(Reinforce, self).__init__()

        self.agent = copy.deepcopy(agent)
        # freeze only encoder
        for param in self.agent.transformer.encoder.parameters():
            param.requires_grad = False

        self.freeze_agent = copy.deepcopy(agent)
        # freeze agent
        for param in self.freeze_agent.parameters():
            param.requires_grad = False

        self.predictor = predictor
        # reward
        self.criterion = config["criterion"]
        assert self.criterion == "continuous-negative" or self.criterion == "continuous-positive" \
               or isinstance(self.criterion, float)
        self.reward_positive = config["reward_positive"]
        self.reward_negative = config["reward_negative"]
        self.exploit_ratio = config["exploit_ratio"]
        # reinforce
        self.lr = config["lr"]
        self.decay_ol = config["decay_ol"]
        self.scheduler = config["scheduler"]
        self.decay_topo = config["decay_topo"]
        self.early_stop = config["early_stop"]

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
        src_mask = self.agent.transformer.make_src_mask(src)

        # gent encoded src
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
                if torch.rand(1)[0] < self.exploit_ratio and self.training:
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

        # convert to selfies and smiles
        try:
            gen_sf = "".join(ol_tokens[:-1])  # remove EOS token
            gen_sm = sf.decoder(gen_sf)
            new_gen_sf = sf.encoder(gen_sm) # to check valid SELFIES
            # m = Chem.MolFromSmiles(gen_sm)
            # gen_sm = Chem.MolToSmiles(m)  # canonical smiles

            assert ol_tokens[-1] == "[EOS]", print("The last token is not [EOS]")
            assert new_gen_sf == gen_sf, print("SELFIES error")
        except Exception as e:
            print(f"The failed gen_sf : {gen_sf}")
            print(f"The failed gen_sm : {gen_sm}")
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
                reward, pred = self.get_reward(output_generator=out)

            total_reward += reward

            # Reinforce algorithm
            discounted_reward = reward
            tmp_loss = []
            """ Beta version for alternative learning
            if (self.current_epoch // 2) % 2 == 0:
                self.decay_ol = 0.
                self.decay_topo = 1.
            else:
                self.decay_ol = 1.
                self.decay_topo = 0.
            """
            # topology, metal cluster (the probability of metal cluster is almost zero, because it is used as input)
            for prob in out["list_prob"][:2]:
                rl_loss -= torch.log(prob) * reward * self.decay_topo
                tmp_loss.append(-torch.log(prob).item() * reward)
            # oragnic linker (discounter_reward is applied only for organic linkers)
            for i, prob in enumerate(out["list_prob"][2:][::-1]):
                discounted_reward = discounted_reward * self.decay_ol  # switch
                rl_loss -= torch.log(prob) * discounted_reward  # switch
                tmp_loss.append(-torch.log(prob).item() * discounted_reward)
            print(reward, pred, out["topo"], out["mc"], out["gen_sm"], out["gen_sf"], src[0, 1].item())
            print(rl_loss, tmp_loss[:2], sum(tmp_loss[2:])) #, tmp_loss)

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

        rewards = []
        preds = []
        gen_sms = []
        num_fail = 0
        for src in list_src:
            out = self(src.unsqueeze(0))

            if out["gen_sm"] is None:
                num_fail += 1
                continue

            with torch.no_grad():
                reward, pred = self.get_reward(output_generator=out)

            rewards.append(reward)
            preds.append(pred)
            gen_sms.append(out["gen_sm"])


        n_batch = list_src.shape[0]
        self.log("val/reward", sum(rewards) / n_batch)
        self.log("val/target", sum(preds) / n_batch)
        self.log("val/num_fail", torch.tensor(num_fail) / n_batch)

        # add image to log
        imgs = []
        for i, m in enumerate(gen_sms[:32]):
            try:
                m = Chem.MolFromSmiles(m)
                img = MolToImage(m)
                img = np.array(img)
                img = torch.tensor(img)
                imgs.append(img)
            except Exception as e:
                print(e)
        imgs = np.stack(imgs, axis=0)
        self.logger.experiment.add_image("val/gen_ol/", imgs, self.global_step, dataformats="NHWC")

        # metrics_generator
        metrics_generator(self.agent, list_src, early_stop=self.early_stop)

    def test_step(self, batch, batch_idx):
        return batch

    def test_epoch_end(self, batches):
        list_src = torch.concat([b["encoded_input"] for b in batches], dim=0)

        rewards = []
        preds = []
        gen_sms = []
        gen_mcs = []
        gen_topos = []
        num_fail = 0
        for src in tqdm(list_src):
            out = self(src.unsqueeze(0))
            if out["gen_sm"] is None:
                num_fail += 1
                continue

            with torch.no_grad():
                reward, pred = self.get_reward(output_generator=out)
            rewards.append(reward)
            preds.append(pred)
            gen_sms.append(out["gen_sm"])
            gen_mcs.append(out["mc"])
            gen_topos.append(out["topo"])

        n_batch = list_src.shape[0]
        self.log("val/reward", sum(rewards) / n_batch)
        self.log("val/target", sum(preds) / n_batch)
        self.log("val/num_fail", num_fail / n_batch)
        # save results
        ret = {
            "rewards": rewards,
            "preds": preds,
            "gen_sms": gen_sms,
            "gen_mcs": gen_mcs,
            "gen_topos": gen_topos,
        }

        # add image to log
        imgs = []
        for i, m in enumerate(gen_sms[:32]):
            try:
                m = Chem.MolFromSmiles(m)
                img = MolToImage(m)
                img = np.array(img)
                img = torch.tensor(img)
                imgs.append(img)
            except Exception as e:
                print(e)
        imgs = np.stack(imgs, axis=0)
        self.logger.experiment.add_image("val/gen_ol/", imgs, self.global_step, dataformats="NHWC")

        # metrics_generator
        metrics_generator(self.agent, list_src)

        # save results
        path = os.path.join(self.logger.save_dir, self.logger.name, "results.json")
        json.dump(ret, open(path, "w"))

    def get_reward(self,
                   output_generator,
                   sos_idx=1,
                   pad_idx=0,
                   max_len=128,
                   ):
        # create the input of predictor
        device = self.predictor.device
        batch = {}
        batch["mc"] = torch.LongTensor([output_generator["mc_idx"]]).to(device)
        batch["topo"] = torch.LongTensor([output_generator["topo_idx"]]).to(device)
        batch["ol"] = torch.LongTensor([[sos_idx] +  # add sos_idx
                                        output_generator["ol_idx"][:max_len - 4] +
                                        [pad_idx] * (max_len - 4 - len(output_generator["ol_idx"]))]).to(
            device)  # add pad_idx

        infer = self.predictor.infer(batch)
        pred = self.predictor.regression_head(infer["cls_feats"])
        pred = self.predictor.normalizer.decode(pred)
        pred = pred.detach().item()

        # get reward
        if self.criterion == "continuous-positive":
            reward = pred / 50
        elif self.criterion == "continuous-negative":
            reward = - pred / 50
        elif self.criterion > 0:
            if pred > self.criterion:
                reward = self.reward_positive
            else:
                reward = self.reward_negative
        else:
            if pred < self.criterion:  # negative
                reward = self.reward_positive
            else:
                reward = self.reward_negative
        return reward, pred
