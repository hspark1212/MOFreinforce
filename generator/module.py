import os
import json
import torch

import selfies as sf
from rdkit import Chem

from pytorch_lightning import LightningModule

from generator import objectives
from generator.transformer import Transformer
from utils import module_utils

from utils.metrics import metrics_generator


class Generator(LightningModule):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.save_hyperparameters()

        self.max_len = config["max_len"]
        # topo
        path_topo_to_idx = config["path_topo_to_idx"]
        self.topo_to_idx = json.load(open(path_topo_to_idx))
        self.idx_to_topo = {v: k for k, v in self.topo_to_idx.items()}
        # mc
        path_mc_to_idx = config["path_mc_to_idx"]
        self.mc_to_idx = json.load(open(path_mc_to_idx))
        self.idx_to_mc = {v: k for k, v in self.mc_to_idx.items()}
        # ol
        path_vocab = config["path_vocab"]
        self.vocab_to_idx = json.load(open(path_vocab))
        self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}


        self.transformer = Transformer(
            input_dim=len(self.vocab_to_idx),
            output_dim=len(self.vocab_to_idx),
            topo_dim=len(self.topo_to_idx),
            mc_dim=len(self.mc_to_idx),
            hid_dim=config["hid_dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            pf_dim=config["pf_dim"],
            dropout=config["dropout"],
            max_len=config["max_len"],
            src_pad_idx=config["src_pad_idx"],
            trg_pad_idx=config["trg_pad_idx"],
        )

        module_utils.set_metrics(self)
        module_utils.set_task(self)
        # ===================== load model ======================

        if config["load_path"] != "":
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

    def infer(self, batch):
        src = batch["encoded_input"]  # [B, max_len]

        tgt_input = batch["encoded_output"][:, :-1]  # [B, seq_len-1]
        tgt_label = batch["encoded_output"][:, 1:]  # [B, seq_len-1]

        # get mask
        out = self.transformer(src, tgt_input)  # [B, seq_len-1, vocab_dim]

        out.update({
            "src": src,
            "tgt": batch["encoded_output"],
            "tgt_label": tgt_label,
        })

        return out

    def evaluate(self, src, max_len=128):
        """
        src : torch.LongTensor [B=1, seq_len]
        """
        vocab_to_idx = self.vocab_to_idx
        src_mask = self.transformer.make_src_mask(src)

        # get encoded src
        enc_src = self.transformer.encoder(src, src_mask)  # [B=1, seq_len, hid_dim]

        # get target
        trg_indexes = [vocab_to_idx["[SOS]"]]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(src.device)
            trg_mask = self.transformer.make_trg_mask(trg_tensor)  # [B=1, 1, seq_len, seq_len]

            output = self.transformer.decoder(trg_tensor, enc_src, trg_mask, src_mask)  # [B=1, seq_len, vocab_dim]

            if "output_ol" in output.keys():
                out = output["output_ol"]
                pred_token = out.argmax(-1)[:, -1].item()
            elif "output_mc" in output.keys():
                out = output["output_mc"]
                pred_token = out.argmax(-1).item()
            else:
                out = output["output_topo"]
                pred_token = out.argmax(-1).item()
            trg_indexes.append(pred_token)

            if i > 3 and pred_token == vocab_to_idx["[EOS]"]:
                break

        # get topo
        topo_idx = trg_indexes[1]
        topo = self.idx_to_topo[topo_idx]
        # get mc
        mc_idx = trg_indexes[2]
        mc = self.idx_to_mc[mc_idx]
        # get ol
        ol_idx = trg_indexes[3:]
        ol_tokens = [self.idx_to_vocab[idx] for idx in ol_idx]
        # convert to selfies and smiles
        gen_sf = None
        gen_sm = None
        try:
            gen_sf = "".join(ol_tokens[:-1])  # remove EOS token
            gen_sm = sf.decoder(gen_sf)
            m = Chem.MolFromSmiles(gen_sm)
            gen_sm = Chem.MolToSmiles(m) # canonical smiles
        except Exception as e:
            print(e)
            pass

        ret = {
            "topo" : topo,
            "mc" : mc,
            "topo_idx" : topo_idx,
            "mc_idx" : mc_idx,
            "ol_idx" : ol_idx,
            "gen_sf" : gen_sf,
            "gen_sm" : gen_sm,
        }
        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))

        if "generator" in self.current_tasks:
            ret.update(objectives.compute_generator(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        module_utils.set_task(self)
        ret = self(batch)
        total_loss = sum([v for k, v in ret.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outputs):
        module_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        module_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, output):
        module_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        module_utils.set_task(self)
        return batch

    def test_epoch_end(self, batches):
        module_utils.epoch_wrapup(self)
        src = torch.concat([b["encoded_input"] for b in batches], dim=0)
        metrics_generator(self, src, "test")

    def configure_optimizers(self):
        return module_utils.set_schedule(self)
