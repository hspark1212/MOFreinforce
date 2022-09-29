import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter

from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import selfies as sf

class Reinforce(object):
    def __init__(self,
                 generator,
                 rmsd_predictor,
                 target_predictor,
                 get_reward_rmsd,
                 get_reward_target,
                 vocab_to_idx,
                 mc_to_idx,
                 topo_to_idx,
                 emb_dim=128,
                 hid_dim=128,
                 config=None,
                 ):
        """
        REINFORCE algorithm to generate MOFs to maximize reward functions (rsmsd and target)

        Parameters;
        generator_v0 (nn.Module): generator_v0
        rmsd_predictor (nn.Module): predictor for rmsd
        target_predictor (nn.Module): predictor for target
        get_reward_rmsd (function): get rewards for rmsd
        get_rewad_target (function): get rewards for target
        vocab_to_idx (dictionary) dictionary for token of selfies with index
        mc_to_idx (dictionary): dictionary for metal cluster with index
        topo_to_idx (dictionary): dictionary for topology with index
        emb_dim (int): dimension of embedding for metal cluster and topology
        hid_dim (int): dimension of hidden for metal culster and topology
        """
        super(Reinforce, self).__init__()
        self.generator = generator
        self.rmsd_predictor = rmsd_predictor
        self.target_predictor = target_predictor
        self.get_reward_rmsd = partial(get_reward_rmsd,
                                       criterion=config["criterion_rmsd"],
                                       reward_positive=config["reward_positive_rmsd"],
                                       reward_negative=config["reward_negative_rmsd"],
                                       )
        self.get_reward_target = partial(get_reward_target,
                                         criterion=config["criterion_target"],
                                         reward_positive=config["reward_positive_target"],
                                         reward_negative=config["reward_negative_target"],
                                         )

        self.vocab_to_idx = vocab_to_idx
        self.mc_to_idx = mc_to_idx
        self.topo_to_idx = topo_to_idx
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim

        self.rmsd_predictor.eval()
        self.target_predictor.eval()
        self.upgrade_generator()

        self.log_dir = f"{config['log_dir']}/{config['exp_name']}"
        if os.path.exists(self.log_dir) and config["test_only"] is not True:
            raise Exception(f"{self.log_dir} is already exist")

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def upgrade_generator(self):
        """
        add metal clutser and topology to generator_v0
        """
        self.generator.embedding_topo = nn.Embedding(len(self.topo_to_idx), self.emb_dim)
        self.generator.rc_mc_1 = nn.Linear(self.emb_dim, self.hid_dim)
        self.generator.rc_mc_2 = nn.Linear(self.hid_dim, len(self.mc_to_idx))

        self.generator.optimizer = self.generator.optimizer_instance(self.generator.parameters(),
                                                                     lr=1e-3,
                                                                     weight_decay=0.00001)
        self.generator = self.generator.cuda()

    def load_model(self, load_filename):
        self.generator.load_state_dict(torch.load(load_filename))

    def generate_mc(self, topo, return_loss=False):
        emb_topo = self.generator.embedding_topo(topo)
        logits = F.relu(self.generator.rc_mc_1(emb_topo))
        pred_mc = F.softmax(self.generator.rc_mc_2(logits), dim=-1)
        m = Categorical(pred_mc)
        mc = m.sample()
        loss_mc = torch.log(pred_mc[mc])

        if return_loss:
            return mc, loss_mc
        else:
            return mc

    def generate_ol(self):
        num_fail = 0
        # generate smiles
        valid_smiles = False
        while not valid_smiles:
            gen_sf = self.generator.evaluate()
            try:
                gen_sm = sf.decoder(gen_sf[5:-5])
                m = Chem.MolFromSmiles(gen_sm)
            except Exception:
                num_fail += 1
                continue

            # encodig_sf
            encoded_sf = sf.selfies_to_encoding(selfies=gen_sf,
                                                vocab_stoi=self.vocab_to_idx,
                                                enc_type="label")

            if m and 10 < len(encoded_sf) <= 100:
                valid_smiles = True
            else:
                num_fail += 1

        return encoded_sf, gen_sm, num_fail

    def init_generator(self):
        """
        Initialize stackRNN
        """

        hidden = self.generator.init_hidden()
        if self.generator.has_cell:
            cell = self.generator.init_cell()
            hidden = (hidden, cell)
        if self.generator.has_stack:
            stack = self.generator.init_stack()
        else:
            stack = None

        return hidden, stack

    def policy_gradient(self, n_batch, gamma, topo_idx=-1):
        """
        REINFORCE algorithm
        """
        self.generator.train()

        rl_loss = 0
        self.generator.optimizer.zero_grad()
        total_reward = 0
        num_fail = 0

        for n_epi in range(n_batch):
            # topology
            if topo_idx < 0:
                topo = torch.randint(0, len(self.topo_to_idx), size=(1,))[0].cuda()
            else:
                topo = torch.tensor(topo_idx).cuda()

            # metal cluster
            mc, loss_mc = self.generate_mc(topo, return_loss=True)
            rl_loss -= loss_mc

            # organic linker
            encoded_sf, _, n_f = self.generate_ol()
            num_fail += n_f

            # rmsd reward
            reward_rmsd, output_rmsd = self.get_reward_rmsd(topo, mc, encoded_sf, self.rmsd_predictor)

            # target reward
            reward_target, output_target = self.get_reward_target(topo, mc, encoded_sf, self.target_predictor)

            # REINFORCE algorithm
            discounted_reward = reward_rmsd + reward_target
            total_reward += reward_rmsd + reward_target

            # accumulate trajectory
            hidden, stack = self.init_generator()

            #  accumulate trajectory
            trajectory = torch.LongTensor(encoded_sf).cuda()

            for p in range(len(trajectory) - 1):

                output, hidden, stack = self.generator(trajectory[p], hidden, stack)
                log_probs = F.log_softmax(output, dim=-1)
                top_i = trajectory[p + 1]
                rl_loss -= (log_probs[0, top_i] * discounted_reward)
                discounted_reward = discounted_reward * gamma

        # backpropagation
        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch
        num_fail = num_fail / n_batch

        rl_loss.backward()
        self.generator.optimizer.step()

        return total_reward, rl_loss.item(), num_fail

    def test_estimate(self, n_to_generate, topo_idx=-1):
        with torch.no_grad():
            self.generator.evaluate()
            gen_mofs = []
            rewards = {"rmsd": [], "target": []}
            outputs = {"rmsd": [], "target": []}
            num_fail = 0

            for i in range(n_to_generate):
                # topology
                if topo_idx < 0:
                    topo = torch.randint(0, len(self.topo_to_idx), size=(1,))[0].cuda()
                else:
                    topo = torch.tensor(topo_idx).cuda()

                # metal cluster
                mc = self.generate_mc(topo)

                # organic linker
                encoded_sf, gen_sm, n_f = self.generate_ol()
                num_fail += n_f
                gen_mofs.append([topo.detach().item(), mc.detach().item(), gen_sm])

                # rmsd reward
                reward_rmsd, output_rmsd = self.get_reward_rmsd(topo, mc, encoded_sf, self.rmsd_predictor)

                # target reward
                reward_target, output_target = self.get_reward_target(topo, mc, encoded_sf, self.target_predictor)

                rewards["rmsd"].append(reward_rmsd)
                rewards["target"].append(reward_target)

                outputs["rmsd"].append(output_rmsd)
                outputs["target"].append(output_target)

        return rewards, outputs, num_fail, gen_mofs


    def write_logs_test_and_estimate(self, test_reward, test_output, test_num_fail, gen_mofs, n_iter, n_to_generate):
        reward_rmsd = np.array(test_reward["rmsd"])
        reward_target = np.array(test_reward["target"])
        output_rmsd = np.array(test_output["rmsd"])
        output_target = np.array(test_output["target"])

        # add scalar to log
        self.writer.add_scalar("test/reward/rmsd", reward_rmsd.mean(), n_iter)
        self.writer.add_scalar("test/reward/target", reward_target.mean(), n_iter)
        self.writer.add_scalar("test/output/rmsd", output_rmsd.mean(), n_iter)
        self.writer.add_scalar("test/output/target", output_target.mean(), n_iter)
        self.writer.add_scalar("test/num_fail", test_num_fail / n_to_generate, n_iter)
        # add histogram to log
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        data = [reward_rmsd, reward_target, output_rmsd, output_target]
        title = ["reward_rmsd", "reward_target", "output_rmsd", "output_target"]
        for i in range(4):
            ax = axes[i // 2, i % 2]
            ax.set_title(title[i])
            sns.histplot(data[i], ax=ax)
        self.writer.add_figure("test", fig, n_iter)

        # add image to log
        imgs = []
        for i, m in enumerate(gen_mofs):
            m = Chem.MolFromSmiles(m[2])
            img = MolToImage(m)
            img = np.array(img)
            img = torch.tensor(img)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        self.writer.add_images("test/gen_ol/", imgs, n_iter, dataformats="NHWC")


    def train(self,
              n_iters=10000,
              n_print=100,
              n_to_generate=200,
              n_batch=10,
              gamma=0.80,
              topo_idx=-1,
              ):
        """

        :param n_iters:  number of iterations
        :param n_print: number of iterations to print
        :param n_to_generate: how many mof will be generated when test_and_estimate
        :param n_batch: number of batch size
        :param gamma: discount ratio of REINFORCE
        :param topo_idx: if topo_idx < 0, randomly selecting topology
        :return: None
        """

        reward = 0
        loss = 0
        num_fail = 0
        callback = 0

        for n_iter in range(n_iters):

            batch_reward, batch_loss, batch_num_fail = self.policy_gradient(n_batch, gamma, topo_idx)
            reward += batch_reward
            loss += batch_loss
            num_fail += batch_num_fail

            if n_iter % n_print == 0:
                self.writer.add_scalar("epoch", n_iter, n_iter)
                if n_iter != 0:
                    print(f"########## iteration {n_iter} ##########")
                    print(
                        f"train | reward : {reward / n_print:.3f} ,"
                        f" loss : {loss / n_print:.3f} , "
                        f"num_fail : {num_fail / n_print:.3f}")
                    self.writer.add_scalar("train/loss", loss / n_print, n_iter)
                    self.writer.add_scalar("train/reward", reward / n_print, n_iter)
                    self.writer.add_scalar("train/num_fail", num_fail / n_print, n_iter)
                reward = 0
                loss = 0
                num_fail = 0

                test_reward, test_output, test_num_fail, gen_mofs = self.test_estimate(n_to_generate, topo_idx)

                test_reward_rmsd = np.mean(test_reward['rmsd'])
                test_reward_target = np.mean(test_reward["target"])
                total_test_reward = test_reward_rmsd + test_reward_target

                test_output_rmsd = np.mean(test_output['rmsd'])
                test_output_target = np.mean(test_output["target"])

                print(
                    f"test  | reward_rmsd : {test_reward_rmsd:.3f} , reward_target : {test_reward_target:.3f}, "
                    f"output_rmsd : {test_output_rmsd:.3f}, output_target : {test_output_target:.3f},"
                    f" num_fail : {test_num_fail}/{n_to_generate}")
                print(gen_mofs[:5])

                self.write_logs_test_and_estimate(test_reward,
                                                  test_output,
                                                  test_num_fail,
                                                  gen_mofs,
                                                  n_iter,
                                                  n_to_generate,
                                                  )

                if total_test_reward > callback:
                    path_ckpt = os.path.join(self.log_dir, f"reinforce_model_{n_iter}.ckpt")
                    torch.save(self.generator.state_dict(), path_ckpt)
                    print("model save !!!")
                    callback = total_test_reward
