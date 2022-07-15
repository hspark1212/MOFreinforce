import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from predictor.config_predictor import config, _loss_names
from predictor.module import Predictor

import torch
from generator.stackRNN import StackAugmentedRNN

from reinforce.reinforce import Reinforce
from reinforce.reward_functions import get_reward_trc, get_reward_vfr

from reinforce.config_reinforce import ex

@ex.automain
def main(_config):

    vocab_to_idx = json.load(open("data/v3/vocab_to_idx.json"))
    mc_to_idx = json.load(open("data/v3/mc_to_idx.json"))
    topo_to_idx = json.load(open("data/v3/topo_to_idx.json"))

    # load predictor for rsmd
    rmsd_config = config()
    rmsd_config["loss_names"] = _loss_names({"trc" : 1})
    rmsd_config["load_path"] = _config["load_path_rmsd"]
    trc_predictor = Predictor(rmsd_config)
    trc_predictor = trc_predictor.cuda()

    # load predictor for target
    target_config = config()
    target_config["loss_names"] = _loss_names({"vfr" : 1})
    target_config["load_path"] = _config["load_path_target"]
    vfr_predictor = Predictor(target_config)
    vfr_predictor = vfr_predictor.cuda()


    # load generator
    use_cuda = torch.cuda.is_available()
    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta
    generator = StackAugmentedRNN(vocab_to_idx=vocab_to_idx,
                                  input_size=len(vocab_to_idx), hidden_size=hidden_size,
                                  output_size=len(vocab_to_idx), layer_type=layer_type,
                                  n_layers=1, is_bidirectional=False, has_stack=True,
                                  stack_width=stack_width, stack_depth=stack_depth,
                                  use_cuda=use_cuda,
                                  optimizer_instance=optimizer_instance, lr=lr)
    generator.load_model(_config["load_path_generator"])

    # REINFORCE algorithm
    reinforce = Reinforce(generator=generator,
                          rmsd_predictor=trc_predictor,
                          target_predictor=vfr_predictor,
                          get_reward_rmsd=get_reward_trc,
                          get_reward_target=get_reward_vfr,
                          vocab_to_idx=vocab_to_idx,
                          mc_to_idx=mc_to_idx,
                          topo_to_idx=topo_to_idx,
                          emb_dim=_config["emb_dim"],
                          hid_dim=_config["hid_dim"],
                          config=_config,
                          )

    # training
    reinforce.train(n_batch=_config["n_batch"],
                    gamma=_config["gamma"],
                    n_iters=_config["n_iters"],
                    n_print=_config["n_print"],
                    topo_idx=_config["topo_idx"],
                    )

if __name__ == "__main__":
    main()


