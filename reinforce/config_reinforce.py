from sacred import Experiment

ex = Experiment("reinforce")


@ex.config
def config():
    seed = 0
    exp_name = ""
    gpu_idx = 0

    # load predictor  - rmsd
    load_path_rmsd = "predictor/result_transformer/task_trc_seed0_from_/version_0/checkpoints/epoch=29-step=990.ckpt"

    # load predictor - target
    load_path_target = "predictor/result_transformer/task_vfr_seed0_from_/version_0/checkpoints/epoch=41-step=3360.ckpt"

    # load generator
    load_path_generator = "generator/saved_model_selfies_2/generator_1000000.ch"

    # get_reward_rmsd
    criterion_rmsd = 0
    reward_positive_rmsd = 11
    reward_negative_rmsd = 1

    # get_reward_target
    criterion_target = 0.90
    reward_positive_target = 11
    reward_negative_target = 1

    # REINFORCE class
    emb_dim = 128
    hid_dim = 128
    test_only = False

    # train
    n_iters = 2000
    n_print = 100
    n_to_generate = 200  # test_estimate
    n_batch = 10
    gamma = 0.98
    topo_idx = -1 # set topology idx or random topology when -1

    # tensorboard
    save_dir = None
    log_dir = "reinforce/logs"

"""
v1 
 - reward : target O, rmsd X
 - fixed topo_idx
"""
@ex.named_config
def v1():
    exp_name = "v1"

    # get_reward_rmsd
    reward_positive_rmsd = 0
    reward_negative_rmsd = 0

    # train
    topo_idx = 0

"""
v2
 - reward : target X, rmsd O
 - fixed topo_idx
"""
@ex.named_config
def v2():
    exp_name = "v2"

    # get_reward_target
    reward_positive_target = 0
    reward_negative_target = 0

    # train
    topo_idx = 0


"""
v3
 - reward : target O, rmsd O
 - fixed topo_idx
"""
@ex.named_config
def v3():
    exp_name = "v3"

    # train
    topo_idx = 0


"""
v4
 - reward : target O, rmsd X
 - random topo_idx
"""
@ex.named_config
def v4():
    exp_name = "v4"

    # get_reward_rmsd
    reward_positive_rmsd = 0
    reward_negative_rmsd = 0

    # train
    topo_idx = -1

"""
v5
 - reward : target X, rmsd O
 - random topo_idx
"""
@ex.named_config
def v5():
    exp_name = "v5"

    # get_reward_target
    reward_positive_target = 0
    reward_negative_target = 0

    # train
    topo_idx = -1


"""
v6
 - reward : target O, rmsd O
 - random topo_idx
"""
@ex.named_config
def v6():
    exp_name = "v6"

    # train
    topo_idx = -1

