import json
from sacred import Experiment

ex = Experiment("predictor")

mc_to_idx = json.load(open("1_data_preprocessing/mc_to_idx.json"))
topo_to_idx = json.load(open("1_data_preprocessing/topo_to_idx.json"))
char_to_idx = json.load(open("1_data_preprocessing/char_to_idx.json"))


@ex.config
def config():
    seed = 42
    precision = 16
    exp_name = "predictor"
    dataset_dir = "...write your dataset directory..."

    # model setting
    char_dim = len(char_to_idx)
    mc_dim = len(mc_to_idx)
    topo_dim = len(topo_to_idx)
    embed_dim = 100
    hidden_dim = 256

    # run setting
    batch_size = 256
    per_gpu_batchsize = 128
    max_epochs = 100
    load_path = ""
    log_dir = "predictor/log"
    num_workers = 16
    num_nodes = 1
    num_gpus = 2

    # trainer setting
    resume_from = None
    val_check_interval = 1.0
    test_only = False


@ex.named_config
def env_ifactor():
    dataset_dir = "/home/hspark8/2_reinforce_mof/2_reinforcement_learning/1_data_preprocessing"
