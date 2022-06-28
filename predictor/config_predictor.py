import json
from sacred import Experiment

ex = Experiment("predictor")

mc_to_idx = json.load(open("data/mc_to_idx.json"))
topo_to_idx = json.load(open("data/topo_to_idx.json"))
char_to_idx = json.load(open("data/char_to_idx.json"))


@ex.config
def config():
    loss_name = "classification"  # classification
    threshold = 0.25 # classification for Topology rmsd
    seed = 0
    precision = 16
    exp_name = "predictor"
    dataset_dir = "data"

    # model setting
    char_dim = len(char_to_idx)
    mc_dim = len(mc_to_idx)
    topo_dim = len(topo_to_idx)
    embed_dim = 256
    hidden_dim = 256

    # run setting
    batch_size = 512
    per_gpu_batchsize = 256
    max_epochs = 50
    load_path = ""
    log_dir = "predictor/log"
    num_workers = 8
    num_nodes = 1
    num_gpus = 2

    # trainer setting
    resume_from = None
    val_check_interval = 1.0
    test_only = False

    # normalize (when regression)
    mean = 0.335  # None
    std = 0.157  # None


@ex.named_config
def env_ifactor():
    pass
