import json
from sacred import Experiment

ex = Experiment("predictor")

mc_to_idx = json.load(open("data/v2/mc_to_idx.json"))
topo_to_idx = json.load(open("data/v2/topo_to_idx.json"))
vocab_to_idx = json.load(open("data/v2/vocab_to_idx.json"))


@ex.config
def config():
    seed = 0
    exp_name = "predictor"
    target = "rmsd" # rmsd, vf

    dataset_dir = "data/v2"
    loss_name = "classification"  # regression, classification
    threshold = None # binary classification for Topology rmsd (0.25)
    precision = 16

    # model setting
    vocab_dim = len(vocab_to_idx)
    mc_dim = len(mc_to_idx)
    topo_dim = len(topo_to_idx)
    embed_dim = 256
    hidden_dim = 256

    # run setting
    batch_size = 512
    per_gpu_batchsize = 256
    max_epochs = 50
    load_path = ""
    log_dir = "predictor/logs"
    num_workers = 8
    num_nodes = 1
    num_gpus = 2

    # trainer setting
    resume_from = None
    val_check_interval = 1.0
    test_only = False

    # normalize (when regression)
    mean = None
    std = None


@ex.named_config
def env_ifactor():
    pass

@ex.named_config
def rmsd_regression():
    exp_name = "rmsd_regression"
    target = "rmsd"

    # regression
    loss_name = "regression"
    # mean = 0.335
    # std = 0.157


@ex.named_config
def rmsd_classification():
    exp_name = "rmsd_classification"
    target = "rmsd"

    # classification
    loss_name = "classification"
    threshold = 0.25 # binary classification for Topology rmsd


@ex.named_config
def vf_regression():
    exp_name = "vf_regression"
    target="vf"

    # regression
    loss_name = "regression"

    # run setting
    batch_size = 128
    per_gpu_batchsize = 64


