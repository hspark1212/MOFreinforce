import json
from sacred import Experiment

ex = Experiment("predictor")

mc_to_idx = json.load(open("data/v3/mc_to_idx.json"))
topo_to_idx = json.load(open("data/v3/topo_to_idx.json"))
vocab_to_idx = json.load(open("data/v3/vocab_to_idx.json")) # vocab for selfies

def _loss_names(d):
    ret = {
        "trc": 0, # Topology Rmsd Regression
        "vfr": 0, # Void Fraction Regression
        "classification": 0,  # classification
        "regression": 0,  # regression
    }
    ret.update(d)
    return ret

@ex.config
def config():
    seed = 0
    exp_name = "predictor"

    dataset_dir = "data/v3"
    loss_names = _loss_names({})


    # model setting
    vocab_dim = len(vocab_to_idx)
    mc_dim = len(mc_to_idx)
    topo_dim = len(topo_to_idx)

    # transformer setting
    hid_dim = 256
    num_heads = 4 # hid_dim / 64
    num_layers = 4
    mlp_ratio = 4
    drop_rate = 0.1

    # run setting
    batch_size = 128
    per_gpu_batchsize = 64
    load_path = ""
    log_dir = "predictor/logs"
    num_workers = 8 # recommend num_gpus * 4
    num_nodes = 1
    num_gpus = 2
    precision = 16

    # downstream
    downstream = ""
    n_classes = 0

    # Optimizer Setting
    optim_type = "adamw"  # adamw, adam, sgd (momentum=0.9)
    learning_rate = 1e-4
    weight_decay = 1e-2
    decay_power = 1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    max_epochs = 50
    max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
    warmup_steps = 0.05  # int or float ( max_steps * warmup_steps)
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

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
def task_trc():
    exp_name = "task_trc"
    log_dir = "predictor/result_transformer"

    # trainer
    max_epochs = 50
    batch_size = 512
    per_gpu_batchsize = 16

    # model
    loss_names = _loss_names({"trc": 1})

@ex.named_config
def task_vfr():
    exp_name = "task_vfr"
    log_dir = "predictor/result_transformer"

    # trainer
    max_epochs = 50
    batch_size = 128
    per_gpu_batchsize = 16

    # model
    loss_names = _loss_names({"vfr": 1})

