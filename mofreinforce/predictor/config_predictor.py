import json
from sacred import Experiment

ex = Experiment("predictor")

mc_to_idx = json.load(open("data/mc_to_idx.json"))
topo_to_idx = json.load(open("data/topo_to_idx.json"))
vocab_to_idx = json.load(open("data/vocab_to_idx.json"))  # vocab for selfies


def _loss_names(d):
    ret = {
        "regression": 0,  # regression
    }
    ret.update(d)
    return ret


@ex.config
def config():
    seed = 0
    exp_name = "predictor"

    dataset_dir = "###"
    loss_names = _loss_names({"regression": 1})

    # model setting
    max_len = 128  # cls + mc + topo + ol_len
    vocab_dim = len(vocab_to_idx)
    mc_dim = len(mc_to_idx)
    topo_dim = len(topo_to_idx)
    weight_loss = None

    # transformer setting
    hid_dim = 256
    num_heads = 4  # hid_dim / 64
    num_layers = 4
    mlp_ratio = 4
    drop_rate = 0.1

    # run setting
    batch_size = 128
    per_gpu_batchsize = 64
    load_path = ""
    log_dir = "predictor/logs"
    num_workers = 8  # recommend num_gpus * 4
    num_nodes = 1
    devices = 2
    precision = 16

    # downstream
    downstream = ""
    n_classes = 0
    threshold_classification = None

    # Optimizer Setting
    optim_type = "adamw"  # adamw, adam, sgd (momentum=0.9)
    learning_rate = 1e-4
    weight_decay = 1e-2
    decay_power = (
        1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    )
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

    # visualize attention score
    visualize_attention = False


@ex.named_config
def env_ifactor():
    pass


@ex.named_config
def regression_qkh_round1():
    exp_name = "regression_qkh_round1"
    dataset_dir = "data/dataset_predictor/qkh/round1"

    # trainer
    max_epochs = 50
    batch_size = 64
    per_gpu_batchsize = 16

    # normalize (when regression)
    mean = -20.331
    std = -10.383


@ex.named_config
def regression_qkh_round2():
    exp_name = "regression_qkh_round2"
    dataset_dir = "data/dataset_predictor/qkh/round2"

    # trainer
    max_epochs = 50
    batch_size = 64
    per_gpu_batchsize = 2

    # normalize (when regression)
    mean = -21.068
    std = 10.950


@ex.named_config
def regression_qkh_round3():
    exp_name = "regression_qkh_round3"
    dataset_dir = "data/dataset_predictor/qkh/round3"

    # trainer
    max_epochs = 50
    batch_size = 64
    per_gpu_batchsize = 2

    # normalize (when regression)
    mean = -21.810
    std = 11.452


"""
v1_selectivity
"""


@ex.named_config
def regression_selectivity_round1():
    exp_name = "regression_selectivity_round1"
    dataset_dir = "data/dataset_predictor/selectivity/round1"

    # trainer
    max_epochs = 50
    batch_size = 128
    per_gpu_batchsize = 16

    # normalize (when regression)
    mean = 1.872
    std = 1.922


@ex.named_config
def regression_selectivity_round2():
    exp_name = "regression_selectivity_round2"
    dataset_dir = "data/dataset_predictor/selectivity/round2"

    # trainer
    max_epochs = 50
    batch_size = 128
    per_gpu_batchsize = 16

    # normalize (when regression)
    mean = 2.085
    std = 2.052


@ex.named_config
def regression_selectivity_round3():
    exp_name = "regression_selectivity_round3"
    dataset_dir = "data/dataset_predictor/selectivity/round3"

    # trainer
    max_epochs = 50
    batch_size = 128
    per_gpu_batchsize = 16

    # normalize (when regression)
    mean = 2.258
    std = 2.128
