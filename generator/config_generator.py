import json
from sacred import Experiment

ex = Experiment("generator")

mc_to_idx = json.load(open("data/mc_to_idx.json"))
topo_to_idx = json.load(open("data/topo_to_idx.json"))

@ex.config
def config():
    seed = 0
    exp_name = "generator"
    log_dir = "generator/logs"
    loss_names = {"generator" : 1}

    # datamodule
    dataset_dir = "data/dataset_generator"
    batch_size = 256
    num_workers = 8 # recommend num_gpus * 4
    max_len = 128

    # transformer
    path_topo_to_idx = "data/topo_to_idx.json"
    path_mc_to_idx = "data/mc_to_idx.json"
    path_vocab = "data/vocab_to_idx.json"
    # input_dim = len(vocab_to_idx)
    # output_dim = len(vocab_to_idx)
    hid_dim = 256
    n_layers = 3
    n_heads = 8
    pf_dim = 512
    dropout = 0.1
    max_len = 128
    src_pad_idx = 0
    trg_pad_idx = 0

    # Trainer
    per_gpu_batchsize = 128
    num_nodes = 1
    num_devices = 2
    precision = 16
    resume_from = None
    val_check_interval = 1.0
    test_only = False
    load_path = ""
    gradient_clip_val = None

    # Optimizer Setting
    optim_type = "adam"  # adamw, adam, sgd (momentum=0.9)
    learning_rate = 5e-4
    weight_decay = 0
    decay_power = "constant"  # default polynomial decay, [cosine, constant, constant_with_warmup]
    max_epochs = 100
    max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
    warmup_steps = 0.0  # int or float ( max_steps * warmup_steps)
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

@ex.named_config
def v0():
    exp_name = "v0"

@ex.named_config
def test():
    exp_name = "test"

@ex.named_config
def v0_test():
    exp_name = "v0_test"
    load_path = "model/generator.ckpt"

    test_only=True
    num_devices=1

"""
old experiments
"""
@ex.named_config
def v0_grad_clip():
    exp_name = "v0_grad_clip"

    gradient_clip_val = 0.5




