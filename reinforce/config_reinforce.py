from sacred import Experiment

ex = Experiment("reinforce")


@ex.config
def config():
    seed = 0
    exp_name = ""
    gpu_idx = 0

    # load predictor
    predictor_load_path = "predictor/logs/regression_vf_seed0_from_/version_0/checkpoints/epoch=38-step=2847.ckpt"
    mean = None
    std = None

    # load generator
    dataset_dir = "data/v4/dataset_generator"
    generator_load_path = "generator/logs/v0_seed0_from_/version_0/checkpoints/last.ckpt"

    # get_reward
    criterion = 0.8 # "continuous-positive" or "continuous-negative" or Float
    reward_positive = 10
    reward_negative = 3

    # REINFORCE
    lr = 1e-4
    decay_ol = 1.
    decay_topo = 1.
    exploit_ratio = 0.3 # ratio for exploitation
    """ to decrease effect of topology when calculating loss.
    Given topology is the discrete feature, its loss could much larger than the sum of organic linker.
    It gives rise to training failure.
    """
    scheduler = "warmup"  # warmup = get_linear_schedule_with_warmup, constant = get_constant_schedule
    early_stop = .6  # early_stop when the accuracy of scaffold is less than it.

    # Trainer
    max_epochs = 100
    batch_size = 16
    accumulate_grad_batches = 2
    num_nodes = 1
    num_devices = 2
    precision = 16
    resume_from = None
    limit_train_batches = 500
    limit_val_batches = 30
    val_check_interval = 1.0
    test_only = False
    load_path = ""
    gradient_clip_val = None

    # tensorboard
    log_dir = "reinforce/logs"


@ex.named_config
def test():
    exp_name = "test"
    limit_val_batches = 2
    lr = 1e-4

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -40.  # negative


"""
Q_kH
"""


@ex.named_config
def v0_qkh():
    exp_name = "v0_qkh"
    lr = 1e-5

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -40.  # negative


@ex.named_config
def v0_qkh_lr_e4():
    exp_name = "v0_qkh_lr_e4"
    lr = 1e-4

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -40.  # negative


@ex.named_config
def v0_qkh_lr_e4_constant():
    exp_name = "v0_qkh_lr_e4_constant"
    lr = 1e-4
    scheduler = "constant"

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -40.  # negative


@ex.named_config
def v1_qkh():
    exp_name = "v1_qkh"
    lr = 1e-5

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -30.  # negative


@ex.named_config
def v1_qkh_lr_e4():
    exp_name = "v1_qkh_lr_e4"
    lr = 1e-4

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -30.  # negative


@ex.named_config
def v1_qkh_lr_e4_gamma_98():
    exp_name = "v1_qkh_lr_e4_gamma_98"
    lr = 1e-4
    gamma = 0.98

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -30.  # negative


@ex.named_config
def v2_qkh():
    exp_name = "v2_qkh"
    lr = 1e-5

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -20.  # negative


@ex.named_config
def v2_qkh_gamma_90():
    exp_name = "v2_qkh_gamma_90"
    lr = 1e-5
    gamma = 0.9

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -20.  # negative


@ex.named_config
def v2_qkh_decay_topo_50():
    exp_name = "v2_qkh_decay_topo_50"
    lr = 1e-5
    decay_topo = 0.5

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -20.  # negative

@ex.named_config
def v2_qkh_decay_topo_0():
    exp_name = "v2_qkh_decay_topo_0"
    lr = 1e-5
    decay_topo = 0.

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -20.  # negative

# v3 starts the alternative training between topo and ol
@ex.named_config
def v3_qkh():
    exp_name = "v3_qkh"
    lr = 1e-5
    scheduler = "constant"

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -25.  # negative


@ex.named_config
def v3_qkh_test():
    exp_name = "v3_qkh_test"
    lr = 1e-5
    scheduler = "constant"

    # test
    accumulate_grad_batches = 1
    resume_from = "reinforce/logs/v3_qkh_seed0_from_/version_0/checkpoints/tmp_epoch_21.ckpt"

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -25.  # negative

@ex.named_config
def v4_qkh():
    exp_name = "v4_qkh"
    lr = 1e-5
    scheduler = "constant"

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = -25.  # negative
    reward_positive = 1.
    reward_negative = 0.

@ex.named_config
def v5_qkh():
    exp_name = "v5_qkh"
    lr = 1e-4
    scheduler = "constant"

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = "continuous-negative"

@ex.named_config
def v6_qkh():
    exp_name = "v6_qkh"

    # reinforce
    lr = 1e-4
    scheduler = "constant"
    decay_topo = 0.
    exploit_ratio = 0.5  # ratio for exploitation

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = "continuous-negative"

@ex.named_config
def v7_qkh():
    exp_name = "v7_qkh"
    max_epochs = 10

    # reinforce
    lr = 1e-4
    scheduler = "constant"
    exploit_ratio = .5  # ratio for exploitation
    early_stop = 0.  # early_stop when the accuracy of scaffold is less than it.

    # predictor
    predictor_load_path = "predictor/logs/regression_qkh_seed0_from_/version_0/checkpoints/epoch=21-step=2244.ckpt"
    mean = -19.418
    std = -9.162

    # reward
    criterion = "continuous-negative"



"""
Void Fraction
"""

"""
replace algorithm with REINFORCE in minimalRL
"""


@ex.named_config
def v4():
    exp_name = "v4"
    lr = 1e-5
    # get_reward
    criterion = 0.8
    reward_positive = 10
    reward_negative = 3


@ex.named_config
def v0():
    exp_name = "v0"


@ex.named_config
def v0_clip():
    exp_name = "v0_clip"
    gradient_clip_val = 0.5


@ex.named_config
def v0_clip_1():
    exp_name = "v0_clip_1"
    gradient_clip_val = 1.0


@ex.named_config
def v0_criterion_07():
    exp_name = "v0_criterion_07"
    # get_reward
    criterion = 0.7


@ex.named_config
def v0_norm_reward():
    exp_name = "v0_norm_reward"
    # get_reward
    reward_positive = 1.
    reward_negative = .3


@ex.named_config
def v0_gamma_098():
    exp_name = "v0_gamma_098"
    # get_reward
    gamma = .98
    reward_positive = 1.
    reward_negative = .3


@ex.named_config
def v0_gamma_09():
    exp_name = "v0_gamma_09"
    # get_reward
    gamma = .9
    reward_positive = 1.
    reward_negative = .3


@ex.named_config
def v0_gamma_09_v2():
    exp_name = "v0_gamma_09_v2"
    # get_reward
    gamma = .9
    reward_positive = 10
    reward_negative = 3


@ex.named_config
def v0_gamma_08():
    exp_name = "v0_gamma_08"
    # get_reward
    gamma = .8
    reward_positive = 1.
    reward_negative = .3


@ex.named_config
def v0_gamma_07():
    exp_name = "v0_gamma_07"
    # get_reward
    gamma = .7
    reward_positive = 1.
    reward_negative = .3


"""
v1: criterion 0.6
"""


@ex.named_config
def v1_gamma_09():
    exp_name = "v1_gamma_09"
    gamma = .9
    # get_reward
    criterion = 0.6


@ex.named_config
def v1_gamma_08():
    exp_name = "v1_gamma_08"
    gamma = .8
    # get_reward
    criterion = 0.6


@ex.named_config
def v1_gamma_07():
    exp_name = "v1_gamma_07"
    gamma = .7
    # get_reward
    criterion = 0.6


@ex.named_config
def v1_gamma_06():
    exp_name = "v1_gamma_06"
    gamma = .6
    # get_reward
    criterion = 0.6


@ex.named_config
def v2():
    exp_name = "v2"
    gamma = .9
    criterion = 0.7


@ex.named_config
def v2_lr_e5():
    exp_name = "v2_lr_e5"
    lr = 1e-5
    criterion = 0.7


@ex.named_config
def v2_lr_e6():
    exp_name = "v2_lr_e6"
    lr = 1e-6
    criterion = 0.7


@ex.named_config
def v3():
    exp_name = "v3"
    lr = 1e-5
    criterion = 0.8


@ex.named_config
def v3_gamma_80():
    exp_name = "v3_gamma_80"
    lr = 1e-5
    criterion = 0.8
    gamma = 0.8


@ex.named_config
def v3_lr_e6():
    exp_name = "v3_lr_e6"
    lr = 1e-6
    criterion = 0.8
