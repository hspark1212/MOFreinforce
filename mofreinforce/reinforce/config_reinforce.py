from sacred import Experiment

ex = Experiment("reinforce")


@ex.config
def config():
    seed = 0
    exp_name = ""
    gpu_idx = 0

    # load predictor
    predictor_load_path = ["model/predictor_qkh.ckpt"]
    mean = [None]
    std = [None]

    # get reward
    threshold = False
    """
    (1) threshold == True
     if |pred| >= |reward_max|, reward = 1, else : reward = |pred| / |reward_max|
    (2) threshold == False
    reward = |pred| / |reward_max|
    """
    reward_max = [1.0]  # if you want to normalize by max reward.

    # load generator
    dataset_dir = "data/dataset_generator"
    generator_load_path = "model/generator/generator.ckpt"

    # REINFORCE
    lr = 1e-4
    decay_ol = 1.0
    decay_topo = 1.0
    scheduler = "constant"  # warmup = get_linear_schedule_with_warmup, constant = get_constant_schedule
    early_stop = 0.5  # early_stop when the accuracy of scaffold is less than it.
    ratio_exploit = 0.0  # ratio for exploitation (freeze)
    """ to decrease effect of topology when calculating loss.
    Given topology is the discrete feature, its loss could much larger than the sum of organic linker.
    It gives rise to training failure.
    """
    ratio_mask_mc = 0.0  # ratio for masking mc of input_src

    # Trainer
    max_epochs = 20
    batch_size = 16
    accumulate_grad_batches = 2
    devices = 1
    num_nodes = 1
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


"""
v0 Q_kH
"""


@ex.named_config
def v0_scratch():
    exp_name = "v0_scratch"
    test_only = True

    # reward
    reward_max = [-60.0]

    # predictor
    predictor_load_path = ["model/predictor/best_predictor_v0_qkh_round1.ckpt"]
    mean = [-20.331]
    std = [-10.383]


@ex.named_config
def v0_qkh_round1():
    """
    omit mc in the input
    """
    exp_name = "v0_qkh_round1"
    max_epochs = 20

    # reward
    reward_max = [-60.0]

    # reinforce
    early_stop = 0.5  # early_stop when the accuracy of scaffold is less than it.
    ratio_exploit = 0.5  # ratio for exploitation
    ratio_mask_mc = 0.5  # ratio for masking mc of input_src

    # predictor
    predictor_load_path = ["model/predictor/best_predictor_v0_qkh_round1.ckpt"]
    mean = [-20.331]
    std = [-10.383]


@ex.named_config
def v0_qkh_round2():
    """
    omit mc in the input
    """
    exp_name = "v0_qkh_round2"
    max_epochs = 20

    # reward
    reward_max = [-60.0]

    # reinforce
    early_stop = 0.5  # early_stop when the accuracy of scaffold is less than it.
    ratio_exploit = 0.5  # ratio for exploitation
    ratio_mask_mc = 0.5  # ratio for masking mc of input_src

    # predictor
    predictor_load_path = ["model/predictor/best_predictor_v0_qkh_round2.ckpt"]
    mean = [-21.068]
    std = [10.950]


@ex.named_config
def v0_qkh_round3():
    """
    omit mc in the input
    """
    exp_name = "v0_qkh_round3"
    max_epochs = 20

    # reward
    reward_max = [-60.0]

    # reinforce
    early_stop = 0.5  # early_stop when the accuracy of scaffold is less than it.
    ratio_exploit = 0.5  # ratio for exploitation
    ratio_mask_mc = 0.5  # ratio for masking mc of input_src

    # predictor
    predictor_load_path = ["model/predictor/best_predictor_v0_qkh_round3.ckpt"]
    mean = [-21.810]
    std = [11.452]


"""
Selectivity (v1)
"""


@ex.named_config
def v1_scratch():
    exp_name = "v1_scratch"
    test_only = True

    # reward
    reward_max = [10.0]

    # predictor
    predictor_load_path = ["model/predictor/best_predictor_v1_selectivity_round1.ckpt"]
    mean = [1.871]
    std = [1.922]


@ex.named_config
def v1_selectivity_round1():
    exp_name = "v1_selectivity_round1"
    max_epochs = 20

    # reward
    reward_max = [10.0]

    # reinforce
    early_stop = 0.5  # early_stop when the accuracy of scaffold is less than it.
    ratio_exploit = 0.5  # ratio for exploitation
    ratio_mask_mc = 0.5  # ratio for masking mc of input_src

    # predictor
    predictor_load_path = ["model/predictor/best_predictor_v1_selectivity_round1.ckpt"]
    mean = [1.871]
    std = [1.922]


@ex.named_config
def v1_selectivity_round2():
    exp_name = "v1_selectivity_round2"
    max_epochs = 20

    # reward
    reward_max = [10.0]

    # reinforce
    early_stop = 0.5  # early_stop when the accuracy of scaffold is less than it.
    ratio_exploit = 0.5  # ratio for exploitation
    ratio_mask_mc = 0.5  # ratio for masking mc of input_src

    # predictor
    predictor_load_path = ["model/predictor/best_predictor_v1_selectivity_round2.ckpt"]
    mean = [2.085]
    std = [2.052]


@ex.named_config
def v1_selectivity_round3():
    exp_name = "v1_selectivity_round3"
    max_epochs = 20

    # reward
    reward_max = [10.0]

    # reinforce
    early_stop = 0.5  # early_stop when the accuracy of scaffold is less than it.
    ratio_exploit = 0.5  # ratio for exploitation
    ratio_mask_mc = 0.5  # ratio for masking mc of input_src

    # predictor
    predictor_load_path = ["model/predictor/best_predictor_v1_selectivity_round3.ckpt"]
    mean = [2.258]
    std = [2.128]
