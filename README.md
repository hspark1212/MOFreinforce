# Reinforcement Learning Framework For MOFs
![scheme_rl-01](https://user-images.githubusercontent.com/64190846/218362539-740997c9-d198-4e0a-89e0-3277c5b45a51.jpg)
This repository is a reinforcement learning framework for MOFs (Metal-Organic Frameworks) that can generate MOF structures with user-desired properties.
The framework consists of `agent` and `environment` which are a generator and a predictor, respectively.
The agent takes an action (generating a MOF structure) which is evaluated in the environment by the predictor. 
Based on the prediction, a reward is returned in form of an update to the agent to generate the next round of MOFs.

## Installation

### OS and hardware requirements
This package requires Linux Ubuntu 20.04 or 22.04. For optimal performance, we recommend running it with GPUs.

### Dependencies
This package requires Python 3.8 or higher.

### Install
To install this package, please install PyTorch (version 1.12.0 or higher) according to your environment, and then follow these steps:
```
$ git clone https://github.com/hspark1212/MOFreinforce.git
$ pip install -e .
```

## Getting Started

### [download pre-trained models](https://figshare.com/articles/dataset/default_tar_gz_for_MOFreinforce/22085948)

To train the reinforcement learning framework, the predictor (environment) and generator (agent) should be pre-trained. 
We provide pre-trained generators and predictors for DAC, which you can download by running the following command:

```angular2html
$ mofreinforce download default
```
Then, the pre-trained generator and predictors can be found in the mofreinforce/model directory.

### [Predictor](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/predictor)
<p align="left">
  <img src="https://user-images.githubusercontent.com/64190846/218362135-275e50d4-5a1b-4c5d-b8f3-3434193a3de9.jpg" width="500")
</p>

The pre-trained predictors `model/preditor_qkh.ckpt` and `model/predictor_selectivity.ckpt` for CO2 heat of adsorption and CO2/H2O selectivity, respectively, can be found in the `model` directory. 
If you want to train the predictor for your own desired property, please refer to [predictor.md](https://github.com/hspark1212/MOFreinforce/blob/master/predictor.md).

### [Generator](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/generator)
<p align="left">
  <img src="https://user-images.githubusercontent.com/64190846/218362193-5540b285-d622-4698-8be9-f2bd789da264.jpg" width="800")
</p>

The pre-trained generator, which selects a topology and a metal cluster, and creates an organic linker represented by SELFIES string, was pre-trained with about 650,000 MOFs created by PORMAKE, allowing for generating feasible MOFs. 
The pre-trained generator `model/generator.ckpt` can be found at the `model` directory.

### [Reinforcement Learning](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/reinforce)
To implement reinforcement learning with CO2 heat of adsorption, run the following command:
```angular2html
$ python mofreinforce/run_reinforce.py with v0_qkh
```

To implement reinforcement learning with CO2/H2O selectivity, run the following command:
```angular2html
$ python mofreinforce/run_reinforce.py with v1_selectivity
```

You can experiment with other parameters by modifying the [`mofreinforce/reinforce/config_reinforce.py`](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/reinforce/config_reinforce.py) file.

you can train the reinforcement learning with your own pre-trained predictor to generate high-performing MOFs with your defined reward function.

### testing and construction of MOFs by PORMAKE 
To test the reinforcement learning, run the following command: 
```angular2html
$ python run_reinforce.py with v0_qkh log_dir=test test_only=True load_path=model/reinforced_qkh.ckpt
```
The optimized generators for CO2 heat of adsorption and CO2/H2O selectivity can be found in the `model` directory.

The generated MOFs obtained from the test set (10,000 data) will be constructed by the [PORMAKE](https://github.com/Sangwon91/PORMAKE).
The details are summarized in [`tutorial.ipynb`](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/tutorial.ipynb) file.


using the model  

## Contributing

Contributions are welcome! If you have any suggestions or find any issues, please open an issue or a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.