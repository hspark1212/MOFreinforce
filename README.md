# Reinforcement Learning Framework For MOFs
![scheme_rl-01](https://user-images.githubusercontent.com/64190846/218362539-740997c9-d198-4e0a-89e0-3277c5b45a51.jpg)
This package is a reinforcement learning framework for MOFs. 
The framework consists of `agent` and `environment` which are a generator and a predictor, respectively.
The agent takes an action (which is generating a MOF structure). 
This action is then evaluated in the environment by the predictor, which predicts the value of the property we are interested in. 
Based on the prediction, a reward is returned in form of an update to the agent to generate the next round of MOFs.

## Installation

### OS and hardware requirements

Linux : Ubuntu 20.04, 22.04

For optimal performance, we recommend running with GPUs

### Dependencies
```angular2html
python>=3.8
```

### Install
Please install pytorch (>= 1.12.0) according to your environments before installation of requirements.
```angular2html
$ git clone https://github.com/hspark1212/MOFreinforce.git
$ pip install -e .
```

## Getting Started

### [download pre-trained models]()

In order to train the reinforcement learning framework, the `predictor` (environment) and `generator` (agent) should be pre-trained.
So, we provide the pre-trained generator and predictors for DAC.

```angular2html
$ mofreinforce download default
```
Then, you can find the pre-trained generator and predictors in `mofreinforce/model`.

### [Predictor]()
<p align="left">
  <img src="https://user-images.githubusercontent.com/64190846/218362135-275e50d4-5a1b-4c5d-b8f3-3434193a3de9.jpg" width="500")
</p>

Once you download the pre-trained models, you can find the pre-trained predictors `model/preditor_qkh.ckpt` and `model/predictor_selectivity.ckpt` for CO2 heat of adsorption and CO2/H2O selectivity, respectively.

If you want to train the predictor for your own desired property, please refer to [predictor.md]().

### [Generator]()
<p align="left">
  <img src="https://user-images.githubusercontent.com/64190846/218362193-5540b285-d622-4698-8be9-f2bd789da264.jpg" width="800")
</p>

We provide a generator which selects a topology and a metal cluster, which are categorical variables, in order and then creates an organic linker represented by SELFIES string.
The generator was pre-trained with about 650,000 MOFs created by PORMAKE, which allows for generating feasible MOFs.
You can find the pre-trained generator at `model/generator.ckpt`.

### [Reinforcement Learning]()

(1) reinforcement learning with CO2 heat of adsorption
```angular2html
$ python mofreinforce/run_reinforce.py with v0_qkh
```

(2) reinforcement learning with CO2/H2O selectivity
```angular2html
$ python mofreinforce/run_reinforce.py with v1_selectivity
```

if you want to experiment with other parameters by modifying [`mofreinforce/reinforce/config_freinforce.py`](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/reinforce/config_reinforce.py) 