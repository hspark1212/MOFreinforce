![scheme_rl-01](https://user-images.githubusercontent.com/64190846/218362539-740997c9-d198-4e0a-89e0-3277c5b45a51.jpg)

# Reinforcement Learning Framework For MOFs

This package 

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
pip install -r requirements.txt
```

## Getting Started
In order to run the reinforcement learning framework, `predictor` (environment) and `generator` (agent) should be pre-trained. 

### [Predictor]()
<p align="left">
  <img src="https://user-images.githubusercontent.com/64190846/218362135-275e50d4-5a1b-4c5d-b8f3-3434193a3de9.jpg" width="300")
</p>

we provide predictors (in a format of .ckpt file) for DAC (CO2 Heat of adsorption and CO2/H2O selectivity) via figshare. 
The models were pre-trained with 30,000 structures with Wisdom calculation using RASPA code. 
The details of calculations are summarized in our paper. 

#### download pre-trained predictor
```angular2html
download ~~~ ### update
```

If you want to train the predictor for your own desired property, please refer to [predictor.md]().

### [Generator]()
<p align="left">
  <img src="https://user-images.githubusercontent.com/64190846/218362193-5540b285-d622-4698-8be9-f2bd789da264.jpg" width="700")
</p>

We provide a generator which selects a topology and a metal cluster, which are categorical variables, in order and then creates an organic linker represented by SELFIES string.
The generator was pre-trained with about 650,000 MOFs created by PORMAKE, which allows for generating feasible MOFs.
You can download the ckpt file of generator via Figshare.
```angular2html
download ~~~ ### update
```

### [Reinforcement Learning]()
