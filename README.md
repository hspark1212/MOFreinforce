# Reinforcement Learning Framework For MOFs 🚀
![scheme_rl-01](https://user-images.githubusercontent.com/64190846/218362539-740997c9-d198-4e0a-89e0-3277c5b45a51.jpg)
This repository is a reinforcement learning framework for Metal-Organic Frameworks (MOFs), designed to generate MOF structures with user-defined properties. 🔍

The framework consists of two key components: the agent and the environment. The agent (i.e., generator) generates MOF structures by taking actions, which are then evaluated by the environment (i.e., predictor) to predict the properties of the generated MOFs. Based on the prediction, a reward is returned to the agent, which is then used to generate the next round of MOFs, continually improving the generation process. 

## Installation - Get started in minutes! 🌟

### OS and hardware requirements 
This package requires Linux Ubuntu 20.04 or 22.04. For optimal performance, we recommend running it with GPUs.

### Dependencies 
This package requires Python 3.8 or higher.

### Install 
To install this package, install **PyTorch** (version 1.12.0 or higher) according to your environment, and then follow these steps:

```
$ git clone https://github.com/hspark1212/MOFreinforce.git
$ cd MOFreinforce
$ pip install -e .
```

## Getting Started 💥

### [download pre-trained models](https://figshare.com/ndownloader/files/39472138)

To train the reinforcement learning framework, you'll need to use pre-trained predictors for DAC and pre-trained generator. You can download by running the following command in the `MOFreinforce/mofreinforce` directory:

```angular2html
$ mofreinforce download default
```
Once downloaded, you can find the pre-trained generator and predictor models in the `mofreinforce/model` directory, and the data files in the `mofreinforce/data` directory. 

### [Predictor](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/predictor)
<p align="left">
  <img src="https://user-images.githubusercontent.com/64190846/218362135-275e50d4-5a1b-4c5d-b8f3-3434193a3de9.jpg" width="500")
</p>

In the model directory, you'll find the pre-trained predictors `model/predictor/preditor_qkh.ckpt` and `model/predictor/predictor_selectivity.ckpt` for CO2 heat of adsorption and CO2/H2O selectivity, respectively. If you want to train your own predictor for your desired property, you can refer to [predictor.md](https://github.com/hspark1212/MOFreinforce/blob/master/predictor.md).

### [Generator](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/generator)
<p align="left">
  <img src="https://user-images.githubusercontent.com/64190846/218362193-5540b285-d622-4698-8be9-f2bd789da264.jpg" width="800")
</p>

The pre-trained generator, which selects a topology and a metal cluster and creates an organic linker represented by a SELFIES string, was pre-trained with about 650,000 MOFs created by PORMAKE, allowing for generating feasible MOFs. The pre-trained generator `model/generator/generator.ckpt` can be found in the model directory.

### [Reinforcement Learning](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/reinforce)
To implement reinforcement learning with CO2 heat of adsorption, run in the `mofreinforce` directory:
```angular2html
$ python run_reinforce.py with v0_qkh_round3
```

To implement reinforcement learning with CO2/H2O selectivity, run in the `mofreinforce` directory:
```angular2html
$ python run_reinforce.py with v1_selectivity_round3
```

You can experiment with other parameters by modifying the [`mofreinforce/reinforce/config_reinforce.py`](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/reinforce/config_reinforce.py) file. You can also train the reinforcement learning with your own pre-trained predictor to generate high-performing MOFs with your defined reward function.

### testing and construction of MOFs by PORMAKE 
To test the reinforcement learning, run in the `mofreinforce` directory: 
```angular2html
$ python run_reinforce.py with v0_qkh_round3 log_dir=test test_only=True load_path=model/reinforce/best_v0_qkh_round3.ckpt
```
The optimized generators for CO2 heat of adsorption and CO2/H2O selectivity can be found in the `mofreinforce/model` directory. 

The generated MOFs obtained from the test set (10,000 data) can be constructed by the [PORMAKE](https://github.com/Sangwon91/PORMAKE).
The details are summarized in [`tutorial.ipynb`](https://github.com/hspark1212/MOFreinforce/blob/master/mofreinforce/tutorial.ipynb) file.

## Contributing 🙌

Contributions are welcome! If you have any suggestions or find any issues, please open an issue or a pull request.

## License 📄

This project is licensed under the MIT License. See the `LICENSE` file for more information.
