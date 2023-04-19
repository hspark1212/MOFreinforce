# Predictor
## 1. Prepare dataset

Once you download default data by running the following command,
```angular2html
$ mofreinforce download default
```

Then, examples of dataset of predictors will be downloaded at `data/dataset_predictor/qkh` or `data/dataset_predictor/selectivity`

The dataset directory should include `train.json`, `val.json` and `test.json`.

The json consists of names of structures (key) and dictionary of descriptions of structures (values).
The descriptions include `topo_name`, `mc_name`, `ol_name`, `ol_selfies`, `topo`, `mc`, `ol`, `target`.

(optional)
- `topo_name` : (string) name of topology. 
- `mc_name` : (string) name of metal cluster.
- `ol_name` : (string) name of organic linker.
(required)
The topologies, metal clusters and organic linkers should be vectorized. 
Given topologies and metal clusters are categorical variables, they need to be converted to idx. 
- `topo` : (int) index of topology.  The index can be found in `data/mc_to_idx.json`
- `mc` : (int) index of metal cluster. The index can be found in `data/topo_to_idx.json`
When it comes to organic linkers, it is represented by SELFIES. 
- `ol_selfies` : (string) SELFIES string of organic linker. The smiles can be converted into SELFIES using `sf.decode(SMILES)` in `libs/selfies`.
- `ol` : (list) a list of index of SELFIES string. The index of vocabulary of SELFIES can be found in `data/vocab_to_idx.json`
Finally, the target property you want to optimize should be defined.
- `target` : (float) target property.

## 2. Training predictor

Here is an example to train predictor for heat of adsorption in the `mofreinforce` directory
```angular2html
# python run_predictor.py with regression_qkh_round3
```
By modifying `predictor/config_predictor.py`, you can train your predictors.


