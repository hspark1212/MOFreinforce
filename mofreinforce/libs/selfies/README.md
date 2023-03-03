This SELFIES directory is adapted from the "selfies" repository: https://github.com/aspuru-guzik-group/selfies.git

The following modifications have been made to the original repository to recognize dummy atoms "*" in SMILES, 
which were not provided for in the official "selfies" repository:

(1) In selfies.utils.smiles_utils.py, the following lines have been added to line 91:
```python
elif smiles[i] == "*":
    token = SMILESToken(bond_idx, i, i + 1,
                        SMILESTokenTypes.ATOM, smiles[i:i + 1])
```

(2) The dummy atom "*" has been added to `ORGANIC_SUBSET` in `selfies.constants.py`:
```python
ORGANIC_SUBSET = {"*", "B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
```

(3) "*" has been added to the regex rule in line 110 of `selfies.grammar_rules.py`:
```python
r"([A-Z][a-z]?|\*)"  # element symbol
```
