[![check](https://github.com/retospect/proteoxystis/actions/workflows/check.yml/badge.svg)](https://github.com/retospect/proteoxystis/actions/workflows/check.yml)
# Proteoxystis - Crystal solvent prediction for XRay Crystallography

**Under Construction**

Given the DNA sequence, predicts a set of likely solvents for that protein

## Determining structure

- install package with ```pip install protexystis```.
- run prediction with ```protex sequence.fa```. The output is a list of predicted solvents. 

## Training the model and updating the package

- clone the repo
- use the scripts in the ```run``` directory to 
  1. Get the data from PDB
  2. Sanitize the data
  3. Train the model
  4. Update the package with new training data
  5. Use maintainernotes to publish package
  
