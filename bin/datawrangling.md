# Datawrangling

- [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/retospect/proteoxystis/blob/main/bin/phagetrix.ipynb)

TLDR:
- Install https://git-lfs.com/
- After checking out the repo, run ```pickelize.py``` to make the 9GB pickle of the one hot encoded data.
- then mess with the train*.py scripts to make the model better.

Then add a copy/variant of the training script and make it awesome.

## Example

### Predict a pdb entry:
```
train.py --predict 7usp         
Loading data...done
seqs.shape:    (75371, 86688)
output.shape:  (75371, 1464)
Preloading model from model.pt
Testing...Using CPU
done
Test loss: 7260.45166015625
Predicting 7usp
Delta for PDBID: 7usp
key                 actual     predicted  delta     
peg_length               -1.00     441.86 missing     
matthews                 51.12      39.18       -11.95
dtt_mm                   10.00      -1.00 excess      
sodium_citrate__mm      100.00      -1.00 excess      
peg_percent              -1.00       8.32 missing     
vm_a_pwr_da               2.52       2.28        -0.24
ph                        6.50       5.75        -0.75
```

## Tools

Tools, in order of dataflow path. 
(todo, add mermaid diagram or something)

TLDR:
- extract.py - pulls all data form PDB files (if you have pdb files) -> pdb.toml.gz
- tidy.py - parses pdb.toml.gz, extracts chemistry words etc -> training.toml
- pickelize.py - parses training.toml, normalizes and vectorizes data, training/test split, metadata -> training_data.pickle
- train.py - makes model and tests it and makes predictions

### extract.py - Pulls data from PDB [runs in 15min]
We've pre-extracted all the text we are likely to need into the ```pdb.toml.gz``` file. So the processing can start from that file. See the "Pulling all the PDB files" section below.
It also generates pdb.error.gz file with all the current errors to be improved.

The -q option runs it on only a few entries.

### tidy.py - fixes sequences and extracts agent vectors [runs in 1min]
Converts the sequences where possible - writes error.

creates tidy.toml.gz file, and a sampled tidy_s.toml file with only 10k entries, selected at random, for rapid testing. 

There's at least one test in the tests directory, run the tests with nose2 (just type nose2). 
More tests should be added, in particular from pdb-parsing-bugs.txt where we keep a running list of oddities. 

### train.py - generates a new model based on training data
We need to write this.

### Error file format

As we are continually improving the data pipeline here, a standardized error system should simplify the processing. 

Error files like pdb.err.gz and tidy.err.gz have the format:

```ERROR 01: [gaga2] This error happened.```

where 
- ```01``` is the sequence ID of errors (so its easy to grep|wc the lines).
- ```[gaga2]``` is the PDB identifier where this error occured
- ```This error happened.``` is a quick description of the error

### Error list and current count

When a file is left out, we try to categorize and identify root causes and fix the reasonable ones first. 

### Special codons

- ```u``` - UNK - Unknown.

## Pulling all the PDB files

The Big Index is availabe from the [FTP site](https://www.wwpdb.org/ftp/pdb-ftp-sites). Follow instructions there. 

```
rsync -rLP --port=33444 rsync.rcsb.org::ftp_data/structures/all/pdb data
```
It's faster with rsync v3 and above. Should take ~5 hours the first time.

Then:
- ```extract.py``` will reduce the data to just the sequence and all the crystallography data, in pdb.toml (15 minutes on M1)
- ```tidy.py``` will try and improve the data to actually extract fields from plain text. (a few seconds).

### Before pushing to github

- please run https://github.com/psf/black on your code so we don't have editor churn.

# Files

- pdb-few.toml.gz: the first ~100k lines of the pdb.toml file. For quick testing of the extraction procedure.

# Setup

## Install packages

```pip install -r requirements.txt```

Also, data is stored as large files. See https://git-lfs.com/
There are 2 datafiles stored in this fashion:
- pdb.toml.gz
- training.toml.gz

