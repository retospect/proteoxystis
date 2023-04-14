# Datawrangling

The Big Index is availabe from the [FTP site](https://www.wwpdb.org/ftp/pdb-ftp-sites). Follow instructions there. Use

```
rsync -rptgoDvL --port=33444 rsync.rcsb.org::ftp_data/structures/all/pdb
```
It's faster with rsync v3 and above. Should take ~5 hours the first time.

Then:
- ```extract.py``` will reduce the data to just the sequence and all the crystallography data, in pdb.toml (15 minutes on M1)
- ```tidy.py``` will try and improve the data to actually extract fields from plain text. (a few seconds).

# Setup

## Install packages

```pip install -r requirements.txt```

Also, data is stored as large files. See https://git-lfs.com/
There are 2 datafiles stored in this fashion:
- pdb.toml.gz
- training.toml.gz

## Tools

- pdbfetch.py - Pulls data from PDB
- tidy.py - fixes sequences and agent names
- train.py - generates a new model based on training data
