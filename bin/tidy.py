#! python

# Opens pdb.toml and converts it to a Python dictionary
# Then it processes each top level entry, and extracts the solvent condition.
# It shows a tqdl progress bar during processing
# There is a short option that only processes the first 1000 top level keys.
# These are not consistently written, so we have to try varios methods.
# It then writes a new training.toml file with the solvent conditions and the original sequence.

import toml
import tqdm
import sys
import gzip
import re

# Open the pdb.toml file
with gzip.open("pdb.toml.gz", "rt") as f:
    print("Loading data...")
    pdb = toml.load(f)

seqlen = []
# Open the training.toml file
with open("training.toml.gz", "wt") as f:
    # Write the header

    # Process each top level key
    for key in tqdm.tqdm(pdb.keys()):
        # Get the value
        value = pdb[key]
        l = len(value["sequence"])
        seqlen.append(l)
        solvent = None
        # Try to get the solvent condition
        crco = value.get("crystal_conditions")
        # Case insensitive match with regexp: "Conditions: .... PH: 7.0"
        if crco is not None:
            solvent = re.search("(?i)Conditions: (.*) PH:", crco)
        if solvent is None:
            continue
        solvent = pdb[key]["crystal_conditions"]

        sequence = pdb[key]["sequence"]
        f.write(f"[{key}]\n")
        f.write(f'sequence="{sequence}"\n')
        f.write(f'solvent="{solvent}"\n')

# print out average, stddev, mean for seqlen
import numpy as np

print(np.mean(seqlen), np.std(seqlen), np.median(seqlen))

# print max and min of sqlen
print(max(seqlen), min(seqlen))
