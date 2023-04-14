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

# Open the pdb.toml file
with gzip.open("pdb.toml.gz", "rt") as f:
    print("Loading data...")
    pdb = toml.load(f)

# Open the training.toml file
with gzip.open("training.toml.gz", "wt") as f:
    # Write the header

    # Process each top level key
    for key in tqdm.tqdm(pdb.keys()):
        # Get the value
        value = pdb[key]

        #f.write(f"[{key}]\n")


