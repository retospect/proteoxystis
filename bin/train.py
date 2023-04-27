#! python

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


print("Loading data...", end="")
with open("training_data.pickle", "rb") as f:
    metadata, seqs, output, seqs_test, output_test = pickle.load(f)
print("done")


# Data structure:
# metadata[pdb_names] - the names of the pdb entry names from the pdb database (e.g. 1agd)
# metadata[sig_keys]  - the names of all the output keys as they are in the output array
#                       ex: trisodium_citrate_dihydrate_mm
#                           meaning that the corresponding output array is the trisodium_citrate_dihydrate in milimolar
# metadata[aminoacids] - the aminoacid single letter ids in order used for the one-hot encoding
#                         "." means unknown fwiw, the other weird ones have been removed
# seqs   - an np.int8    array, rows = pdb entries, columns = aminoacids with one-hot encoding
# output - an np.float16 array, rows = pdb entries, columns = output floats

# Optimize for M1 and M2 when able.

# Print shapes of seqs and output
print("seqs.shape:   ", seqs.shape)
print("output.shape: ", output.shape)
