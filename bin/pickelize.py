#! python

# reads the training.toml data, trims some items out, and pickles it for use in the training script

import pickle, os
import toml
from tqdm import tqdm
import numpy as np

# read the toml file
print("Reading toml file")
with open("training.toml", "r") as f:
    data = toml.load(f)

# the top level keys that are not metadata
pdb_keys = [k for k in data.keys() if k != "metadata"]

# find the keys in metadata:sig_key_count that have a value > 10 and add them to the set of useable keys
useable_keys = set()
for k in data["metadata"]["sig_key_count"].keys():
    if data["metadata"]["sig_key_count"][k] > 10:
        useable_keys.add(k)

# Remove the keys that are not useable from all the pdb_keys
# they may not exist
print("Removing stupid data")
for k in pdb_keys:
    del_keys = []
    for key in data[k]["values"].keys():
        if key not in useable_keys:
            del_keys.append(key)
    for key in del_keys:
        del data[k]["values"][key]
    # if values now has less than 4 keys, remove it from the pdb_keys
    if len(data[k]["values"].keys()) < 4:
        pdb_keys.remove(k)

# Print the length of the pdb_keys
print("after removing stupid data, this many remain ", len(pdb_keys))

# Make an alphabetic list with all the useable keys
useable_keys = sorted(list(useable_keys))
pdb_names = sorted(pdb_keys)


# find the numer of aminoacids used in the data
aminoacids = set()
for k in pdb_keys:
    for seq in data[k]["sequence"]:
        for aa in seq:
            aminoacids.add(aa)

# Find the longest sequence
longest_seq = 0
for k in pdb_keys:
    if len(data[k]["sequence"]) > longest_seq:
        longest_seq = len(data[k]["sequence"])

# alphabetize the aminoacids
aminoacids = sorted(list(aminoacids))

# make the metadata dict to be pickeld
metadata = {}
metadata["pdb_names"] = pdb_names
metadata["sig_keys"] = useable_keys
metadata["aminoacids"] = aminoacids

nr_of_distinct_aminoacids = len(aminoacids)

# make the numpy sequence array for the data. Each row is a pdb in alphabetical order, with one-hot encoding of the sequence
seqs = np.zeros(
    (len(pdb_names), longest_seq * nr_of_distinct_aminoacids), dtype=np.int8
)  # maybe use dtype=np.int8) to reduce memory
print("Input array size (pdb x one-hot sequence): ", seqs.shape)
print("Filling sequence array")
for i, k in tqdm(list(enumerate(pdb_names))):
    for j, aa in enumerate(data[k]["sequence"]):
        seqs[i, j * nr_of_distinct_aminoacids + aminoacids.index(aa)] = 1

# make the numpy array of output values. Each row is a pdb in alphabetical order.
# Each row has a value for each of the useable keys, and a one-hot encoder stating if the value was present in the data

output = np.zeros(
    (len(pdb_names), len(useable_keys) * 2), dtype=np.float16
)  # maybe use dtype=np.int8) to reduce memory
print("Output array size (pdb x one-hot output): ", output.shape)
print("Filling output array")
np.seterr(all="raise")
for i, k in tqdm(list(enumerate(pdb_names))):
    for j, key in enumerate(useable_keys):
        if key in data[k]["values"].keys():
            output[i, j * 2] = 1
            try:
                output[i, j * 2 + 1] = data[k]["values"][key]
            except FloatingPointError as someEx:
                print("YOLO")
                print(k)
                print("Error in ", k, " ", key)
                print(someEx)
                print(data[k]["values"][key])
                output[i, j * 2 + 1] = np.finfo(np.half).max
        else:
            output[i, j * 2] = 0
            output[i, j * 2 + 1] = 0

# pickle the data
print("Pickeling data")
with open("training_data.pickle", "wb") as f:
    pickle.dump((metadata, seqs, output), f)
