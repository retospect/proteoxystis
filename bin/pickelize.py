#! python

# reads the training.toml data, trims some items out, and pickles it for use in the training script

import pickle, os
import toml
from tqdm import tqdm
import torch
import gzip
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
for k in tqdm(pdb_keys):
    del_keys = []
    for key in data[k]["values"].keys():
        if key not in useable_keys:
            del_keys.append(key)
    for key in del_keys:
        del data[k]["values"][key]
    # if values now has less than 4 keys, remove it from the pdb_keys
    if len(data[k]["values"].keys()) < 4:
        pdb_keys.remove(k)
        continue

    # if ph is > 30, it's nonsense data. Delete the PDB entry
    # Check if the key exists first
    if "ph" in data[k]["values"].keys():
        if data[k]["values"]["ph"] > 30:
            # print("ph is > 30, removing pdb entry ", k)
            pdb_keys.remove(k)
            continue

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
metadata["sig_keys"] = useable_keys
metadata["aminoacids"] = aminoacids

nr_of_distinct_aminoacids = len(aminoacids)

# make the numpy sequence array for the data. Each row is a pdb in alphabetical order, with one-hot encoding of the sequence
seqs = np.zeros(
    (len(pdb_names), longest_seq * nr_of_distinct_aminoacids), dtype=np.float16
)
print("Input array size (pdb x one-hot sequence): ", seqs.shape)
print("Filling sequence array")
for i, k in tqdm(list(enumerate(pdb_names))):
    for j, aa in enumerate(data[k]["sequence"]):
        seqs[i, j * nr_of_distinct_aminoacids + aminoacids.index(aa)] = 1

# Normalize all the output values.
# The values are normalized by dividing by the standard deviation of the values
# The new value is then divided by the mean of the new values
# The the total coorection factor is stored in the metadata dict
print("Normalizing output values")
metadata["correction_factor"] = {}
metadata["correction_offset"] = {}
floatmax = np.finfo(np.float64).min * 0.1
floatmin = 0.0
for key in tqdm(useable_keys):
    values = []
    value_owner = []
    for k in pdb_keys:
        if key in data[k]["values"].keys():
            values.append(data[k]["values"][key])
            value_owner.append(k)
    values = np.array(values)
    maxi = np.max(values)
    mini = np.min(values)
    rang = maxi - mini
    if rang == 0:
        rang = 1
    factor = 1.0*rang/floatmax
    values = values / factor
    offset = np.min(values)
    values = values - offset
    print(values)
    # show the key, value name, new mean and new std
    # print(key, np.mean(values), np.std(values), min(values), max(values))
    correction_factor = factor
    metadata["correction_factor"][key] = correction_factor
    metadata["correction_offset"][key] = offset
    for i in range(len(values)):
        data[value_owner[i]]["values"][key] = values[i]
# make the numpy array of output values. Each row is a pdb in alphabetical order.
# Each row has a value for each of the useable keys, and a one-hot encoder stating if the value was present in the data

output = np.zeros(
    (len(pdb_names), len(useable_keys) * 2)
)  # maybe use dtype=np.int8) to reduce memory
print("Output array size (pdb x one-hot output): ", output.shape)
print("Filling output array")
np.seterr(all="raise")

# The relevant values array is used to mask out the values that are not interesting in the output
# Every odd row should have a 1. Because that's the categorization output.
# Also, every even row where we have a value should have a 1. Because that's the value output
relevant_values = np.zeros((len(pdb_names), len(useable_keys) * 2))
for i in range(len(useable_keys)):
    relevant_values[:, i * 2] = 1

# the magic number for "this category is present"
# increase for more category fit
metadata["category_is_present_magic_number"] = floatmax

for i, k in tqdm(list(enumerate(pdb_names))):
    for j, key in enumerate(useable_keys):
        if key in data[k]["values"].keys():
            output[i, j * 2] = metadata["category_is_present_magic_number"]
            relevant_values[
                i, j * 2 + 1
            ] = 1  # This value is relevant, don't mask it out.
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

# Write the testdata to test.toml file
# do this rarely if ever lest we learn about the training data
# Regenerate test data
if False:
    # generate a list of test data, pick 25% of the pdb names at random
    testpdb_names = np.random.choice(
        pdb_names, int(len(pdb_names) * 0.25), replace=False
    )
    print("Writing test data to test.toml")
    with open("test.toml", "w") as f:
        f.write("# Test data records. These are sacred. Do not use for training.\n")
        f.write("[testdata]\n")
        f.write("pdb_names = [")
        for i, pdb in enumerate(testpdb_names):
            f.write('"' + pdb + '"')
            if i != len(testpdb_names) - 1:
                f.write(", ")
        f.write("]\n")

# read the test.toml file into the testpdb_names set
print("Reading test data from test.toml")
with open("test.toml", "r") as f:
    testdata = toml.load(f)
# make the set
testpdb_names = set(testdata["testdata"]["pdb_names"])

# separate the test data from the training data
# This should make the
# - np arrays seqs_test and seqs_train from seqs
# - np arrays output_test and output_train from output
# - lists pdb_names_test and pdb_names_train from pdb_names
# - list useable_keys_test and useable_keys_train from useable_keys
print("Separating test data from training data")
# initialize pdb_names_test and pdb_names_train
pdb_names_test = []
pdb_names_train = []
# initialize seqs_test and seqs_train to the size they will have
seqs_test = np.zeros(
    (len(testpdb_names), longest_seq * nr_of_distinct_aminoacids), dtype=np.float16
)
seqs_train = np.zeros(
    (len(pdb_names) - len(testpdb_names), longest_seq * nr_of_distinct_aminoacids),
    dtype=np.float16,
)
# initialize output_test and output_train to the size they will have
output_test = np.zeros((len(testpdb_names), len(useable_keys) * 2))
output_train = np.zeros((len(pdb_names) - len(testpdb_names), len(useable_keys) * 2))

relevant_values_test = np.zeros((len(testpdb_names), len(useable_keys) * 2))
relevant_values_train = np.zeros(
    (len(pdb_names) - len(testpdb_names), len(useable_keys) * 2)
)

# fill the test and train data
for i, pdb in tqdm(list(enumerate(pdb_names))):
    if pdb in testpdb_names:
        pdb_names_test.append(pdb)
        seqs_test[len(pdb_names_test) - 1, :] = seqs[i, :]
        output_test[len(pdb_names_test) - 1, :] = output[i, :]
        relevant_values_test[len(pdb_names_test) - 1, :] = relevant_values[i, :]
    else:
        pdb_names_train.append(pdb)
        # If the next line fails, the test set needs to be regenerated
        # just a few lines up
        seqs_train[len(pdb_names_train) - 1, :] = seqs[i, :]  # Regenerate test data
        output_train[len(pdb_names_train) - 1, :] = output[i, :]
        relevant_values_train[len(pdb_names_train) - 1, :] = relevant_values[i, :]

print("Test data size: ", seqs_test.shape)
print("Training data size: ", seqs_train.shape)

# convert seqs_test, seqs_train, output_test and output_train to pytorch tensors
# (But storing the pytorch tensor in the pickle is maddening slow)
if False:
    print("Converting to pytorch tensors")
    seqs_test = torch.from_numpy(seqs_test).float()
    seqs_train = torch.from_numpy(seqs_train).float()
    output_test = torch.from_numpy(output_test).float()
    output_train = torch.from_numpy(output_train).float()


metadata["pdb_names_test"] = pdb_names_test
metadata["pdb_names_train"] = pdb_names_train

# pickle the data
print("Pickeling data")
# gzip makes this way slow?
with open("training_data.pickle", "wb") as f:
    pickle.dump(
        (
            metadata,
            seqs_train,
            output_train,
            seqs_test,
            output_test,
            relevant_values_train,
            relevant_values_test,
        ),
        f,
    )
