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
            #print("ph is > 30, removing pdb entry ", k)
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
seqs = np.zeros( (len(pdb_names), longest_seq * nr_of_distinct_aminoacids), dtype=np.bool_)  
print("Input array size (pdb x one-hot sequence): ", seqs.shape)
print("Filling sequence array")
for i, k in tqdm(list(enumerate(pdb_names))):
    for j, aa in enumerate(data[k]["sequence"]):
        seqs[i, j * nr_of_distinct_aminoacids + aminoacids.index(aa)] = 1

# Convert all the ph fields to 10**ph
# That's done because pH is actually a logarithmic scale
#for k in pdb_keys:
#    if "ph" in data[k]["values"].keys():
#        data[k]["values"]["ph"] = 10 ** data[k]["values"]["ph"]

# Normalize all the output values.
# The values are normalized by dividing by the standard deviation of the values
# The new value is then divided by the mean of the new values
# The the total coorection factor is stored in the metadata dict
print("Normalizing output values")
metadata["correction_factor"] = {}
for key in tqdm(useable_keys):
    values = []
    value_owner = []
    for k in pdb_keys:
        if key in data[k]["values"].keys():
            values.append(data[k]["values"][key])
            value_owner.append(k)
    values = np.array(values)
    mean = np.mean(values)
    values = values/mean
    std = np.std(values)
    if std == 0:
        std = 1
    values = values / std
    values = 10* values
    # show the key, value name, new mean and new std
    # print(key, np.mean(values), np.std(values), min(values), max(values))
    correction_factor = mean*std/10
    metadata["correction_factor"][key] = correction_factor
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
for i, k in tqdm(list(enumerate(pdb_names))):
    for j, key in enumerate(useable_keys):
        if key in data[k]["values"].keys():
            output[i, j * 2] = 3 # means the value is present. We can adjust the importance of the correct value selection by changing this number
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
    (len(testpdb_names), longest_seq * nr_of_distinct_aminoacids), dtype=np.int8
)
seqs_train = np.zeros(
    (len(pdb_names) - len(testpdb_names), longest_seq * nr_of_distinct_aminoacids),
    dtype=np.int8,
)
# initialize output_test and output_train to the size they will have
output_test = np.zeros((len(testpdb_names), len(useable_keys) * 2))
output_train = np.zeros(
    (len(pdb_names) - len(testpdb_names), len(useable_keys) * 2))

for i, pdb in tqdm(list(enumerate(pdb_names))):
    if pdb in testpdb_names:
        pdb_names_test.append(pdb)
        seqs_test[len(pdb_names_test) - 1, :] = seqs[i, :]
        output_test[len(pdb_names_test) - 1, :] = output[i, :]
    else:
        pdb_names_train.append(pdb)
        seqs_train[len(pdb_names_train) - 1, :] = seqs[i, :]
        output_train[len(pdb_names_train) - 1, :] = output[i, :]

print("Test data size: ", seqs_test.shape)
print("Training data size: ", seqs_train.shape)

metadata["pdb_names_test"] = pdb_names_test
metadata["pdb_names_train"] = pdb_names_train

# pickle the data
print("Pickeling data")
with open("training_data.pickle", "wb") as f:
    pickle.dump(
        (
            metadata,
            seqs_train,
            output_train,
            seqs_test,
            output_test,
        ),
        f,
    )
