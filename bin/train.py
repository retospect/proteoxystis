#! python

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from functools import lru_cache


def load_data():
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
    return metadata, seqs, output, seqs_test, output_test

# use caching for this method
@lru_cache(maxsize=1)
def find_torch_training_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    print("Using MPS")
    #    return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

def setup_model(seqs, output):
    # Setup model
    N = 1000  # num_samples_per_class
    D = seqs.shape[1]  # num_features
    C = output.shape[1]  # num_classes
    H = 100  # num_hidden_units
    inner_count = 9
    model = nn.Sequential(
        nn.Linear(D, H)
    )
    model.append(nn.ReLU())
    for i in range(inner_count):
        model.append(nn.Linear(H, H))
        model.append(nn.ReLU())
    model.append(nn.Linear(H, C))
    model.to(find_torch_training_device())
    print(model)
    return model

def train(model, seqs, output):
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train
    print("Training...", end="")

    in_data = torch.from_numpy(seqs).float().to(find_torch_training_device())
    out_data = torch.from_numpy(output).float().to(find_torch_training_device())

    for epoch in range(100):
        optimizer.zero_grad()
        output_pred = model(in_data  )
        loss = criterion(output_pred, out_data)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("epoch: ", epoch, " loss: ", loss.item())
    print("done")


def init_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def main():
    metadata, seqs, output, seqs_test, output_test = load_data()
    init_seeds(42)
    model = setup_model(seqs, output)
    train(model, seqs, output)

if __name__ == "__main__":
    main()

