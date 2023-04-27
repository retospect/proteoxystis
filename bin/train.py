#! python

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import argparse
import copy
from encoding import get_active_values
from functools import lru_cache


def parse_commandline():
    """Setup commandline arguments and parse them."""
    parser = argparse.ArgumentParser(description="Train a neural network on the data.")
    # preload this model file. Takes a filename. Verify that the file exists.
    # default: model.pt
    # check if file model.pt exists
    # if it does, make it the default
    load_model_default = None
    try:
        preferred_model_name = "model.pt"
        with open(preferred_model_name, "rb") as f:
            load_model_default = preferred_model_name
    except FileNotFoundError:
        pass
    parser.add_argument(
        "--preload",
        type=str,
        default=load_model_default,
        help="preload a model from this model file",
    )

    default_epochs = 10000
    # just do testing
    parser.add_argument("--test", action="store_true", help="just test the model")

    # number of training epochs, default 10
    parser.add_argument(
        "--epochs", type=int, default=10000, help="number of training epochs"
    )

    # predict a pdb entry by pdbid
    parser.add_argument(
        "--predict", type=str, default=None, help="predict a pdb entry by pdb id"
    )

    # show the model details
    parser.add_argument("--model", action="store_true", help="show the model details")

    # do the parsing
    args = parser.parse_args()

    return args


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
    # elif torch.backends.mps.is_available():
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
    H = 60  # num_hidden_units
    inner_count = 30
    model = nn.Sequential(nn.Linear(D, H))
    model.append(nn.ReLU())
    for i in range(inner_count):
        model.append(nn.Linear(H, H))
        model.append(nn.ReLU())
    model.append(nn.Linear(H, C))
    model.to(find_torch_training_device())
    return model


def train(model, seqs, output, training_epochs):
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train
    print("Training...", end="")

    in_data = torch.from_numpy(seqs).float().to(find_torch_training_device())
    out_data = torch.from_numpy(output).float().to(find_torch_training_device())
    best_model = model
    # calculate current model loss
    output_pred = model(in_data)
    prev_loss = criterion(output_pred, out_data).item()
    prev_save_epoch = 0
    for epoch in range(training_epochs):
        optimizer.zero_grad()
        output_pred = model(in_data)
        loss = criterion(output_pred, out_data)
        loss.backward()
        optimizer.step()
        # print with format string: epoch, loss, best_loss
        if loss.item() < prev_loss:
            best_model = copy.deepcopy(model)
            prev_loss = loss.item()
            if epoch - prev_save_epoch > 100:  # reduce for excessive saving
                prev_save_epoch = epoch
                torch.save(model, "model.pt")
        print(
            "Epoch: {:d}({:d}) Loss: {:10.2f} Best: {:10.2f}            ".format(
                epoch, prev_save_epoch, loss.item(), prev_loss
            ),
            end="\r",
        )
    print()
    print("done")
    return best_model


def test(model, seqs_test, output_test):
    # Test
    print("Testing...", end="")
    in_data = torch.from_numpy(seqs_test).float().to(find_torch_training_device())
    out_data = torch.from_numpy(output_test).float().to(find_torch_training_device())
    criterion = nn.MSELoss()
    output_pred = model(in_data)
    loss = criterion(output_pred, out_data)
    print("done")
    print("Test loss:", loss.item())


def init_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Runs the model on one particular record extracted from the pdb database
def predict(model, metadata, seqs_train, output_train, seqs_test, output_test, pdbid):
    # find the encoded input and output of the pdb in either the training or the test set
    # is it in the training set?

    # is it in the test set?
    if pdbid in metadata["pdb_names_test"]:
        index = metadata["pdb_names_test"].index(pdbid)
        in_data = (
            torch.from_numpy(seqs_test[index]).float().to(find_torch_training_device())
        )
        out_data = (
            torch.from_numpy(output_test[index])
            .float()
            .to(find_torch_training_device())
        )
    if pdbid in metadata["pdb_names_train"]:
        index = metadata["pdb_names_train"].index(pdbid)
        in_data = (
            torch.from_numpy(seqs_train[index]).float().to(find_torch_training_device())
        )
        out_data = (
            torch.from_numpy(output_train[index])
            .float()
            .to(find_torch_training_device())
        )

    # Run the in_data thru the model
    output_pred = model(in_data)

    pred_hash = get_active_values(output_pred, metadata["sig_keys"])
    actual_hash = get_active_values(out_data, metadata["sig_keys"])

    # Print pred_hash and the actual hash as a table
    # Combine the keys of the two hashes and use them as the first column
    all_keys = set(pred_hash.keys()).union(set(actual_hash.keys()))
    # The second column is the actual value output
    # The third column is the predicted output
    # The fourth column is the difference between the predicted and actual output

    print("Delta for PDBID:", pdbid)
    # Print the headers
    print("{:19s} {:10s} {:10s} {:10s}".format("key", "actual", "predicted", "delta"))

    for k in all_keys:
        # get the value from the actual hash or - if it does not exist
        actual_value = actual_hash.get(k, -1)
        # get the value from the predicted hash or - if it does not exist
        predicted_value = pred_hash.get(k, -1)

        if actual_value == -1:
            delta = "{:12s}".format("missing")
        elif predicted_value == -1:
            delta = "{:12s}".format("excess")
        else:
            delta = "{:12.2f}".format(predicted_value - actual_value)
        # print the row
        print(
            "{:19s} {:10.2f} {:10.2f} {}".format(
                k, actual_value, predicted_value, delta
            )
        )


def main():
    metadata, seqs, output, seqs_test, output_test = load_data()
    args = parse_commandline()
    init_seeds(42)
    if args.preload:
        print("Preloading model from", args.preload)
        model = torch.load(args.preload)
    else:
        model = setup_model(seqs, output)
    if args.model:
        print(model)

    if not args.test and None == args.predict:
        train(model, seqs, output, args.epochs)

    test(model, seqs_test, output_test)

    if args.predict != None:
        print("Predicting", args.predict)
        predict(model, metadata, seqs, output, seqs_test, output_test, args.predict)


if __name__ == "__main__":
    main()
