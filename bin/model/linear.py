# Linear model


#! python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def append_to_model(model, n_input, n_output, layers, relu_spacing):
    model.append(nn.Linear(n_input, n_output))
    model.append(nn.ReLU())
    for i in range(layers):
        model.append(nn.Linear(n_output, n_output))
        if i % relu_spacing == 0:
            model.append(nn.ReLU())
    return model


def setup_model(seqs, output, metadata):
    # Setup model, model definition
    print(
        "Setting up new linear model (will overwrite the old if it exists)...",
        end="",
        flush=True,
    )
    D = seqs.shape[1]  # num_features
    C = output.shape[1]  # num_output_values
    # input with a 1d convolutional network, for the whole list (lenght D).
    # the step size is a multiple of the aminoacid encoding
    amino_acid_encoding_length = len(metadata["aminoacids"])
    # TODO: Convolutional 1d network with a window size of the aminoacid encoding length or a multiple
    # model = torch.nn.Conv1d(C, D, amino_acid_encoding_length)

    model = nn.Sequential(nn.Linear(D, 512))
    model = append_to_model(model, 512, 128, 2, 1)
    model = append_to_model(model, 128, 32, 100, 10)
    model = append_to_model(model, 32, 200, 5, 3)
    model.append(nn.Linear(200, C))

    print("done")
    return model
