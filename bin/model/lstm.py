import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def setup_model(seqs, output, metadata):
    # Setup model, model definition
    print(
        "Setting up new lstm model (will overwrite the old if it exists)...",
        end="",
        flush=True,
    )
    print("\nNot implemented: lstm")
    exit(1)
