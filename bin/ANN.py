import os
import torch
from train import *
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def data_split(seqs_train, output_train, seqs_test, output_test):
    train_dataset = TensorDataset(torch.tensor(seqs_train), torch.tensor(output_train))

    size = len(train_dataset)
    train_size = int(0.9 * size)
    val_size = size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    test_dataset = TensorDataset(torch.tensor(seqs_test), torch.tensor(output_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        input = 754
        hidden = 10
        output = 86688
        ratio = 0.5

        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

        self.dropout = nn.Dropout(ratio)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

def train_model(model, optimizer, epochs, train_loader, val_loader):
    model.train()

    prev_train_accuracy = float(0)

    train_losses = []
    val_losses = []

    train_predicts = []
    train_targets = []

    val_predicts = []
    val_targets = []

    for epoch in range(epochs):

        batch_idx = 0

        for batch in train_loader:

            target = batch[0].float()
            data = batch[1].float()

            output = model(data)

            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()

            batch_idx += 1

            predictions = torch.argmax(output, dim=1)
            targets = torch.argmax(target, dim=1)

            train_predicts.extend(predictions)
            train_targets.extend(targets)

            accuracy = accuracy_score(train_targets, train_predicts) * 100

            if accuracy >= prev_train_accuracy:
                train_losses.append(loss)
                prev_train_accuracy = accuracy
            else:
                break

            print("At batch number {b} in epoch {e}, the training loss is {l:.4f} and the training accuracy is {a:.4f}%".format(
                b=batch_idx, e=epoch, l=loss, a=accuracy))

        model.eval()

        with torch.no_grad():

            batch_idx = 0

            for batch in val_loader:

                target = batch[0].float()
                data = batch[1].float()

                output = model(data)

                loss = F.cross_entropy(output, target)

                batch_idx += 1

                val_losses.append(loss)

                predictions = torch.argmax(output, dim=1)
                targets = torch.argmax(target, dim=1)

                val_predicts.extend(predictions)
                val_targets.extend(targets)

                accuracy = accuracy_score(val_targets, val_predicts) * 100

                print("At batch number {b} in epoch {e} the validation loss is {l:.4f} and the validation accuracy is {a:.4f}%".
                                                                            format(b=batch_idx, e=epoch, l=loss, a=accuracy))

    final_training_accuracy = accuracy_score(train_targets, train_predicts) * 100
    final_validation_accuracy = accuracy_score(val_targets, val_predicts) * 100

    print("................................................................................")

    print("Training is complete")

    return train_losses, val_losses, final_training_accuracy, final_validation_accuracy

def test_model(model, test_loader):

    test_predicts = []
    test_targets = []

    batch_idx = 0

    for batch in test_loader:

        model.eval()

        with torch.no_grad():

            target = batch[0].float()
            data = batch[1].float()

            output = model(data)

            batch_idx += 1

            predictions = torch.argmax(output, dim=1)
            targets = torch.argmax(target, dim=1)

            test_predicts.extend(predictions)
            test_targets.extend(targets)

            accuracy = accuracy_score(test_targets, test_predicts) * 100

            print("At batch number {b} the testing accuracy is {a:.4f}%".format(b=batch_idx, a=accuracy))

    final_testing_accuracy = accuracy_score(test_targets, test_predicts) * 100

    print("................................................................................")

    print("Testing is complete")

    return final_testing_accuracy

def plot_loss(train_losses, val_losses, folder, filename):
    figure, axis = plt.subplots(1, 2, figsize=(10, 5))

    axis[0].plot(range(len(train_losses)), train_losses)
    axis[0].set_title("Training Loss over Epochs")

    axis[1].plot(range(len(val_losses)), val_losses)
    axis[1].set_title("Validation Loss over Epochs")

    figure.subplots_adjust(wspace=0.5)

    image_path = os.path.join(folder, filename)
    plt.savefig(image_path)

def main():
    metadata, seqs_train, output_train, seqs_test, output_test, relevant_test, relevant_train = load_data()

    train_loader, val_loader, test_loader = data_split(seqs_train, output_train, seqs_test, output_test)

    epochs = 5
    learning_rate = 0.001
    weight_decay = 0.01

    model = ANN()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses, final_training_accuracy, final_validation_accuracy = train_model(model, optimizer, epochs, train_loader, val_loader)
    final_testing_accuracy = test_model(model, test_loader)

    train_losses = torch.tensor(train_losses).detach().numpy()
    val_losses = torch.tensor(val_losses).detach().numpy()

    folder = '/Users/arpitha/Documents/cse144/Final_Project/proteoxystis/bin'
    filename = 'Losses'

    plot_loss(train_losses, val_losses, folder, filename)

    print("The final training accuracy is {ftra:.4f}%".format(ftra=final_training_accuracy))
    print("The final validation accuracy is {fva:.4f}%".format(fva=final_validation_accuracy))
    print("The final testing accuracy is {fta:.4f}%".format(fta=final_testing_accuracy))

if __name__ == "__main__":
    main()

# Citations:
# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
# https://www.tutorialspoint.com/how-to-plot-a-graph-in-python
# https://www.geeksforgeeks.org/plot-multiple-plots-in-matplotlib/
