import os
import torch
from train import *
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def data_split(seqs_train, output_train, seqs_test, output_test):
    train_dataset = TensorDataset(torch.tensor(seqs_train), torch.tensor(output_train))

    size = len(train_dataset)
    train_size = int(0.9 * size)
    val_size = size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    test_dataset = TensorDataset(torch.tensor(seqs_test), torch.tensor(output_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

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

    old_model = copy.deepcopy(model)

    train_losses = []
    val_losses = []

    train_predicts = []
    train_targets = []

    val_predicts = []
    val_targets = []

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        batch_idx = 0

        prev_train_accuracy = 0

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

            if float(accuracy) >= float(prev_train_accuracy):
                train_losses.append(loss.item())
                prev_train_accuracy = accuracy
                train_accuracies.append(accuracy)
                old_model = copy.deepcopy(model)
            else:
                model = copy.deepcopy(old_model)

            print("At batch number {b} in epoch {e}, the training loss is {l:.4f} and the training accuracy is {a:.4f}%".format(
                b=batch_idx, e=(epoch+1), l=loss, a=accuracy))

        model.eval()

        with torch.no_grad():

            prev_val_accuracy = 0

            batch_idx = 0

            for batch in val_loader:

                target = batch[0].float()
                data = batch[1].float()

                output = model(data)

                loss = F.cross_entropy(output, target)

                batch_idx += 1

                predictions = torch.argmax(output, dim=1)
                targets = torch.argmax(target, dim=1)

                val_predicts.extend(predictions)
                val_targets.extend(targets)

                accuracy = accuracy_score(val_targets, val_predicts) * 100

                if float(accuracy) >= float(prev_val_accuracy):
                    val_losses.append(loss.item())
                    prev_val_accuracy = accuracy
                    val_accuracies.append(accuracy)
                    old_model = copy.deepcopy(model)
                else:
                    model = copy.deepcopy(old_model)

                print("At batch number {b} in epoch {e} the validation loss is {l:.4f} and the validation accuracy is {a:.4f}%".
                                                                            format(b=batch_idx, e=(epoch+1), l=loss, a=accuracy))

    final_training_accuracy = accuracy_score(train_targets, train_predicts) * 100
    final_validation_accuracy = accuracy_score(val_targets, val_predicts) * 100

    train_matrix = confusion_matrix(train_targets, train_predicts)
    val_matrix = confusion_matrix(val_targets, val_predicts)

    print("...................................................................................")

    print("Training is complete")

    return train_losses, val_losses, final_training_accuracy, final_validation_accuracy, train_accuracies, val_accuracies, train_matrix, val_matrix

def test_model(model, test_loader):

    test_predicts = []
    test_targets = []

    test_accuracies = []

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
            test_accuracies.append(accuracy)

            print("At batch number {b} the testing accuracy is {a:.4f}%".format(b=batch_idx, a=accuracy))

    final_testing_accuracy = accuracy_score(test_targets, test_predicts) * 100

    test_matrix = confusion_matrix(test_targets, test_predicts)

    print("...................................................................................")

    print("Testing is complete")

    return final_testing_accuracy, test_accuracies, test_matrix

def plot_loss(train_losses, val_losses, folder, filename):
    figure, axis = plt.subplots(1, 2, figsize=(10, 5))

    axis[0].plot(range(len(train_losses)), train_losses)
    axis[0].set_xlabel("Iterations")
    axis[0].set_ylabel("Loss")
    axis[0].set_title("Training Loss over Epochs")

    axis[1].plot(range(len(val_losses)), val_losses)
    axis[1].set_xlabel("Iterations")
    axis[1].set_ylabel("Loss")
    axis[1].set_title("Validation Loss over Epochs")

    figure.subplots_adjust(wspace=0.5)

    image_path = os.path.join(folder, filename)
    plt.savefig(image_path)

    print("...................................................................................")

    print("Done plotting losses")

def plot_accuracy(train_accuracies, val_accuracies, test_accuracies, folder, filename):
    figure, axis = plt.subplots(1, 3, figsize=(15, 5))

    axis[0].plot(range(len(train_accuracies)), train_accuracies)
    axis[0].set_xlabel("Iterations")
    axis[0].set_ylabel("Accuracy")
    axis[0].set_title("Training Accuracies over Epochs")

    axis[1].plot(range(len(val_accuracies)), val_accuracies)
    axis[1].set_xlabel("Iterations")
    axis[1].set_ylabel("Accuracy")
    axis[1].set_title("Validation Accuracies over Epochs")

    axis[2].plot(range(len(test_accuracies)), test_accuracies)
    axis[2].set_xlabel("Iterations")
    axis[2].set_ylabel("Accuracy")
    axis[2].set_title("Testing Accuracies over Epochs")

    figure.subplots_adjust(wspace=0.5)

    image_path = os.path.join(folder, filename)
    plt.savefig(image_path)

    print("...................................................................................")

    print("Done plotting accuracies")

def plot_cm(train_matrix, val_matrix, test_matrix, folder, filename):
    figure, axis = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(train_matrix, annot=True, fmt='d', cmap='Blues', ax=axis[0])
    axis[0].set_title("Training Confusion Matrix Heatmap")

    sns.heatmap(val_matrix, annot=True, fmt='d', cmap='Blues', ax=axis[1])
    axis[1].set_title("Validation Confusion Matrix Heatmap")

    sns.heatmap(test_matrix, annot=True, fmt='d', cmap='Blues', ax=axis[2])
    axis[2].set_title("Testing Confusion Matrix Heatmap")

    figure.subplots_adjust(wspace=0.5)

    image_path = os.path.join(folder, filename)
    plt.savefig(image_path)

    print("...................................................................................")

    print("Done plotting heatmaps")

def main():
    metadata, seqs_train, output_train, seqs_test, output_test, relevant_test, relevant_train = load_data()

    train_loader, val_loader, test_loader = data_split(seqs_train, output_train, seqs_test, output_test)

    epochs = 1
    learning_rate = 0.001
    weight_decay = 0.01

    model = ANN()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses, final_training_accuracy, final_validation_accuracy, train_accuracies, val_accuracies, train_matrix, val_matrix = train_model(model, optimizer, epochs, train_loader, val_loader)
    final_testing_accuracy, test_accuracies, test_matrix = test_model(model, test_loader)

    train_losses = torch.tensor(train_losses).detach().numpy()
    val_losses = torch.tensor(val_losses).detach().numpy()

    folder = '/Users/arpitha/Documents/cse144/Final_Project/proteoxystis/bin'

    loss_filename = 'Losses_Plot'
    accuracy_filename = 'Accuracies_Plot'
    matrix_filename = "Confusion_Matrix_Heatmap"

    plot_loss(train_losses, val_losses, folder, loss_filename)
    plot_accuracy(train_accuracies, val_accuracies, test_accuracies, folder, accuracy_filename)
    # plot_cm(train_matrix, val_matrix, test_matrix, folder, matrix_filename)

    print("...................................................................................")

    print("Confusion Matrices:")
    print("Train Confusion Matrix: {tcm:.4f}".format(tcm=train_matrix))
    print("Validation Confusion Matrix: {vcm:.4f}".format(vcm=val_matrix))
    print("Test Confusion Matrix: {tecm:.4f}".format(tecm=test_matrix))

    print("...................................................................................")

    print("Accuracies:")
    print("The final training accuracy is {ftra}%".format(ftra=final_training_accuracy))
    print("The final validation accuracy is {fva}%".format(fva=final_validation_accuracy))
    print("The final testing accuracy is {fta}%".format(fta=final_testing_accuracy))

if __name__ == "__main__":
    main()

# Citations:
# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
# https://www.tutorialspoint.com/how-to-plot-a-graph-in-python
# https://www.geeksforgeeks.org/plot-multiple-plots-in-matplotlib/
