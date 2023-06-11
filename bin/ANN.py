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

def standardize_preprocess(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)

    new_matrix = (matrix - mean)/std

    return new_matrix

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
    def __init__(self, input, hidden, output, ratio):
        super(ANN, self).__init__()

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
            else:
                break

            print("At batch number {b} in epoch {e}, the training loss is {l:.4f} and the training accuracy is {a:.4f}%".format(
                b=batch_idx, e=(epoch+1), l=loss, a=accuracy))

        model.eval()

        with torch.no_grad():

            prev_val_accuracy = 0

            batch_idx = 0

            old_model = copy.deepcopy(model)

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
    final_training_precision = precision_score(train_targets, train_predicts, average='weighted', zero_division=1.0) * 100
    final_training_recall = recall_score(train_targets, train_predicts, average='macro', zero_division=1.0) * 100
    final_training_f1_score = f1_score(train_targets, train_predicts, average='weighted') * 100

    final_validation_accuracy = accuracy_score(val_targets, val_predicts) * 100
    final_validation_precision = precision_score(val_targets, val_predicts, average='weighted', zero_division=1.0) * 100
    final_validation_recall = recall_score(val_targets, val_predicts, average='macro', zero_division=1.0) * 100
    final_validation_f1_score = f1_score(val_targets, val_predicts, average='weighted') * 100

    print("...................................................................................")

    print("Training is complete")

    return train_losses, val_losses, final_training_accuracy, final_training_precision, final_training_recall, final_training_f1_score, final_validation_accuracy, final_validation_precision, final_validation_recall, final_validation_f1_score, train_accuracies, val_accuracies

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
    final_testing_precision = precision_score(test_targets, test_predicts, average='weighted', zero_division=1.0) * 100
    final_testing_recall = recall_score(test_targets, test_predicts, average='macro', zero_division=1.0) * 100
    final_testing_f1_score = f1_score(test_targets, test_predicts, average='weighted') * 100

    print("...................................................................................")

    print("Testing is complete")

    return final_testing_accuracy, test_accuracies, final_testing_accuracy, final_testing_precision, final_testing_recall, final_testing_f1_score

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

def main():
    metadata, seqs_train, output_train, seqs_test, output_test, relevant_test, relevant_train = load_data()

    seqs_train_standardized = standardize_preprocess(seqs_train)
    seqs_test_standardized = standardize_preprocess(seqs_test)
    output_train_standardized = standardize_preprocess(output_train)
    output_test_standardized = standardize_preprocess(output_test)

    # seqs_train_standardized = preprocess(seqs_train, output_train, seqs_test, output_test)

    # with open("seqs_train.txt", "w") as file:
    #     for data in seqs_train:
    #         file.write(str(data) + "\n")
    #
    # with open("output_train.txt", "w") as file:
    #     for data in output_train:
    #         file.write(str(data) + "\n")
    #
    # with open("seqs_test.txt", "w") as file:
    #     for data in seqs_test:
    #         file.write(str(data) + "\n")
    #
    # with open("output_test.txt", "w") as file:
    #     for data in output_test:
    #         file.write(str(data) + "\n")

    train_loader, val_loader, test_loader = data_split(seqs_train_standardized, output_train_standardized, seqs_test_standardized, output_test_standardized)

    input = 698
    hidden = 10
    output = 86688
    ratio = 0.5

    epochs = 10
    learning_rate = 0.001
    weight_decay = 0.01
    momentum = 0.2

    model = ANN(input, hidden, output, ratio)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    train_losses, val_losses, final_training_accuracy, final_training_precision, final_training_recall, final_training_f1_score, final_validation_accuracy, final_validation_precision, final_validation_recall, final_validation_f1_score, train_accuracies, val_accuracies = train_model(model, optimizer, epochs, train_loader, val_loader)
    final_testing_accuracy, test_accuracies, final_testing_accuracy, final_testing_precision, final_testing_recall, final_testing_f1_score = test_model(model, test_loader)

    train_losses = torch.tensor(train_losses).detach().numpy()
    val_losses = torch.tensor(val_losses).detach().numpy()

    # with open("train_loss.txt", "w") as file:
    #     for loss in train_losses:
    #         file.write(str(loss) + "\n")
    #
    # with open("val_loss.txt", "w") as file:
    #     for loss in val_losses:
    #         file.write(str(loss) + "\n")
    #
    # with open("train_result.txt", "w") as file:
    #     for accuracy in train_accuracies:
    #         file.write(str(accuracy) + "\n")
    #
    # with open("val_result.txt", "w") as file:
    #     for accuracy in val_accuracies:
    #         file.write(str(accuracy) + "\n")
    #
    # with open("test_result.txt", "w") as file:
    #     for accuracy in test_accuracies:
    #         file.write(str(accuracy) + "\n")

    folder = '/Users/arpitha/Documents/cse144/Final_Project/proteoxystis/bin'

    loss_filename = 'Losses_Plot'
    accuracy_filename = 'Accuracies_Plot'
    matrix_filename = "Confusion_Matrix_Heatmap"

    plot_loss(train_losses, val_losses, folder, loss_filename)
    plot_accuracy(train_accuracies, val_accuracies, test_accuracies, folder, accuracy_filename)

    print("...................................................................................")

    print("Accuracy Scores:")
    print("The final training accuracy is {:.4f}%".format(final_training_accuracy))
    print("The final validation accuracy is {:.4f}%".format(final_validation_accuracy))
    print("The final testing accuracy is {:.4f}%".format(final_testing_accuracy))

    print("...................................................................................")

    print("Precision Scores:")
    print("The final training precision is {:.4f}%".format(final_training_precision))
    print("The final validation precision is {:.4f}%".format(final_validation_precision))
    print("The final testing precision is {:.4f}%".format(final_testing_precision))

    print("...................................................................................")

    print("Recall Scores:")
    print("The final training recall is {:.4f}%".format(final_training_recall))
    print("The final validation recall is {:.4f}%".format(final_validation_recall))
    print("The final testing recall is {:.4f}%".format(final_testing_recall))

    print("...................................................................................")

    print("F1 Scores:")
    print("The final training f1 score is {:.4f}%".format(final_training_f1_score))
    print("The final validation f1 score is {:.4f}%".format(final_validation_f1_score))
    print("The final testing f1 score is {:.4f}%".format(final_testing_f1_score))

    print("...................................................................................")

if __name__ == "__main__":
    main()

# Citations:
# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
# https://www.tutorialspoint.com/how-to-plot-a-graph-in-python
# https://www.geeksforgeeks.org/plot-multiple-plots-in-matplotlib/
# https://www.projectpro.io/recipes/what-is-feature-selection-neural-networks#:~:text=Feature%20selection%20reduces%20the%20overfitting,of%20the%20neural%20network%20model.
