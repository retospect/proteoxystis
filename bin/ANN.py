import torch
from train import *
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metadata, seqs_train, output_train, seqs_test, output_test, relevant_test, relevant_train = load_data()

train_dataset = TensorDataset(torch.tensor(seqs_train), torch.tensor(output_train))

size = len(train_dataset)
train_size = int(0.8 * size)
val_size = size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
test_dataset = TensorDataset(torch.tensor(seqs_test), torch.tensor(output_test))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        input = 754
        output = 86688

        self.fc1 = nn.Linear(input, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output)

        self.dropout = nn.Dropout(0.5)

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

def train_model(model, optimizer, epochs):
    model.train()

    train_predicts = []
    train_targets = []

    val_predicts = []
    val_targets = []

    for epoch in range(epochs):

        batch_idx = 0
        prev_accuracy = float(0)

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

            if accuracy >= prev_accuracy:
                prev_accuracy = accuracy
            else:
                break

            print("At batch number {b} in epoch {e}, the training loss is {l:.4f} and the training accuracy is {a:.4f}%".format(
                b=batch_idx, e=epoch, l=loss, a=accuracy))

        model.eval()

        with torch.no_grad():

            batch_idx = 0

            prev_accuracy = 0

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

                if accuracy >= prev_accuracy:
                    prev_accuracy = accuracy
                else:
                    break

                print("At batch number {b} in epoch {e} the validation loss is {l:.4f} and the validation accuracy is {a:.4f}%".
                                                                            format(b=batch_idx, e=epoch, l=loss, a=accuracy))

    final_training_accuracy = accuracy_score(train_targets, train_predicts) * 100
    final_validation_accuracy = accuracy_score(val_targets, val_predicts) * 100

    print("................................................................................")

    print("Training is complete")

    return final_training_accuracy, final_validation_accuracy

def test_model(model):

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

epochs = 20
learning_rate = 0.001
weight_decay = 0.01

model = ANN()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

final_training_accuracy, final_validation_accuracy = train_model(model, optimizer, epochs)
final_testing_accuracy = test_model(model)

print("The final training accuracy is {ftra:.4f}%".format(ftra=final_training_accuracy))
print("The final validation accuracy is {fva:.4f}%".format(fva=final_validation_accuracy))
print("The final testing accuracy is {fta:.4f}%".format(fta=final_testing_accuracy))
