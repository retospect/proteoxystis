import torch
from train import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

args = parse_commandline()
metadata, seqs_train, output_train, seqs_test, output_test, relevant_test, relevant_train = load_data()

seqs_train, seqs_test = torch.tensor(seqs_train), torch.tensor(seqs_test)
output_train, output_test = torch.tensor(output_train), torch.tensor(output_test)

if seqs_train.shape[0] == output_train.shape[0]:
    train_dataset = data.TensorDataset(seqs_train, output_train)

if seqs_test.shape[0] == output_test.shape[0]:
    test_dataset = data.TensorDataset(seqs_test, output_test)

size = len(train_dataset)
train_size = int(0.7 * size)
val_size = size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=1)

        self.fc1_input_dim = 86688

        self.fc1 = nn.Linear(self.fc1_input_dim, 1024)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = x.view(-1, self.fc1_input_dim)

        x = F.relu(self.fc1(x))

        x = F.log_softmax(x, dim=1)

        return x

def train_model(model):
    model.train()

    epochs = 10
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = data.to(device).float(), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                  len(trainLoader.dataset), 100 * batch_idx / len(trainLoader), loss.item()))

model = CNN()
model.to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_model(model)
