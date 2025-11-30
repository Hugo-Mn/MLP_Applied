import torch
import torch.nn as nn
import torch.nn.functional as F


class Perceptron(nn.Module):
    def __init__(self, input_size, output_size, fActivation=F.relu, hiddenLayers=[], dropoutRate=0.0):
        super(Perceptron, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hiddenLayers = hiddenLayers
        self.fActivation = fActivation
        self.dropoutRate = dropoutRate
        self.network = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.createNetwork()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        self.to(self.device)


    def createNetwork(self):
        layers = []
        prev_size = self.input_size

        for hidden_size in self.hiddenLayers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Dropout(self.dropoutRate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        for i, layer in enumerate(self.network):
            x = layer(x)
            if isinstance(layer, nn.Linear) and i < len(self.network) - 1:
                x = self.fActivation(x)
        return x

    def train(self, train_loader, loss_function, optimizer):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def train_epoch(self, train_loader, loss_function, optimizer):
        return self.train(train_loader, loss_function, self.optimizer)

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
        return outputs