import torch
import torch.nn as nn

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(100, 512)
        self.layer1 = self.make_layers(512, num_repeat=100)
        self.fc5 = nn.Linear(512, 100)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layer1(x)
        # x = nn.Dropout(0.2)(x)
        x = self.fc5(x)
        return x

    def make_layers(self, value, num_repeat):
        layers = []
        for _ in range(num_repeat):
            layers.append(nn.Linear(value, value))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))

        return nn.Sequential(*layers)
