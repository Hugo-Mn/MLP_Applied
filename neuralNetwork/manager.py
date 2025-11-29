from neuralNetwork import perceptron as p
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import os


class NeuralNetworkManager:
    def __init__(self, configPath=None, modelPath=None, epochs=50, patience=5, batch_size=32):
        self.networks = {}
        self.path = configPath
        self.modelPath = modelPath
        self.config = {
            'epochs': epochs,
            'patience': patience,
            'batch_size': batch_size
        }

    def setConfig(self):
        with open(self.path, 'r') as f:
            config = json.load(f)
            self.config['input_size'] = config['input_size']
            self.config['output_size'] = config['output_size']
            self.config['FActivation'] = self.foundActivationFunction(config['activationFunction'])
            self.config['learningRate'] = config['learningRate']
            self.config['lossFunction'] = self.foundActivationFunction(config['lossFunction'])
            self.config['dropoutRate'] = config['dropoutRate']
            self.config['hiddenLayers'] = config['hiddenLayers']

    def CreatePerceptron(self, name):
        self.networks[name] = p.Perceptron(
            input_size=self.config['input_size'],
            output_size=self.config['output_size'],
            fActivation=self.config['FActivation'],
            hiddenLayers=self.config['hiddenLayers'],
            dropoutRate=self.config['dropoutRate']
        )

    def CreatePerceptronFromDimensions(self, name, input_size=1024, output_size=5, hiddenLayers=[], dropoutRate=0.0, fActivation=F.gelu):
        self.networks[name] = p.Perceptron(
            input_size,
            output_size,
            fActivation,
            hiddenLayers,
            dropoutRate
        )

    def foundActivationFunction(self, name):
        functions = {
            'relu': F.relu,
            'gelu': F.gelu,
            'sigmoid': F.sigmoid,
            'crossentropy': F.cross_entropy,
            'MSE': F.mse_loss
        }
        return functions[name]

    def train(self, name, train_loader):
        model = self.networks[name]
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        os.makedirs('checkpoints', exist_ok=True)
        best_loss = float('inf')
        epochs_without_improvement = 0
        patience = self.config['patience']

        for epoch in range(self.config['epochs']):
            train_loss = model.train_epoch(train_loader, self.config['lossFunction'], optimizer)

            if train_loss < best_loss:
                best_loss = train_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f'checkpoints/{name}_best.pth')
                print(f"Epoch {epoch+1}/{self.config['epochs']} | Loss: {train_loss:.4f} ✓")
            else:
                epochs_without_improvement += 1
                print(f"Epoch {epoch+1}/{self.config['epochs']} | Loss: {train_loss:.4f} ({epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                print(f"Arrêt : pas d'amélioration depuis {patience} epochs")
                break

        model.load_state_dict(torch.load(f'checkpoints/{name}_best.pth'))
        print(f"Meilleur modèle chargé (Loss: {best_loss:.4f})")

    def prediction(self, name, input_data):
        model = self.networks[name]
        model.eval()
        with torch.no_grad():
            outputs = model(input_data)
        return outputs

    def loadPerceptron(self, name, model_path):
        if name in self.networks:
            self.networks[name].load_state_dict(torch.load(model_path))
            print(f"Modèle {name} chargé depuis {model_path}")

    def load_checkpoint(self, name, checkpoint_path):
        if name in self.networks:
            self.networks[name].load_state_dict(torch.load(checkpoint_path))
            print(f"Modèle {name} chargé depuis {checkpoint_path}")

    def savePerceptron(self, name, save_path):
        if name in self.networks:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.networks[name].state_dict(), save_path)
            print(f"Model '{name}' saved to {save_path}")
        else:
            print(f"Network '{name}' not found!")