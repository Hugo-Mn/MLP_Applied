#pyton
from neuralNetwork import manager as nn_manager
import argparse
from neuralNetwork.dataset_loader import create_dataloader

class Trainer:
    def __init__(self):
        self.args = self.parserManager()
        self.manager = nn_manager.NeuralNetworkManager(
            self.args.config,
            self.args.modelPath,
            self.args.epochs,
            self.args.patience,
            self.args.batch_size
        )

    def parserManager(self):
        parser = argparse.ArgumentParser(description="Neural Network Trainer")
        parser.add_argument('action', choices=['train', 'predict', 'create'], help='Action to perform: train, predict or create')
        parser.add_argument('modelPath', type=str, nargs='?', default=None, help='Path of the neural network')
        parser.add_argument('--dataset', type=str, nargs='?', default=None, help='Path to the training data')
        parser.add_argument('--config', type=str,nargs='?', default=None, help='Path to the configuration file (required for create action)')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
        parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping during training')
        args = parser.parse_args()
        return args

    def loadNeuralNetwork(self):
        name = self.args.modelPath.split('/')[-1].split('_')[0]
        self.manager.CreatePerceptronFromDimensions(name)
        self.manager.loadPerceptron(name, self.args.modelPath)

    def commandManager(self):
        if self.args.action == 'create':
            if not self.args.config:
                raise ValueError("Configuration file path is required for create action.")
            self.manager.setConfig()
            name = input("Enter a name for the created model: ")
            self.manager.CreatePerceptron(name)
            self.manager.savePerceptron(name, f'./checkpoints/{name}_model.pth')
            print("Neural network created successfully.")
        elif self.args.action == 'train':
            if not self.args.dataset:
                raise ValueError("Dataset path required for train action.")
            if not self.args.config:
                raise ValueError("Config is required for train action (--config).")

            self.manager.setConfig()
            if self.args.modelPath:
                # Load existing model
                name = self.args.modelPath.split('/')[-1].split('_')[0]
                self.manager.CreatePerceptron(name)
                self.manager.loadPerceptron(name, self.args.modelPath)
            else:
                # Create new model
                self.manager.CreatePerceptron('default')
                name = 'default'

            train_loader = create_dataloader(self.args.dataset, self.args.batch_size)
            self.manager.train(name, train_loader)
            print("Training completed successfully.")
        elif self.args.action == 'predict':
            if not self.args.modelPath:
                raise ValueError("Model path is required for predict action.")
            self.loadNeuralNetwork()
            self.manager.prediction('default', self.args.dataset)
            print("Prediction completed successfully.")



if __name__ == '__main__':
    trainer = Trainer()
    trainer.commandManager()