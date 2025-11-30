#pyton
from neuralNetwork import manager as nn_manager
import argparse
from neuralNetwork.dataset_loader import create_dataloader
import torch
import pickle
import os

# Enable CUDA benchmark for faster training
torch.backends.cudnn.benchmark = True

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

    def save_train_loader(self, train_loader, filepath='./checkpoints/train_loader.pkl'):
        """Save train_loader to a pickle file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(train_loader, f)
        print(f"Train loader saved to {filepath}")

    def load_train_loader(self, filepath='./checkpoints/train_loader.pkl'):
        """Load train_loader from a pickle file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                train_loader = pickle.load(f)
            print(f"Train loader loaded from {filepath}")
            return train_loader
        return None

    def parserManager(self):
        parser = argparse.ArgumentParser(description="Neural Network Trainer")
        parser.add_argument('action', choices=['train', 'predict', 'create', 'evaluate'], help='Action to perform: train, predict, create or evaluate')
        parser.add_argument('modelPath', type=str, nargs='?', default=None, help='Path of the neural network')
        parser.add_argument('--dataset', type=str, nargs='?', default=None, help='Path to the training data')
        parser.add_argument('--config', type=str,nargs='?', default=None, help='Path to the configuration file (required for create action)')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
        parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping during training')
        args = parser.parse_args()
        return args

    def loadNeuralNetwork(self):
        name = self.args.modelPath.split('/')[-1].split('_')[0]
        # Load config first to get the right architecture
        if self.args.config:
            self.manager.setConfig()
            self.manager.CreatePerceptron(name)
        else:
            # Fallback to default dimensions if no config provided
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
            self.save_train_loader(train_loader)  # Save train_loader to file
            self.manager.config['epochs'] = 500  # Override epochs to 500 before training
            self.manager.train(name, train_loader)
            print("Training completed successfully.")
        elif self.args.action == 'predict':
            if not self.args.modelPath:
                raise ValueError("Model path is required for predict action.")
            self.loadNeuralNetwork()
            self.manager.prediction('default', self.args.dataset)
            print("Prediction completed successfully.")
        elif self.args.action == 'evaluate':
            if not self.args.modelPath:
                raise ValueError("Model path is required for evaluate action.")
            if not self.args.dataset:
                raise ValueError("Dataset path required for evaluate action.")
            self.loadNeuralNetwork()
            name = self.args.modelPath.split('/')[-1].split('_')[0]
            test_loader = create_dataloader(self.args.dataset, self.args.batch_size)
            accuracy = self.manager.evaluate(name, test_loader)
            print(f"Evaluation completed. Accuracy: {accuracy:.2f}%")



if __name__ == '__main__':
    trainer = Trainer()
    trainer.commandManager()