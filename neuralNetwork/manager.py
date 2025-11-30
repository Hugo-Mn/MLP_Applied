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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
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
        
        # Extract just the model name from path if needed
        model_name = name.split('/')[-1].split('\\')[-1] if '/' in name or '\\' in name else name
        
        print("Starting training...")

        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch+1}/{self.config['epochs']})")
            train_loss = model.train_epoch(train_loader, self.config['lossFunction'], optimizer)
            if train_loss < best_loss:
                best_loss = train_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')
                print(f"Epoch {epoch+1}/{self.config['epochs']} | Loss: {train_loss:.4f} [BEST]")
            else:
                epochs_without_improvement += 1
                print(f"Epoch {epoch+1}/{self.config['epochs']} | Loss: {train_loss:.4f} ({epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break

        model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pth'))
        print(f"Best model loaded (Loss: {best_loss:.4f})")

    def prediction(self, name, input_data):
        model = self.networks[name]
        model.eval()
        with torch.no_grad():
            outputs = model(input_data)
        return outputs

    def evaluate(self, name, test_loader):
        model = self.networks[name]
        
        # Label mapping from dataset_loader
        label_mapping = {
            0: "French",
            1: "Polish",
            2: "Portuguese",
            3: "Italian",
            4: "Spanish"
        }
        
        correct = 0
        total = 0
        correct_per_label = {}
        total_per_label = {}
        
        for embeddings, labels in test_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            outputs = model.predict(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Count per label
            for pred, label in zip(predicted, labels):
                label_val = label.item()
                pred_val = pred.item()
                
                if label_val not in total_per_label:
                    total_per_label[label_val] = 0
                    correct_per_label[label_val] = 0
                
                total_per_label[label_val] += 1
                if pred_val == label_val:
                    correct_per_label[label_val] += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"\n=== Overall Accuracy: {accuracy:.2f}% ({correct}/{total}) ===\n")
        
        print("=== Accuracy per Language ===")
        for label_id in sorted(correct_per_label.keys()):
            label_name = label_mapping.get(label_id, f"Unknown({label_id})")
            label_correct = correct_per_label[label_id]
            label_total = total_per_label[label_id]
            label_accuracy = (label_correct / label_total) * 100 if label_total > 0 else 0
            print(f"{label_name:12}: {label_accuracy:.2f}% ({label_correct}/{label_total})")
        
        print()
        return accuracy

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