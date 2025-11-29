import os
import torch
from torch.utils.data import Dataset, DataLoader
from neuralNetwork.encoder import Encoder


class AudioTextDataset(Dataset):

    def __init__(self, dataset_path):
        self.encoder = Encoder()
        self.samples = []
        self.loadDataset(dataset_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        final_input, target = self.encoder.encodeOneSet(sample['text'], sample['audio'])
        label = self.countryToLabel(target)

        return final_input.squeeze(0), label

    def loadDataset(self, dataset_path):
        for extract_folder in sorted(os.listdir(dataset_path)):
            extract_path = os.path.join(dataset_path, extract_folder)

            if not os.path.isdir(extract_path):
                continue

            audio_file = None
            text_file = None

            for file in os.listdir(extract_path):
                if file.endswith('.opus'):
                    audio_file = os.path.join(extract_path, file)
                elif file == 'text.txt':
                    text_file = os.path.join(extract_path, file)

            if audio_file or text_file:
                self.samples.append({
                    'audio': audio_file,
                    'text': text_file,
                    'extract_name': extract_folder
                })

        print(f"Dataset chargé: {len(self.samples)} samples trouvés")

    def countryToLabel(self, country):
        if country is None:
            return torch.tensor(0, dtype=torch.long)
        mapping = {
            "French": 0,
            "Polish": 1,
            "Portuguese": 2,
            "Italian": 3,
            "Spanish": 4
        }
        label = mapping.get(country, 0)
        return torch.tensor(label, dtype=torch.long)


def create_dataloader(dataset_path, batch_size=32):
    dataset = AudioTextDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == '__main__':
    # Test: python -m neuralNetwork.dataset_loader
    import os
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasetTrain')
    
    print("Creating dataset...")
    dataset = AudioTextDataset(dataset_path)
    
    print(f"\nDataset info:")
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        inputs, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Label: {label}")
    
    print(f"\nCreating dataloader...")
    train_loader = create_dataloader(dataset_path, batch_size=2)
    
    print(f"DataLoader info:")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Batches: {len(train_loader)}")
    
    print(f"\nFirst batch:")
    for inputs, labels in train_loader:
        print(f"  Input shape: {inputs.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        break
