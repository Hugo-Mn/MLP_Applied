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
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset path does not exist: {dataset_path}")
            return

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

        print(f"Dataset loaded: {len(self.samples)} samples found")

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