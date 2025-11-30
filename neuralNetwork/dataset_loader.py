import os
import torch
from torch.utils.data import Dataset, DataLoader
from neuralNetwork.encoder import Encoder


class AudioTextDataset(Dataset):

    def __init__(self, dataset_path):
        self.encoder = Encoder()
        self.samples = []
        self.loadDataset(dataset_path)
        print("Pre-computing embeddings... (this may take a while)")
        self.precompute_embeddings()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Embeddings are already pre-computed, just return them
        final_input = sample['embedding']
        label = sample['label']
        return final_input, label

    def precompute_embeddings(self):
        for i, sample in enumerate(self.samples):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(self.samples)} samples")
            try:
                final_input, target = self.encoder.encodeOneSet(sample['text'], sample['audio'])
                label = self.countryToLabel(target)
                sample['embedding'] = final_input.squeeze(0)
                sample['label'] = label
            except Exception as e:
                print(f"  Error processing sample {i}: {e}")
                sample['embedding'] = torch.zeros(4352)  # Default embedding
                sample['label'] = torch.tensor(0, dtype=torch.long)

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
    
    def save_current_embeddings(self, filename="embeddings_backup.pkl"):
        """Sauvegarde les embeddings actuels en fichier"""
        import pickle
        data = {}
        for i, sample in enumerate(self.samples):
            if 'embedding' in sample:
                data[i] = {
                    'embedding': sample['embedding'].cpu(),
                    'label': sample['label']
                }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(data)} embeddings to {filename}")



def create_dataloader(dataset_path, batch_size=32):
    dataset = AudioTextDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
