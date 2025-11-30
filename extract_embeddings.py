import sys
sys.path.insert(0, r'D:\Work\Evil work folder\applied predictive analysis\MLP_Applied')

import pickle
import os

# Chemin du dataset
dataset_path = r'..\datasets\Split_1'

# Cherche s'il existe déjà un fichier de cache
cache_files = [
    os.path.join(dataset_path, '.embeddings_cache.pt'),
    os.path.join(dataset_path, '.embeddings_backup.pkl'),
]

# Essaye de charger le cache existant
loaded_data = {}
for cache_file in cache_files:
    if os.path.exists(cache_file):
        print(f"Found cache file: {cache_file}")
        try:
            if cache_file.endswith('.pkl'):
                with open(cache_file, 'rb') as f:
                    loaded_data = pickle.load(f)
            else:
                import torch
                loaded_data = torch.load(cache_file)
            print(f"Loaded {len(loaded_data)} embeddings from cache!")
            break
        except Exception as e:
            print(f"Error loading cache: {e}")

# Sauvegarde les embeddings trouvés
if loaded_data:
    with open('embeddings_extracted.pkl', 'wb') as f:
        pickle.dump(loaded_data, f)
    print(f"Saved {len(loaded_data)} embeddings to embeddings_extracted.pkl!")
else:
    print("No cache file found. Please run the training first to generate embeddings.")
