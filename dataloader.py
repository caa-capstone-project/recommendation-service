import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np

class ChunkedRatingsDataset(IterableDataset):
    def __init__(self, features_file, labels_file, chunk_size=10):
        self.features_file = features_file
        self.labels = pd.read_csv(labels_file)
        self.chunk_size = chunk_size

        self.create_userid_dict()

    def __iter__(self):
        return self._data_generator()

    def _data_generator(self):
        features_iter = pd.read_csv(self.features_file, chunksize=self.chunk_size, iterator=True)
        
        for features_chunk in features_iter:
            for _,features in features_chunk.iterrows():
                yield torch.tensor(features.iloc[1:].values.astype(np.float32)), torch.tensor(self.user_ids[features['userid']])

    def create_userid_dict(self):
        self.user_ids = self.labels.set_index('userid').T.to_dict('list')

# # Example usage
# features_file = 'augmented_data.csv'
# labels_file = 'data.csv'
# chunk_size = 128  # Adjust based on memory constraints
# dataset = ChunkedRatingsDataset(features_file, labels_file, chunk_size)
# data_loader = DataLoader(dataset, batch_size=32)


# from tqdm import tqdm

# # Step 2: Iterate Through the DataLoader with tqdm
# for batch_idx, (features, labels) in enumerate(tqdm(data_loader, desc="Loading Batches")):
#     if batch_idx >= 100:
#         break
