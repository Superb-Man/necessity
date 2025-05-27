import os.path
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from configs import NUM_WORKERS

class AFdataset2(Dataset):
    def __init__(self, ds_npz):
        super().__init__()
        self.ds_npz = ds_npz

    def __getitem__(self, index: int):
        X = self.ds_npz['signal'][index]
        rhythm = self.ds_npz['rhythm'][index]
        patient_id = self.ds_npz['ids'][index] 
        X, rhythm = np.array(X, dtype='float32').reshape((1, 800)), rhythm.astype('float32')
        return X, rhythm, patient_id 

    def __len__(self):
        return len(self.ds_npz['signal'])

def getGenerator2(npz_file, indices_array, total_batch_size, replacement=False, is_train=True, shuffle=True):
    dataset = AFdataset2(npz_file)
    weights = np.zeros(len(npz_file['rhythm']))  

    # Assigning sampling weights
    for indices in indices_array:
        weights[indices] = (1 / len(indices_array)) * (1 / len(indices))

    if is_train:
        batchsampler = WeightedRandomSampler(weights=weights, num_samples=int(np.sum([len(k) for k in indices_array])),
                                             replacement=replacement)
        dataloader = DataLoader(dataset, batch_size=total_batch_size, num_workers=NUM_WORKERS,
                                sampler=batchsampler, pin_memory=True)
        return dataloader

    return DataLoader(dataset, batch_size=total_batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)

def get_weighted_generator2(file_location, batch_size, replacement, is_train=True, shuffle=True, percentage=1.0,
                           mmap=True):
    signal = np.load(os.path.join(file_location, 'signal.npy'), mmap_mode='r' if mmap else None)
    rhythm = np.load(os.path.join(file_location, 'rhythm.npy'))
    ids = np.load(os.path.join(file_location, 'ids.npy'))

    tr = {'signal': signal, 'rhythm': rhythm, 'ids': ids}

    # Categorize signals into different rhythm classes
    indices = []
    indices.append(np.where((tr['rhythm'][:, 0] == 1))[0])  # Non-AF
    indices.append(np.where((tr['rhythm'][:, 1] == 1))[0])  # AF

    for k in range(len(indices)):
        np.random.shuffle(indices[k])
        indices[k] = indices[k][0: int(len(indices[k]) * percentage)]

    return getGenerator2(tr, indices, batch_size, replacement=replacement, is_train=is_train, shuffle=shuffle)
