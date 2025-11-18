from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class MentalHealthDataset(Dataset):
    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def create_dataloaders(
    train_seqs,
    train_labels,
    val_seqs,
    val_labels,
    test_seqs,
    test_labels,
    batch_size: int = 32,
):
    train_ds = MentalHealthDataset(train_seqs, train_labels)
    val_ds = MentalHealthDataset(val_seqs, val_labels)
    test_ds = MentalHealthDataset(test_seqs, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
