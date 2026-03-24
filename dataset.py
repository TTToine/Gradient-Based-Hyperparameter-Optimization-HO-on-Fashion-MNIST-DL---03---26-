import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np

class HyperCleaningDataset(Dataset):
    """
    Wrapper per il dataset di training che introduce rumore artificiale nelle etichette.
    Restituisce la tupla: (immagine, etichetta, indice).
    """
    def __init__(self, subset_dataset, corruption_rate=0.2, num_classes=10):
        self.dataset = subset_dataset
        self.corruption_rate = corruption_rate
        self.num_classes = num_classes
        
        self.num_samples = len(self.dataset)
        
        # Selezioniamo casualmente gli indici da corrompere
        num_corrupted = int(self.num_samples * self.corruption_rate)
        corrupted_idx_array = np.random.choice(self.num_samples, num_corrupted, replace=False)
        self.corrupted_indices = set(corrupted_idx_array)
        
        self.corrupted_labels_map = {}
        print(f"Iniezione del rumore: alterazione del {int(self.corruption_rate * 100)}% delle etichette ({num_corrupted} campioni)...")
        
        for idx in self.corrupted_indices:
            # Estraiamo l'etichetta originale e la modifichiamo
            # Utilizziamo la formula: label_corrotta = (label_originale + 1) % num_classi
            _, orig_label = self.dataset[idx]
            new_label = (orig_label + 1) % self.num_classes
            self.corrupted_labels_map[idx] = new_label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Se l'indice è tra quelli selezionati, applichiamo l'etichetta corrotta
        if idx in self.corrupted_indices:
            label = self.corrupted_labels_map[idx]
            
        return img, label, idx


def get_dataloaders(batch_size=128, data_dir='./data', use_augmentation=True, corruption_rate=0.0, num_workers=2):
    """
    Se use_augmentation=True → applica aug leggera solo al train set.
    """
    # Trasformazioni per VALIDATION e TEST 
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    # Trasformazioni per TRAIN
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])
    else:
        train_transform = val_test_transform
# Caricamento dataset originali (Creiamo DUE istanze separate per il training set)
    train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
    val_full   = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=val_test_transform) # <-- Nuova istanza senza augmentation
    
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=val_test_transform)

    # Split: 20k train, 5k val
    train_subset = Subset(train_full, range(20000))
    val_dataset  = Subset(val_full, range(20000, 25000)) # prende i dati da val_full

    train_dataset = HyperCleaningDataset(train_subset, corruption_rate=corruption_rate)

    # Inizializzazione dei DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Dati caricati! Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Augmentation attiva sul train: {'Sì' if use_augmentation else 'No'}")
    
    return train_loader, val_loader, test_loader

def get_noisy_val_loader(val_dataset, corruption_rate=0.2, batch_size=128, num_workers=2):
    """Genera un dataloader di validazione con etichette corrotte per il meta-learning."""
    noisy_val_dataset = HyperCleaningDataset(
        subset_dataset=val_dataset,
        corruption_rate=corruption_rate,
        num_classes=10
    )
    return DataLoader(
        noisy_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )