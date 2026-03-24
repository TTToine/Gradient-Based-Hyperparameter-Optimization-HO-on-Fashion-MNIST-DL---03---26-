import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SimpleFashionCNN(nn.Module):
    """
    Architettura CNN base per Fashion-MNIST.
    Implementa il Gradient Checkpointing per ridurre l'occupazione di memoria.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # Applichiamo il checkpointing ai blocchi convoluzionali.
        # use_reentrant=False è la best practice raccomandata nelle versioni recenti di PyTorch.
        # dummy_arg è un trucco per forzare il checkpointing anche se l'input x non ha requires_grad=True
        
        x = checkpoint(self.conv1, x, use_reentrant=False)
        x = checkpoint(self.conv2, x, use_reentrant=False)
        x = checkpoint(self.conv3, x, use_reentrant=False)
        
        # I layer fully connected pesano molto sui parametri ma poco sulle attivazioni,
        # quindi possiamo lasciarli calcolare normalmente senza rallentare il training.
        x = self.fc_layers(x)
        
        return x
