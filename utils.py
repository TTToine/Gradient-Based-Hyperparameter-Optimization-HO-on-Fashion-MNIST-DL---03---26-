import torch
import random
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
def set_seed(seed=29):
    """Imposta il seed per la riproducibilità degli esperimenti."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_evaluate(model, optimizer, criterion, train_dl, val_dl, epochs=15, dev='cuda'):
    """
    Versione aggiornata per gestire batch con o senza indici (Hyper-Cleaning).
    """
    train_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_dl:
            # Spacchettamento flessibile: prendiamo i primi due, ignoriamo il resto (indices)
            images, labels = batch[0].to(dev), batch[1].to(dev)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dl.dataset)
        train_losses.append(epoch_loss)

        # Validation (usiamo la funzione evaluate definita sotto)
        _, acc = evaluate(model, val_dl, criterion, dev, phase='Validation', silent=True)
        val_accs.append(acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {acc:.2f}%")

    return train_losses, val_accs

def evaluate(model, loader, criterion, device, phase='Validation', silent=False):
    """
    Valuta il modello. Gestisce batch di lunghezza variabile (2 o 3 elementi).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            # Funziona sia con (img, label) che con (img, label, idx)
            images, labels = batch[0].to(device), batch[1].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    if not silent:
        print(f"{phase} Loss: {avg_loss:.4f} | {phase} Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
import matplotlib.pyplot as plt
import os

def plot_training_history(
    train_losses,
    val_accs,
    title_prefix="",
    save_dir="risultati_esperimenti",
    experiment_name="experiment"
):
    """
    Genera e salva due grafici: train loss e val accuracy.
    Salva in save_dir con nome basato su experiment_name.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Figura 1: Train Loss
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} – Training Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    loss_path = os.path.join(save_dir, f"{experiment_name}_train_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.show()  # mostra a video se stai in Jupyter o ambiente interattivo
    plt.close()
    
    # Figura 2: Validation Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title_prefix} – Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    acc_path = os.path.join(save_dir, f"{experiment_name}_val_acc.png")
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Grafici salvati in:\n  - {loss_path}\n  - {acc_path}")


def plot_hyperparam_trajectory(
    lr_history,
    wd_history,
    val_loss_history,
    save_dir="risultati_esperimenti",
    experiment_name="meta_learning"
):
    """
    Grafico per meta-learning: evoluzione di LR, WD e meta-val loss.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(lr_history) + 1)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # Asse sinistro: LR e WD
    ax1.set_xlabel('Outer Step')
    ax1.set_ylabel('Hyperparameters (log scale)', color='tab:blue')
    ax1.plot(epochs, lr_history, label='Learning Rate', color='blue', linewidth=2)
    ax1.plot(epochs, wd_history, label='Weight Decay', color='cyan', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Asse destro: meta val loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Meta Validation Loss', color='tab:orange')
    ax2.plot(epochs, val_loss_history, label='Meta Val Loss', color='orange', linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    # Titolo e legenda combinata
    plt.title(f'{experiment_name} – Hyperparameter Trajectory & Meta Loss')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{experiment_name}_hyperparam_trajectory.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Traiettoria iperparametri salvata in: {save_path}")
def save_experiment_metrics(experiment_name, metrics_dict, save_dir="risultati_esperimenti"):
    """Salva le metriche in un file JSON per un confronto ordinato."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{experiment_name}_metrics.json")
    
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print(f" Metriche salvate con successo in: {filepath}")
def analyze_hyper_cleaning(lambdas, corrupted_indices, save_dir="risultati_esperimenti"):
    """
    Confronta i valori di lambda assegnati agli esempi corrotti rispetto a quelli puliti.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_indices = set(range(len(lambdas)))
    clean_indices = list(all_indices - corrupted_indices)
    corrupted_indices = list(corrupted_indices)

    lambdas_clean = lambdas[clean_indices]
    lambdas_corrupted = lambdas[corrupted_indices]

    print("\n🔍 Analisi Hyper-Cleaning:")
    print(f"  - Lambda medio (esempi PULITI): {lambdas_clean.mean():.4f}")
    print(f"  - Lambda medio (esempi CORROTTI): {lambdas_corrupted.mean():.4f}")
    
    # Istogramma di confronto
    plt.figure(figsize=(10, 6))
    plt.hist(lambdas_clean, bins=50, alpha=0.5, label='Esempi Puliti', color='green')
    plt.hist(lambdas_corrupted, bins=50, alpha=0.5, label='Esempi Corrotti', color='red')
    plt.title('Distribuzione dei pesi $\lambda_i$ appresi')
    plt.xlabel('Valore $\lambda$ (Weight)')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = os.path.join(save_dir, "hyper_cleaning_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Grafico della distribuzione salvato in: {save_path}")
def train_with_sample_weights(
    model, optimizer, train_dl, val_dl, sample_weights,
    epochs=15, dev='cuda',
    use_scheduler=True,          # CosineAnnealingLR
    patience=5,                  # Early Stopping (0 = disattivato)
    min_delta=0.2                # miglioramento minimo in accuracy (%)
):
    """
    Training con pesi per campione + CosineAnnealingLR + Early Stopping.
    Stessa interfaccia e output di train_and_evaluate.
    """
    criterion_weighted = nn.CrossEntropyLoss(reduction='none')
    train_losses, val_accs = [], []

    sample_weights = sample_weights.to(dev)

    # === SCHEDULER (opzionale) ===
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        print("📉 CosineAnnealingLR attivato (T_max = epochs)")

    # === EARLY STOPPING ===
    best_val_acc = 0.0
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels, indices in train_dl:
            images = images.to(dev)
            labels = labels.to(dev)
            indices = indices.to(dev)

            optimizer.zero_grad()
            outputs = model(images)
            
            per_sample_loss = criterion_weighted(outputs, labels)
            weighted_loss = torch.mean(per_sample_loss * sample_weights[indices])
            
            weighted_loss.backward()
            optimizer.step()

            running_loss += weighted_loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dl.dataset)
        train_losses.append(epoch_loss)

        # Validation
        val_loss, val_acc = evaluate(model, val_dl, nn.CrossEntropyLoss(), dev,
                                     phase='Validation', silent=True)
        val_accs.append(val_acc)

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # === EARLY STOPPING ===
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"⏹️  Early stopping attivato all'epoca {epoch+1} (miglior Val Acc: {best_val_acc:.2f}%)")
            early_stopped = True
            break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:2d}/{epochs} | Weighted Loss: {epoch_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

    if not early_stopped:
        print(f" Training completato ({epochs} epoche). Miglior Val Acc: {max(val_accs):.2f}%")

    return train_losses, val_accs
def get_infinite_iterator(dataloader):
    """Crea un iteratore infinito per i loop interni di meta-learning."""
    while True:
        for batch in dataloader:
            yield batch