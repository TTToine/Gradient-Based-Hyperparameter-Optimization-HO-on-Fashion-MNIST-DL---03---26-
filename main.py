import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import higher
import time
# Import necessari
from dataset import get_dataloaders
from model import SimpleFashionCNN
from utils import (
    set_seed,
    train_and_evaluate,
    evaluate,
    plot_training_history,
    plot_hyperparam_trajectory,
    save_experiment_metrics,
    train_with_sample_weights   # ← AGGIUNTA QUI
)
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
def run_baseline(device, train_loader, val_loader):
    print("\n🏁 Inizio addestramento Baseline (LR fisso = 0.001)")
    
    # Resettiamo il contatore della memoria se usiamo la GPU
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    model = SimpleFashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    losses, val_acc = train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, epochs=15, dev=device)
    
    end_time = time.time()
    elapsed_mins = (end_time - start_time) / 60.0
    
    # Calcoliamo il picco di memoria (in Megabyte)
    max_memory_mb = 0
    if device.type == 'cuda':
        max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    print(f"✅ Baseline completata in {elapsed_mins:.2f} minuti!")
    if device.type == 'cuda':
        print(f"💾 Picco di memoria GPU: {max_memory_mb:.2f} MB")
        
    return model, losses, val_acc, elapsed_mins, max_memory_mb

def run_reverse_mode(device, train_loader, val_loader):
    print("\n🧠 Inizio Meta-Learning (Reverse-Mode) - LR Layer-wise")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    # Inizializziamo 4 LR separati (uno per blocco), log(-4.605) = ~0.01
    log_lrs = nn.Parameter(torch.full((4,), -4.605, device=device))  
    log_wd = nn.Parameter(torch.tensor([-6.907], device=device))  # WD globale

    outer_epochs = 30
    K_inner = 35
    num_val_batches = 12

    outer_opt = optim.Adam([log_lrs, log_wd], lr=0.05)
    scheduler = CosineAnnealingLR(outer_opt, T_max=outer_epochs)
    
    meta_model = SimpleFashionCNN().to(device)
    
    param_groups = [
        {'params': meta_model.conv1.parameters()},
        {'params': meta_model.conv2.parameters()},
        {'params': meta_model.conv3.parameters()},
        {'params': meta_model.fc_layers.parameters()}
    ]
    inner_opt = optim.SGD(param_groups, lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    lr_history_mean = []
    wd_history = []
    val_loss_history = []

    for outer_step in range(outer_epochs):
        outer_opt.zero_grad()
        
        # FIX: Evitiamo esplosioni numeriche clampando i logaritmi
        log_lrs.data.clamp_(min=-10.0, max=0.0) 
        log_wd.data.clamp_(min=-10.0, max=-2.0)
        
        current_lrs = torch.exp(log_lrs)
        current_wd = torch.exp(log_wd)

        lr_history_mean.append(current_lrs.mean().item())
        wd_history.append(current_wd.item())

        with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=True, track_higher_grads=True) as (fmodel, diffopt):
            fmodel.train()
            for _ in range(K_inner):
                try: 
                    images, labels = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    images, labels = next(train_iter)
                images, labels = images.to(device), labels.to(device)
                inner_loss = criterion(fmodel(images), labels)
                
                diffopt.step(inner_loss, override={
                    'lr': [current_lrs[0], current_lrs[1], current_lrs[2], current_lrs[3]], 
                    'weight_decay': current_wd
                })

            fmodel.eval()
            val_loss_accumulated = 0.0
            for _ in range(num_val_batches):
                try: 
                    v_images, v_labels = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    v_images, v_labels = next(val_iter)
                v_images, v_labels = v_images.to(device), v_labels.to(device)
                val_loss_accumulated += criterion(fmodel(v_images), v_labels)

            avg_val_loss = val_loss_accumulated / num_val_batches
            avg_val_loss.backward()

        val_loss_history.append(avg_val_loss.item())

        torch.nn.utils.clip_grad_norm_([log_lrs, log_wd], max_norm=1.0)
        outer_opt.step()
        scheduler.step()

        if (outer_step + 1) % 5 == 0 or outer_step == 0:
            print(f"Outer {outer_step+1:2d}/{outer_epochs} | Val Loss: {avg_val_loss.item():.4f} | "
                  f"LRs: [{current_lrs[0]:.4f}, {current_lrs[1]:.4f}, {current_lrs[2]:.4f}, {current_lrs[3]:.4f}] | "
                  f"WD: {current_wd.item():.5f}")

    plot_hyperparam_trajectory(
        lr_history_mean,
        wd_history,
        val_loss_history,
        experiment_name="reverse_meta"
    )

    end_time = time.time()
    elapsed_mins = (end_time - start_time) / 60.0
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0

    print(f"✅ Reverse-Mode (Meta-Training) completata in {elapsed_mins:.2f} minuti!")
    if device.type == 'cuda':
        print(f"💾 Picco di memoria GPU: {max_memory_mb:.2f} MB")

    return current_lrs.detach().cpu().numpy(), current_wd.item(), elapsed_mins, max_memory_mb 

def run_truncated_mode(device, train_loader, val_loader):
    print("\n⚙️ Avvio Truncated Meta-Learning con Sincronizzazione Pesi - LR Layer-wise")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    log_lrs = nn.Parameter(torch.full((4,), -4.0, device=device))   
    log_wd = nn.Parameter(torch.tensor([-6.0], device=device))

    outer_epochs = 40
    inner_steps  = 12
    val_batches  = 4

    outer_opt = optim.Adam([log_lrs, log_wd], lr=0.08)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    meta_model = SimpleFashionCNN().to(device)
    
    param_groups = [
        {'params': meta_model.conv1.parameters()},
        {'params': meta_model.conv2.parameters()},
        {'params': meta_model.conv3.parameters()},
        {'params': meta_model.fc_layers.parameters()}
    ]
    inner_opt_dummy = optim.SGD(param_groups, lr=0.1, momentum=0.9)

    train_iter = iter(train_loader)
    val_iter   = iter(val_loader)

    lr_history_mean = []
    wd_history = []
    val_loss_history = []

    for outer_step in range(outer_epochs):
        outer_opt.zero_grad()
        
        # FIX: Evitiamo esplosioni numeriche clampando i logaritmi
        log_lrs.data.clamp_(min=-10.0, max=0.0) 
        log_wd.data.clamp_(min=-10.0, max=-2.0)
        
        current_lrs = torch.exp(log_lrs)
        current_wd = torch.exp(log_wd)

        lr_history_mean.append(current_lrs.mean().item())
        wd_history.append(current_wd.item())

        with higher.innerloop_ctx(
            meta_model,
            inner_opt_dummy,
            copy_initial_weights=True,
            track_higher_grads=True,
            override={
                'lr': [current_lrs[0], current_lrs[1], current_lrs[2], current_lrs[3]], 
                'weight_decay': current_wd
            }
        ) as (fmodel, diffopt):

            fmodel.train()

            for step_idx in range(inner_steps):
                try:
                    images, labels = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    images, labels = next(train_iter)

                images = images.to(device)
                labels = labels.to(device)

                logits = fmodel(images)
                inner_loss = criterion(logits, labels)
                diffopt.step(inner_loss)

            fmodel.eval()
            val_loss_accum = 0.0
            count = 0

            for _ in range(val_batches):
                try:
                    v_images, v_labels = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    v_images, v_labels = next(val_iter)

                v_images = v_images.to(device)
                v_labels = v_labels.to(device)

                with torch.enable_grad():
                    val_logits = fmodel(v_images)
                    batch_val_loss = criterion(val_logits, v_labels)
                    val_loss_accum += batch_val_loss
                    count += 1

            if count > 0:
                meta_loss = val_loss_accum / count
            else:
                meta_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if meta_loss.grad_fn is None:
                print("ATTENZIONE: meta_loss non ha grad_fn !")

            meta_loss.backward()

        val_loss_history.append(meta_loss.item())

        torch.nn.utils.clip_grad_norm_([log_lrs, log_wd], max_norm=10.0)
        outer_opt.step()

        with torch.no_grad():
            for p_src, p_tgt in zip(fmodel.parameters(), meta_model.parameters()):
                p_tgt.copy_(p_src)

        if outer_step % 5 == 0 or outer_step < 5:
            print(f"Outer {outer_step+1:3d}/{outer_epochs} | "
                  f"Meta Val Loss: {meta_loss.item():.4f} | "
                  f"LRs: [{current_lrs[0]:.4f}, {current_lrs[1]:.4f}, {current_lrs[2]:.4f}, {current_lrs[3]:.4f}] | "
                  f"WD: {current_wd.item():.6f}")

    plot_hyperparam_trajectory(
        lr_history_mean,
        wd_history,
        val_loss_history,
        experiment_name="truncated_meta"
    )

    end_time = time.time()
    elapsed_mins = (end_time - start_time) / 60.0
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0

    print(f"✅ Truncated-Mode (Meta-Training) completata in {elapsed_mins:.2f} minuti!")
    if device.type == 'cuda':
        print(f"💾 Picco di memoria GPU: {max_memory_mb:.2f} MB")

    return current_lrs.detach().cpu().numpy(), current_wd.item(), elapsed_mins, max_memory_mb
def run_hyper_cleaning(device, train_loader, val_loader, num_train_samples=20000):
    corrupted_indices = train_loader.dataset.corrupted_indices
    num_train_samples = len(train_loader.dataset)
    print("\n🧹 Avvio Data Hyper-Cleaning (Meta-Learning sui pesi dei campioni)")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # Inizializziamo i logit dei lambda a 0 (così la sigmoide parte da 0.5)
    # C'è un parametro per ogni singolo elemento del training set
    raw_lambdas = nn.Parameter(torch.zeros(num_train_samples, device=device))

    outer_epochs = 60
    K_inner = 20  # Numero di step del loop interno
    num_val_batches = 15

    # Ottimizzatore esterno per i pesi lambda
    outer_opt = optim.Adam([raw_lambdas], lr=0.2)
    scheduler = CosineAnnealingLR(outer_opt, T_max=outer_epochs)
    
    meta_model = SimpleFashionCNN().to(device)
    inner_opt = optim.SGD(meta_model.parameters(), lr=0.2, momentum=0.9)
    
    # IMPORTANTE: reduction='none' per poter moltiplicare la loss per i lambda_i
    criterion_inner = nn.CrossEntropyLoss(reduction='none')
    criterion_outer = nn.CrossEntropyLoss(reduction='mean')

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    val_loss_history = []
    lambda_mean_history = []
    # ──────────────────────────────────────────────────────────────
    # Aggiunta: Validation set RUMOROSO per testare meta-objective più sensibile al rumore
    # ──────────────────────────────────────────────────────────────
    from torchvision import datasets
    from torch.utils.data import Subset, DataLoader
    from dataset import HyperCleaningDataset
    # Trasformazioni di validazione (senza augmentation, come val_loader)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    train_full = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=val_transform
    )

    val_indices = range(20000, 25000)
    val_subset_original = Subset(train_full, val_indices)

    noisy_val_dataset = HyperCleaningDataset(
        subset_dataset=val_subset_original,
        corruption_rate=0.2,          # deve essere uguale al train
        num_classes=10
    )

    noisy_val_loader = DataLoader(
        noisy_val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    noisy_val_iter = iter(noisy_val_loader)

    print("Creato noisy_val_loader (20% corrupted) per meta-validation alternativa")
    for outer_step in range(outer_epochs):
        outer_opt.zero_grad()
        
        # Vincoliamo i lambda nell'intervallo [0, 1]
        lambdas = torch.sigmoid(raw_lambdas)
        lambda_mean_history.append(lambdas.mean().item())

        with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=True, track_higher_grads=True) as (fmodel, diffopt):
            fmodel.train()
            for _ in range(K_inner):
                try: 
                    # Ora il train_loader restituisce 3 elementi
                    images, labels, indices = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    images, labels, indices = next(train_iter)
                    
                images, labels = images.to(device), labels.to(device)
                
                # Loss non ridotta: vettore di dimensione [batch_size]
                unreduced_loss = criterion_inner(fmodel(images), labels)
                
                # Selezioniamo i lambda corrispondenti agli indici di questo batch e calcoliamo la media pesata
                batch_lambdas = lambdas[indices]
                inner_loss = torch.mean(unreduced_loss * batch_lambdas)
                
                diffopt.step(inner_loss)

            fmodel.eval()
            val_loss_accumulated = 0.0
            
           # --- FINE DEL LOOP INTERNO (K_inner) ---

            # Validation PULITA (il nostro oracolo)
            fmodel.eval()
            val_loss_accumulated = 0.0
            for _ in range(num_val_batches):
                try: 
                    v_images, v_labels = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    v_images, v_labels = next(val_iter)
                    
                v_images, v_labels = v_images.to(device), v_labels.to(device)
                val_loss_accumulated += criterion_outer(fmodel(v_images), v_labels)

            avg_val_loss_clean = val_loss_accumulated / num_val_batches

            # Validation RUMOROSA (solo per monitoraggio, NON entra nel backward!)
            val_loss_accumulated_noisy = 0.0
            for _ in range(num_val_batches):
                try: 
                    v_images, v_labels, _ = next(noisy_val_iter)
                except StopIteration:
                    noisy_val_iter = iter(noisy_val_loader)
                    v_images, v_labels, _ = next(noisy_val_iter)
                    
                v_images, v_labels = v_images.to(device), v_labels.to(device)
                val_loss_accumulated_noisy += criterion_outer(fmodel(v_images), v_labels)

            avg_val_loss_noisy = val_loss_accumulated_noisy / num_val_batches

            # ──────────────────────────────────────────────────────────────
            # META-OBIETTIVO E BACKWARD (Rigorosamente UN SOLO .backward)
            # ──────────────────────────────────────────────────────────────
            # Penalità L1 per forzare a zero gli esempi inutili/corrotti
            l1_penalty = 0.005 * torch.mean(lambdas)
            meta_objective = avg_val_loss_clean + l1_penalty
            
            # Unico e solo backward! Assicurati che non ce ne siano altri in giro.
            meta_objective.backward()
            
            val_loss_history.append(avg_val_loss_clean.item())
            outer_opt.step()
            scheduler.step()

            # ──────────────────────────────────────────────────────────────
            # CURA PER IL "GIORNO DELLA MARMOTTA": TRAVASO DEI PESI
            # ──────────────────────────────────────────────────────────────
            with torch.no_grad():
                for p_src, p_tgt in zip(fmodel.parameters(), meta_model.parameters()):
                    p_tgt.copy_(p_src)

        # Stampa di fine epoca (allineata al loop for)
        if (outer_step + 1) % 5 == 0 or outer_step == 0:
            print(f"Outer {outer_step+1:2d}/{outer_epochs} | "
                  f"Val Loss (clean): {avg_val_loss_clean.item():.4f} | "
                  f"Val Loss (noisy): {avg_val_loss_noisy.item():.4f} | "
                  f"Lambda Medio: {lambdas.mean().item():.4f}")
    end_time = time.time()
    elapsed_mins = (end_time - start_time) / 60.0
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0

    print(f"✅ Hyper-Cleaning completato in {elapsed_mins:.2f} minuti!")
    if device.type == 'cuda':
        print(f"💾 Picco di memoria GPU: {max_memory_mb:.2f} MB")

    # Restituiamo i lambda effettivi nel range [0, 1]
    final_lambdas = torch.sigmoid(raw_lambdas).detach().cpu().numpy()
    return final_lambdas, corrupted_indices, elapsed_mins, max_memory_mb
def main():
    parser = argparse.ArgumentParser(description="Gradient-Based Hyperparameter Optimization")
    
    parser.add_argument('--experiment', type=str, default='baseline',
                        choices=['baseline', 'reverse', 'truncated', 'hyper_cleaning'],
                        help="Scegli l'esperimento: 'baseline', 'reverse', 'truncated', 'hyper_cleaning'")
    
    parser.add_argument('--augmentation', action='store_true',
                        help="Attiva data augmentation leggera sul train set")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device attivo: {device}")
    set_seed(29)
    os.makedirs("risultati_esperimenti", exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=128,
        data_dir='./data',
        use_augmentation=args.augmentation
    )

    if args.experiment == 'baseline':
        # FIX: Riceviamo il modello addestrato
        model_baseline, losses, val_acc, elapsed_mins, max_memory_mb = run_baseline(device, train_loader, val_loader)
        
        print("\n📊 Generazione grafici baseline")
        plot_training_history(
            losses,
            val_acc,
            title_prefix="Baseline",
            experiment_name="baseline"
        )
        
        print("\n📊 Valutazione FINALE sul TEST set per la Baseline")
        test_loss, test_accuracy = evaluate(model_baseline, test_loader, nn.CrossEntropyLoss(), device, phase='Test')
        
        # Creiamo il dizionario ordinato e lo salviamo!
        metrics = {
            "experiment": "baseline",
            "test_accuracy_percent": round(test_accuracy, 2),
            "test_loss": round(test_loss, 4),
            "execution_time_minutes": round(elapsed_mins, 2),
            "peak_memory_mb": round(max_memory_mb, 2)
        }
        save_experiment_metrics("baseline", metrics)
    elif args.experiment == 'reverse':
        best_lrs, best_wd, meta_time, meta_memory = run_reverse_mode(device, train_loader, val_loader)
        print(f"\n🚀 Training finale con LR appresi: conv1={best_lrs[0]:.5f}, conv2={best_lrs[1]:.5f}, conv3={best_lrs[2]:.5f}, fc={best_lrs[3]:.5f}")
        
        model_hyper = SimpleFashionCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        
        param_groups_final = [
            {'params': model_hyper.conv1.parameters(), 'lr': best_lrs[0]},
            {'params': model_hyper.conv2.parameters(), 'lr': best_lrs[1]},
            {'params': model_hyper.conv3.parameters(), 'lr': best_lrs[2]},
            {'params': model_hyper.fc_layers.parameters(), 'lr': best_lrs[3]}
        ]
        
        optimizer_hyper = optim.SGD(param_groups_final, momentum=0.9, weight_decay=best_wd)
        
        losses_final, val_accs_final = train_and_evaluate(
            model_hyper, optimizer_hyper, criterion, train_loader, val_loader, epochs=15, dev=device
        )
        
        print("\n📊 Generazione grafici training finale (reverse)")
        plot_training_history(
            losses_final,
            val_accs_final,
            title_prefix="Final Training (Reverse-Mode)",
            experiment_name="reverse_final"
        )
        
        print("\n📊 Valutazione FINALE sul TEST set (10.000 immagini reali)")
        test_loss, test_accuracy = evaluate(model_hyper, test_loader, criterion, device, phase='Test')
        
        metrics = {
            "experiment": "reverse",
            "test_accuracy_percent": round(test_accuracy, 2),
            "test_loss": round(test_loss, 4),
            "meta_learning_time_minutes": round(meta_time, 2),
            "meta_learning_peak_memory_mb": round(meta_memory, 2),
            "learned_lrs": [float(lr) for lr in best_lrs],
            "learned_wd": float(best_wd)
        }
        save_experiment_metrics("reverse", metrics)
    elif args.experiment == 'hyper_cleaning':
        # 1. Fase Meta-Learning (impariamo i lambda)
        final_lambdas, corrupted_idxs, meta_time, meta_memory = run_hyper_cleaning(
            device, train_loader, val_loader
        )
        
        # 2. Analisi qualitativa
        from utils import analyze_hyper_cleaning
        analyze_hyper_cleaning(final_lambdas, corrupted_idxs)

        print(f"\n🚀 Avvio Training Finale con pesi lambda + CosineAnnealingLR + Early Stopping...")

        model_final = SimpleFashionCNN().to(device)
        optimizer_final = optim.SGD(
            model_final.parameters(), 
            lr=0.01, 
            momentum=0.9, 
            weight_decay=1e-4
        )

        lambda_tensor = torch.from_numpy(final_lambdas).to(device)

        # === TRAINING ELEGANTISSIMO (con scheduler + early stopping) ===
        losses_final, val_accs_final = train_with_sample_weights(
            model_final,
            optimizer_final,
            train_loader,
            val_loader,
            lambda_tensor,
            epochs=15,
            dev=device,
            use_scheduler=True,
            patience=5,
            min_delta=0.2
        )

        print("\n📊 Generazione grafici training finale (Hyper-Cleaning)")
        plot_training_history(
            losses_final,
            val_accs_final,
            title_prefix="Final Training (Hyper-Cleaning)",
            experiment_name="hyper_cleaning_final"
        )

        print("\n📊 Valutazione FINALE sul TEST set")
        test_loss, test_accuracy = evaluate(model_final, test_loader, 
                                           nn.CrossEntropyLoss(), device, phase='Test')

        # Salvataggio metriche
        metrics = {
            "experiment": "hyper_cleaning",
            "test_accuracy_percent": round(test_accuracy, 2),
            "test_loss": round(test_loss, 4),
            "meta_learning_time_minutes": round(meta_time, 2),
            "meta_learning_peak_memory_mb": round(meta_memory, 2),
            "mean_lambda_clean": float(final_lambdas[list(set(range(len(final_lambdas))) - corrupted_idxs)].mean()),
            "mean_lambda_corrupted": float(final_lambdas[list(corrupted_idxs)].mean())
        }
        save_experiment_metrics("hyper_cleaning", metrics)
if __name__ == "__main__":
    main()