# Gradient-Based-Hyperparameter-Optimization-HO-on-Fashion-MNIST-DL---03---26-
Brief: 
Gradient-based HO on Fashion-MNIST: Reproducing Franceschi (2017) and Maclaurin (2015) using the Higher library. Features: Reverse-mode AD, Truncated Meta-Learning, and Per-layer Learning Rate optimization.

Long:
Gradient-Based Hyperparameter Optimization on Fashion-MNIST
Gradient-based HO on Fashion-MNIST: Reproducing Franceschi (2017) and Maclaurin (2015) using the Higher library. Features: Reverse-mode AD, Truncated Meta-Learning, and Per-layer Learning Rate optimization.

Features
Reverse-Mode Automatic Differentiation: Computes exact hypergradients for hyperparameter optimization using the higher library.

Truncated Meta-Learning: Memory-efficient meta-learning approach utilizing weight synchronization across inner steps.

Data Hyper-Cleaning: Meta-learning sample weights to automatically identify and down-weight artificially injected label noise (20% corruption rate).

Layer-wise Learning Rate Optimization: Learns independent learning rates for individual convolutional blocks and fully connected layers.

Gradient Checkpointing: Memory optimization within the CNN architecture to support the heavy memory requirements of unrolled computation graphs.

Project Structure
main.py: The main entry point. Handles the execution of the baseline model, reverse-mode, truncated mode, and hyper-cleaning experiments.

model.py: Defines the SimpleFashionCNN architecture, integrating PyTorch's gradient checkpointing for memory efficiency.

dataset.py: Contains data loading logic and the HyperCleaningDataset wrapper, which systematically injects noise into the training labels.

utils.py: Core utility functions for training loops, evaluation, early stopping, and generating matplotlib visualizations of the training and hyperparameter trajectories.

genera_tabella.py: A reporting script that reads JSON output metrics and generates a summarized Markdown table comparing execution time, memory usage, and accuracy across experiments.

Requirements.txt: Project dependencies.

Installation
Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

Bash

pip install -r Requirements.txt
Usage
You can launch different experiments through main.py using the --experiment argument. Optional data augmentation can be enabled via the --augmentation flag.

Run the Baseline (Fixed Learning Rate):

Bash

python main.py --experiment baseline
Run Reverse-Mode Meta-Learning:

Bash

python main.py --experiment reverse
Run Truncated Meta-Learning:

Bash

python main.py --experiment truncated
Run Data Hyper-Cleaning (Sample Weight Meta-Learning):

Bash

python main.py --experiment hyper_cleaning
Enable Data Augmentation (Example on Baseline):

Bash

python main.py --experiment baseline --augmentation
Results and Metrics
All experimental results, including loss/accuracy plots, hyperparameter trajectory graphs, and JSON metric files, are automatically saved to the risultati_esperimenti directory.

To generate a comparative Markdown table of all completed experiments, run:

Bash

python genera_tabella.py
