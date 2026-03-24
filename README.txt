Meta-Learning on FashionMNIST: Hyperparameter Optimization and Data Cleaning
Overview
This project demonstrates how to use meta-learning to automatically optimize the training process of a Convolutional Neural Network (CNN). Instead of manually guessing the best hyperparameters (like learning rates) or manually filtering out bad data, this project uses PyTorch and the higher library  to let the model "learn how to learn."

The project tests different approaches on the FashionMNIST dataset, comparing a standard training loop against advanced techniques that adjust parameters dynamically during training.

Project Structure
main.py: The main entry point. It handles command-line arguments and runs the selected experiment.

model.py: Contains the SimpleFashionCNN architecture.

dataset.py: Manages the FashionMNIST dataset, including train/validation splits, data augmentation, and injecting artificial noise for the data cleaning experiment.

utils.py: Helper functions for training loops, evaluation, early stopping, and generating charts.

genera_tabella.py: A utility script that reads the saved metrics and generates a summary table of the results.


requirements.txt: The list of dependencies with locked versions for full reproducibility.

Installation
It is recommended to use a virtual environment (such as venv or conda) to avoid conflicts with other local packages.

Clone this repository or navigate to the project folder:

Bash

cd path/to/your/project
Install the required dependencies:

Bash

pip install -r requirements.txt
How to Run the Experiments
The project is driven by command-line arguments. You can run four different types of experiments. All results, metrics (JSON), and generated charts (PNG) are automatically saved in the risultati_esperimenti/ folder.

You can append the --augmentation flag to any of these commands to apply data augmentation to the training set.

1. Baseline
Runs a standard training loop with a fixed learning rate. This serves as the reference point to measure the improvements of the other methods.

Bash

python main.py --experiment baseline
2. Reverse-Mode Meta-Learning
This method automatically learns and adjusts the learning rate for each layer of the network, as well as the overall weight decay. It unrolls the inner training loop to calculate exact gradients based on the validation loss.

Bash

python main.py --experiment reverse
3. Truncated Meta-Learning
A more memory-efficient version of the reverse-mode. Instead of keeping the entire training history in memory, it synchronizes the model weights at regular intervals. It achieves similar dynamic tuning but uses significantly less RAM.

Bash

python main.py --experiment truncated
4. Data Hyper-Cleaning
This experiment tests the model's ability to identify bad data. The dataset is intentionally corrupted with 20% incorrect labels. The meta-learning algorithm learns to assign a "weight" to each training sample, effectively pushing the weights of the corrupted images to zero so the model ignores them.

Bash

python main.py --experiment hyper_cleaning
Viewing the Results
After running the experiments, you can generate a clean markdown table summarizing the execution time, memory usage, and final test accuracy for each method:

Bash

python genera_tabella.py
