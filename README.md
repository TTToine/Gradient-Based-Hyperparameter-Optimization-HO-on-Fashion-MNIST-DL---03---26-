# Gradient-Based Hyperparameter Optimization on Fashion-MNIST

Hyperparameters learned by **differentiating the validation loss through the training
procedure itself**. The repo implements bilevel optimization with PyTorch and
[`higher`](https://github.com/facebookresearch/higher) and compares three variants
against a fixed-hyperparameter baseline:

| Experiment | What is learned | Hypergradient |
|---|---|---|
| `baseline` | nothing (SGD, lr = 1e-3, wd = 1e-4) | — |
| `reverse` | 4 layer-wise learning rates + weight decay | reverse-mode through K = 35 unrolled SGD steps |
| `truncated` | 4 layer-wise learning rates + weight decay | truncated unrolling (12 steps), weights carried over between outer steps |
| `hyper_cleaning` | one weight per training sample (20 000) | reverse-mode through K = 20 steps, with 20% corrupted labels |

## Method

Bilevel problem: hyperparameters $\lambda$ minimise the validation loss of the weights
obtained by training with them,

$$
\min_{\lambda}\; \mathcal{L}_{\text{val}}\big(w_K(\lambda)\big),
\qquad w_{k+1} = w_k - \eta(\lambda)\,\nabla_w \mathcal{L}_{\text{train}}(w_k, \lambda).
$$

`higher` keeps the inner SGD steps differentiable, so $\nabla_\lambda \mathcal{L}_{\text{val}}$
is obtained by backpropagating through the unrolled training trajectory
(reverse-mode hypergradient, Maclaurin et al. 2015; Franceschi et al. 2017).
Learning rates and weight decay are optimised in log-space with Adam and clipped
to a safe range.

- **Reverse mode**: each outer step unrolls K = 35 inner steps from the same initial
  weights. Exact hypergradient for that horizon; memory grows linearly with K.
- **Truncated mode**: unrolls only 12 steps and then copies the trained weights back
  into the model, so training continues across outer steps. Much cheaper in memory,
  at the cost of a biased (short-horizon) hypergradient (Shaban et al. 2019).
- **Data hyper-cleaning**: 20% of training labels are corrupted with $y \to (y+1) \bmod 10$.
  Each sample gets a weight $\sigma(\theta_i) \in [0,1]$ in the inner loss; the outer
  objective is the clean validation loss plus a small L1 penalty on the weights.
  The method should push weights of corrupted samples towards zero.

After meta-training, a fresh model is trained for 15 epochs with the learned
hyperparameters (or sample weights) and evaluated once on the test set.

**Setup.** Fashion-MNIST, 20 000 training / 5 000 validation images (both from the
official training split), 10 000 test images. Small CNN: three conv blocks
(32–64–128 channels) and a two-layer classifier with dropout. Seed 29.

## Results

Each run saves metrics (test accuracy, test loss, wall time, peak GPU memory) to
`risultati_esperimenti/<experiment>_metrics.json`, plus training curves and
hyperparameter trajectories as PNG. Hyper-cleaning also saves the distribution of
learned sample weights for clean vs corrupted samples.

To build the comparison table after running the experiments:

```bash
python genera_tabella.py
```

<!-- TODO: paste the generated table and the key figures here -->

## Limitations

- Single seed; differences between methods should be checked over several seeds.
- Short unrolling horizons (12–35 steps) give short-horizon bias in the hypergradient.
- Peak memory is measured only on GPU.

## Reproduce

Python 3.10+, CUDA GPU recommended.

```bash
pip install -r requirements.txt
python main.py --experiment baseline        # also: reverse | truncated | hyper_cleaning
python main.py --experiment reverse --augmentation   # optional light augmentation
```

Fashion-MNIST is downloaded automatically into `./data`.

## Repository structure

```
main.py            # experiments and CLI
dataset.py         # data splits, augmentation, label-noise wrapper
model.py           # SimpleFashionCNN
utils.py           # training loops, evaluation, plots, metrics I/O
genera_tabella.py  # summary table from saved metrics
requirements.txt
```

## References

- D. Maclaurin, D. Duvenaud, R. Adams. *Gradient-based Hyperparameter Optimization through Reversible Learning.* ICML 2015.
- L. Franceschi, M. Donini, P. Frasconi, M. Pontil. *Forward and Reverse Gradient-Based Hyperparameter Optimization.* ICML 2017.
- A. Shaban, C.-A. Cheng, N. Hatch, B. Boots. *Truncated Back-propagation for Bilevel Optimization.* AISTATS 2019.
- E. Grefenstette et al. *Generalized Inner Loop Meta-Learning* (`higher`). arXiv 2019.
