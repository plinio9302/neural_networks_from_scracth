# From Autograd to Architectures: Building, Training, and Debugging Neural Networks

An end to end deep learning notebook in four parts. Each part builds a model from first principles, trains it, and walks through a real failure mode: **what broke, how it was diagnosed, and how it was fixed.**

**Author:** Plinio ([github.com/plinio9302](https://github.com/plinio9302))
**Notebook:** `neural_networks_from_scratch.ipynb`

## Overview

| Part | Model | Dataset | Failure mode studied | Diagnostic tool |
|---|---|---|---|---|
| 1 | Scalar autograd engine + MLP (Multi Layer Perceptron) | Two moons (100 points) | Recursion limit in topological sort | Gradient check vs. finite differences and PyTorch |
| 2 | CNN (Convolutional Neural Network) | scikit-learn digits (1797 images, 8 x 8) | Softmax applied before `cross_entropy` | "Overfit one batch" test + analytical loss floor |
| 3 | Transformer (built from scratch) | Synthetic sequence reversal | Missing positional embedding | Permutation equivariance test, attention maps |
| 4 | GNN (Graph Neural Network): GCN (Graph Convolutional Network) | Synthetic SBM (Stochastic Block Model) graph | Ignoring graph structure | Ablation: identity adjacency turns the GCN into an MLP |

Everything runs on a laptop CPU (Central Processing Unit) in a few minutes. No downloads are needed: datasets are bundled with scikit-learn or generated synthetically.

## Quick start

```bash
pip install torch numpy scikit-learn matplotlib jupyter
jupyter notebook neural_networks_from_scratch.ipynb
```

Run all cells top to bottom. Later parts reuse helpers defined earlier (`train`, `accuracy`, `History`), so cells are not independent. Seeds are fixed for Python, NumPy, and PyTorch, and a GPU (Graphics Processing Unit) is used automatically through CUDA (Compute Unified Device Architecture) if available.

Tested with Python 3.11.

## Part 1: Scalar autograd engine (micrograd style)

A `Value` class wraps a single float, records the operations applied to it, and implements reverse mode automatic differentiation.

* **Primitives:** `+`, `*`, `**` (constant exponent), `tanh`, `relu`, `exp`. Subtraction, division, and negation are derived from these.
* **Backward pass:** topological sort, then the chain rule in reverse order with gradient accumulation (`+=`), so a value used twice receives gradient from both uses.
* **Engineering detail:** the sort is an **iterative** DFS (Depth First Search) with an explicit stack. A recursive version hits Python's 1000 frame recursion limit; the notebook stress tests a graph 5000 operations deep.

**Worked example.** For $L = \tanh(ab + c) \cdot f$ with $a=2,\ b=-0.5,\ c=0.3,\ f=-2$:

| input | gradient |
|---|---|
| a | 0.6347 |
| b | -2.5390 |
| c | -1.2695 |
| f | -0.6044 |

These hand derived values are checked three ways: by hand, by central finite differences, and against PyTorch autograd in float64 (tolerance $10^{-9}$).

**Network.** `MicroMLP(2, [16, 16, 1])` with $48 + 272 + 17 = 337$ parameters, trained for 60 steps with hinge loss, L2 regularization ($\alpha = 10^{-4}$), and SGD (Stochastic Gradient Descent) with the learning rate decaying linearly from 1.0 toward 0.1. The output includes a decision boundary plot over the input plane.

## Part 2: CNN on handwritten digits

* **Data:** pixels scaled to $[0, 1]$, stratified 80/20 split (1437 train / 360 validation).
* **Architecture:** three `Conv 3x3 -> BatchNorm -> ReLU` blocks (32, 64, 64 channels), one 2 x 2 max pool, global average pooling, and a linear head. Total: **56,714 parameters**, of which only 650 are in the head thanks to global pooling.
* **Training utility:** one reusable `train()` function (AdamW optimizer) that logs training loss, validation accuracy, and the global gradient norm per step. It flattens logits and targets, so it also serves the Transformer in Part 3.

### The bug: double softmax

The first model ends with `softmax`, but `F.cross_entropy` already applies `log_softmax` internally. The inputs the loss sees are then bounded in $[0, 1]$, which creates a hard floor on the loss for 10 classes:

$$\text{loss}_{\min} = \log\left(1 + \frac{9}{e}\right) \approx 1.461$$

The **overfit one batch** test (300 steps on a single batch of 64) confirms it: the buggy model stalls at the floor while the fixed model drives the loss toward zero. **Fix:** return raw logits from `forward()` and apply softmax only at inference time.

Error analysis includes a confusion matrix and a gallery of misclassified validation digits with their predicted confidence.

## Part 3: Transformer from scratch

**Task:** reverse a sequence of 10 random digits. The target at position $i$ is the input at position $9 - i$. With $10^{10}$ possible inputs and 20,000 training sequences, validation accuracy measures a learned algorithm, not memorization.

* **Multi head self attention** with a fused QKV (Query, Key, Value) projection, 4 heads of 16 dimensions, and $\sqrt{d_k}$ scaling.
* **Pre LayerNorm blocks** (as in GPT, Generative Pre-trained Transformer, style models) with residual connections and a GELU (Gaussian Error Linear Unit) FFN (Feed Forward Network) expanding 64 to 128 dimensions.
* 2 layers, $d_{model} = 64$: **69,002 parameters** with positional embeddings (68,362 without).

### The bug: no positional information

Without positional embeddings, self attention is **permutation equivariant**: $f(Px) = P f(x)$. The model cannot tell position 0 from position 9, so reversal is impossible. The notebook tests this property directly with a random permutation rather than assuming it. **Fix:** a learned positional embedding (640 extra parameters).

Two metrics are reported: **token accuracy** and the much stricter **sequence accuracy** (all 10 tokens correct; 90% independent token accuracy would give only $0.9^{10} \approx 35\%$). Attention maps for every head are plotted to look for the anti diagonal pattern that implements reversal.

## Part 4: GCN from scratch

**Synthetic graph:** 400 nodes in 4 communities of 100. Edge probability is 0.10 within a community and 0.01 across, giving an expected degree of about 13 and about 77% same label neighbors. Node features are deliberately weak (signal std 0.25, noise std 1.0), so the information that matters lives in the edges. Split: 20% train (80 nodes), 20% validation, 60% test.

* **Layer:** $H' = \hat{A} H W$ with $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$.
* **Normalization check:** the notebook verifies $\hat{A}$ is symmetric with eigenvalues in $(-1, 1]$, and compares two hop aggregation on the highest and lowest degree nodes with and without normalization to show how raw sums let degree dominate feature scale.
* **Model:** 2 layers, hidden size 32, **676 parameters**, dropout 0.5, weight decay $5 \times 10^{-4}$, full batch training for 300 epochs, checkpoint selected on best validation accuracy. Test accuracy is reported only for that checkpoint.
* **Ablation:** passing the identity matrix instead of $\hat{A}$ turns the same model into an MLP, so the accuracy gap is attributable to the graph alone. PCA (Principal Component Analysis) plots of hidden embeddings show the cluster separation each model achieves.

## Results

The final cell prints a summary table with the before and after metric for each part:

| Part | Dataset (split) | Problem fixed | Metric |
|---|---|---|---|
| 1 | Two moons (train) | none | training accuracy |
| 2 | Digits (validation) | softmax before `cross_entropy` | validation accuracy and training loss, bug vs. fix |
| 3 | Reversal (validation) | no positional embedding | token accuracy, without vs. with positional embedding |
| 4 | SBM graph (test) | ignoring edges | test accuracy, MLP vs. GCN |

Exact numbers are produced when the notebook runs; the committed notebook has no saved outputs.

## Key lessons

1. **Know the contract of your loss function.** A double softmax does not crash; it silently puts a floor of 1.461 under the loss.
2. **Overfit one batch first.** It separates model and loss bugs from data and hyperparameter problems in under a minute.
3. **Turn hypotheses into tests.** The positional embedding bug was confirmed by checking $f(Px) = P f(x)$ in code.
4. **Ablate to prove what matters.** Same code, same parameters, only the adjacency differs.
5. **Verify internals, not only metrics.** Gradient checks and attention maps confirm the model works for the right reason.

## Notes and known limitations

* Part 1 reports **training** accuracy only; there is no held out set for the two moons model.
* The 28.6% token accuracy ceiling for the order blind Transformer is stated in the text from a 50,000 sequence simulation, but that simulation is not included as a cell.
* The GCN uses a dense adjacency matrix, which is fine for 400 nodes but does not scale to large graphs.

## Possible extensions

* Sparse adjacency (`torch.sparse`) for graphs with millions of nodes.
* Causal masking and autoregressive training for a character level language model.
* GAT (Graph Attention Network), connecting Parts 3 and 4.
* MNIST (Modified National Institute of Standards and Technology) or CIFAR-10 (Canadian Institute For Advanced Research) with data augmentation and a learning rate schedule.

## Tech stack

Python 3.11 · PyTorch · NumPy · scikit-learn · Matplotlib
