# Learned Optimization

This project implements stuff related to learned optimization.

```
## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install torch torchvision tqdm matplotlib
   ```

## Usage (very early)
### Meta-training the optimizer
```
python basic.py --train-meta --meta-epochs 3000 --inner-steps 20
```
This will train the learned optimizer on MNIST and save its weights to `learned_opt.pth`.

### Evaluating optimizers
```
python basic.py --epochs 5
```
This will compare the learned optimizer and Adam on MNIST, plotting training curves and printing test accuracy.

