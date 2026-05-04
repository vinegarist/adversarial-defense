# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project on **adversarial defense algorithms** for deep learning, specifically targeting the MNIST dataset. The project implements and compares various adversarial attack methods and defense strategies, with a focus on occlusion-based attacks and adversarial training.

Key research areas:
- Label Smoothing
- Adversarial Training (FGSM, PGD, CW, MSD)
- Occlusion-based Attacks (Saliency-based, Integrated Gradients-based)
- Transfer Attacks
- AutoAttack evaluation

## Project Structure

### Core Python Modules

| File | Purpose |
|------|---------|
| `models.py` | Neural network architectures (LR, LR10, FCNet, LeNet5) |
| `loss.py` | Loss functions (Label Smoothing Cross Entropy, CW Loss) |
| `pgd.py` | Gradient-based attacks (LinfPGD, L2PGD, FGSM, CWAttack, MSDAttack) |
| `occlusion_attack.py` | Occlusion-based attacks using saliency/IG maps |
| `adversarial_training.py` | Wrapper classes for adversarial training |
| `utils.py` | Data loading utilities (MNIST) |
| `test.py` | Model evaluation functions |

### Experiment Notebooks

The main experiments are organized in numbered Jupyter notebooks:

- `1. 对抗性防御算法之Label Smoothing.ipynb` - Label smoothing baseline
- `4. 对抗性攻击算法之迁移攻击.ipynb` - Transfer attack experiments
- `5. 对抗性攻击算法之AutoAttack.ipynb` - AutoAttack evaluation
- `6-9. 对抗性防御算法之对抗性训练.ipynb` - Various AT methods
- `10-12. 对抗性防御算法之快速/免费/遮蔽攻击对抗性训练.ipynb` - Advanced AT
- `13-17. 全攻击类型评测/多参数评测/IG遮蔽/Inequality遮蔽.ipynb` - Comprehensive evaluation

### Scripts

| Script | Purpose |
|--------|---------|
| `generate_paper_figures.py` | Generate all paper figures (Chapter 4) |
| `generate_thesis_figures.py` | Generate thesis-specific figures |
| `eval_at_model_5_3.py` | Transfer attack evaluation example |
| `export_to_thesis.py` | Export results to thesis directory |
| `scripts/viz_supplementary.py` | Supplementary visualizations |
| `scripts/analysis_supplementary.py` | Supplementary analysis |

### Directory Structure

```
save_model/           # Trained model checkpoints
├── 1epoch/          # 1 epoch models
├── 5epoch/          # 5 epoch models
├── 10epoch/         # 10 epoch models
├── 25epoch/         # 25 epoch models
├── 50epoch/         # 50 epoch models (main results)
├── 100epoch/        # 100 epoch models
└── 200epoch/        # 200 epoch models

results_figures/     # Generated plots and CSV results
paper_figures/       # Paper-ready figures
├── v2/             # Version 2 figures organized by category
│   ├── 4atk/       # 4-attack comparisons
│   ├── ada_inner/  # Adaptive attack inner analysis
│   ├── fix_vs_ada/ # Fixed vs adaptive comparisons
│   ├── min/        # Minimum attack analysis
│   ├── model_cmp/  # Model comparisons
│   └── sal_overlay/# Saliency overlays

auto-attack/auto-attack-master/  # AutoAttack library
```

## Common Development Commands

### Environment Setup

```bash
# The project uses a local virtual environment at .venv/
# Activate with:
source .venv/Scripts/activate  # Windows Git Bash
# or
.venv\Scripts\activate.bat     # Windows CMD

# Key dependencies are in requirements.txt
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run a Jupyter notebook experiment
jupyter notebook "15. Saliency遮蔽攻击对抗性训练.ipynb"

# Run a Python script
python generate_paper_figures.py

# Run transfer attack evaluation
python eval_at_model_5_3.py
```

### Generate Paper Figures

```bash
# Generate all Chapter 4 figures
python generate_paper_figures.py

# Generate thesis figures
python generate_thesis_figures.py

# Run supplementary visualizations
python scripts/viz_supplementary_v3.py
```

### Model Evaluation

Models are evaluated using the `test.test()` function:

```python
from test import test

# Clean accuracy
acc, preds = test(model, samples, labels, bs=100, mode='clean')

# Attack evaluation (model includes attack)
acc, preds = test(model_with_attack, samples, labels, bs=100, mode='attack')
```

## Architecture Overview

### Attack Classes (in `pgd.py` and `occlusion_attack.py`)

All attack classes inherit from `nn.Module` and implement:

```python
class AttackClass(nn.Module):
    def __init__(self, net, ...):
        self.net = net  # Target model

    def forward(self, inputs):
        x, y = inputs   # images and labels
        # Generate adversarial examples
        return x_adv    # adversarial images
```

| Attack Class | File | Description |
|--------------|------|-------------|
| `LinfPGD` | pgd.py | L-infinity PGD attack |
| `L2PGD` | pgd.py | L2 norm PGD attack |
| `FGSM` | pgd.py | Fast Gradient Sign Method |
| `CWAttack` | pgd.py | Carlini-Wagner attack |
| `MSDAttack` | pgd.py | Multi-Scale Diversity attack |
| `SaliencyOcclusionAttack` | occlusion_attack.py | Fixed occlusion using simple gradients |
| `AdaptiveSaliencyOcclusionAttack` | occlusion_attack.py | Adaptive occlusion using simple gradients |
| `AdaptiveIGOcclusionAttack` | occlusion_attack.py | Adaptive occlusion using Integrated Gradients |
| `IGFixedOcclusionAttack` | occlusion_attack.py | Fixed occlusion using Integrated Gradients |

### Adversarial Training Wrappers (in `adversarial_training.py`)

Wrappers combine models with attacks for training:

```python
class AdversarialTraining(nn.Module):
    def __init__(self, model, eps=0.1, step=5, ...):
        self.model = model
        self.adversary = LinfPGD(model, ...)
        self.is_at = False  # Set True during training

    def forward(self, x, y=None):
        if self.is_at and y is not None:
            x_adv = self.adversary((x, y))
            return self.model(x_adv)
        return self.model(x)
```

### Model Architectures (in `models.py`)

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| `LR` | 28×28 | 1 | Binary MNIST (0 vs 1) |
| `LR10` | 28×28 | 10 | Simple linear MNIST |
| `FCNet` | 28×28 | 10 | 3-layer FC network |
| `LeNet5` | 28×28×1 | 10 | Main CNN architecture for experiments |

## Key Parameters

### Occlusion Attack Parameters

- `N` (max_regions): Maximum number of occlusion regions (typically 5)
- `R` (max_radius): Maximum radius of each occlusion region (typically 3)
- `c` (occlu_color): Occlusion fill color, 0=black, 0.5=gray, 1=white
- `top_k`: Number of regions for fixed occlusion attacks (typically 9)
- `kernel_size`: Size of occlusion window for fixed attacks (typically 3)

### PGD Attack Parameters

- `eps` (ε): Maximum perturbation (typically 0.1 or 8/255 for Linf)
- `step_size` (α): Step size per iteration (typically 0.025 or 2/255)
- `step`: Number of iterations (typically 10-40)
- `random_start`: Whether to use random initialization

## Important Implementation Notes

1. **Model State Format**: Saved models may contain either direct state_dict or a dict with 'net' key:
   ```python
   checkpoint = torch.load(path)
   if isinstance(checkpoint, dict) and 'net' in checkpoint:
       model.load_state_dict(checkpoint['net'])
   else:
       model.load_state_dict(checkpoint)
   ```

2. **Attack Mode in Testing**: When testing with attacks, use `mode='attack'` and pass `(x, y)` tuple:
   ```python
   test(nn.Sequential(attack, model), x, y, mode='attack')
   ```

3. **Captum Dependency**: IG-based attacks require `captum` library. Code gracefully handles its absence by falling back to Saliency-based attacks.

4. **Data Normalization**: Models expect MNIST data normalized with mean=0.1307, std=0.3081 (handled in model forward).

5. **Chinese Text in Plots**: Plotting scripts configure matplotlib for Chinese text rendering using Microsoft YaHei or SimHei fonts.
