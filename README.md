# Transmod

[![PyPI version](https://badge.fury.io/py/transmod.svg)](https://badge.fury.io/py/transmod)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/transmod/badge/?version=latest)](https://transmod.readthedocs.io/en/latest/?badge=latest)

Multi-modal deep learning framework for predicting translation initiation mechanisms.

## Overview

Transmod integrates ribosome profiling, RNA structure, modifications, and protein embeddings to predict translation outcomes under initiation factor perturbations. The framework enables:

- 🧬 Multi-modal integration of sequence, structure, and modification data
- 🤖 Factor-agnostic predictions through protein structural embeddings
- 🔍 Interpretable deep learning with attribution methods
- 🎯 Prediction of ribosome occupancy and translation efficiency

## Installation

Transmod is organized with modular dependencies so you can install only what you need for your specific workflow.

### Quick Start (Recommended)

For most users, install with classical ML or deep learning support:

```bash
# For classical machine learning workflows
pip install -e ".[classical-ml]"

# For deep learning workflows  
pip install -e ".[deep-learning]"

# For complete functionality
pip install -e ".[complete]"
```

### Modular Installation Options

#### Core Installation (Minimal)
```bash
# Just basic data handling (numpy, pandas, scipy, etc.)
pip install -e .
```

#### By Use Case
```bash
# Classical ML: scikit-learn, LightGBM + bioinformatics + plotting
pip install -e ".[classical-ml]"

# Deep Learning: PyTorch, Lightning + bioinformatics + plotting  
pip install -e ".[deep-learning]"

# Full Analysis: All ML tools + interpretation
pip install -e ".[full-analysis]"

# Complete: Everything including Jupyter support
pip install -e ".[complete]"
```

#### By Component
```bash
# Individual components
pip install -e ".[bio]"        # BioPython for sequence analysis
pip install -e ".[ml]"         # Classical ML (scikit-learn, LightGBM)
pip install -e ".[dl]"         # Deep learning (PyTorch, Lightning)
pip install -e ".[plots]"      # Visualization (matplotlib, seaborn)
pip install -e ".[interp]"     # Model interpretation (SHAP, Captum)
pip install -e ".[notebook]"   # Jupyter notebook support
pip install -e ".[dev]"        # Development tools

# Custom combinations
pip install -e ".[bio,ml,interp]"  # Classical ML + interpretation
pip install -e ".[bio,dl,plots,dev]"  # Deep learning + development
```

### For Snakemake Workflows

Create environment files for specific workflows:

**Classical ML environment** (`envs/classical_ml.yaml`):
```yaml
name: transmod-classical
channels:
  - conda-forge
  - bioconda
dependencies:
  - python>=3.12
  - pip
  - pip:
    - -e ".[classical-ml]"
```

**Deep Learning environment** (`envs/deep_learning.yaml`):
```yaml
name: transmod-dl
channels:
  - conda-forge
  - pytorch
dependencies:
  - python>=3.12
  - pip
  - pip:
    - -e ".[deep-learning]"
```

### Development Installation

```bash
git clone https://github.com/katarinagresova/transmod.git
cd transmod
pip install -e ".[dev,complete]"
```

### Dependency Groups Explained

| Group | Contains | Use Case |
|-------|----------|----------|
| `bio` | BioPython | Sequence analysis, FASTA/GenBank parsing |
| `ml` | scikit-learn, LightGBM | Classical machine learning |
| `dl` | PyTorch, Lightning | Deep learning models |
| `plots` | matplotlib, seaborn | Data visualization |
| `interp` | SHAP, Captum | Model interpretation |
| `notebook` | Jupyter, IPython | Interactive analysis |
| `dev` | pytest, ruff, mypy | Development tools |
| `docs` | Sphinx | Documentation building |

**Pre-configured combinations:**
- `classical-ml` = `bio + ml + plots`
- `deep-learning` = `bio + dl + plots`  
- `full-analysis` = `bio + ml + dl + plots + interp`
- `complete` = All components including notebooks

## Quick Start

```python
import transmod
from transmod.models import UNetTransformer
from transmod.data import TranslationDataset

# Load your data
dataset = TranslationDataset(
    sequences="path/to/sequences.fa",
    features="path/to/features.csv"
)

# Initialize model
model = UNetTransformer(
    input_dim=4,
    hidden_dim=256,
    num_heads=8
)

# Train model
from transmod.training import Trainer

trainer = Trainer(model, dataset)
trainer.fit(epochs=50)

# Make predictions
predictions = model.predict(dataset)
```

## Features

### Multi-Modal Data Integration
- Sequence features (one-hot, k-mers)
- RNA structure (DMS-seq, SHAPE)
- RNA modifications (m6A, pseudouridine)
- Protein embeddings (AlphaFold)
- Subcellular localization

### Model Architectures
- Hybrid U-Net-Transformer
- Multi-task learning
- Adaptive loss weighting
- Factor-agnostic embeddings

### Interpretability
- Gradient-based attribution
- In silico saturation mutagenesis
- SHAP values
- Attention visualization

## Documentation

Full documentation is available at [transmod.readthedocs.io](https://transmod.readthedocs.io)

## Citation

If you use Transmod in your research, please cite:

```bibtex
@article{transmod2025,
  title={Transmod: Multi-modal deep learning for translation initiation prediction},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Katarina Gresova
- **Email**: katarina.gresova@mdc-berlin.de
- **Issues**: [GitHub Issues](https://github.com/katarinagresova/transmod/issues)
```