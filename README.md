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

### From PyPI (once published)
```bash
pip install transmod
```

### From source (development)
```bash
git clone https://github.com/yourusername/transmod.git
cd transmod
pip install -e ".[dev]"
```

### With conda
```bash
conda create -n transmod python=3.12
conda activate transmod
pip install transmod
```

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