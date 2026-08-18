# Laguerre-Gaussian Mode Classifier

A project for **generation and classification of Laguerre-Gaussian (LG) optical modes**.

The project is divided into two main parts:

1. **`lgbeam`** — a Python library for generating and manipulating Laguerre-Gaussian beams.
2. **Neural network classifier** — a deep learning model for identifying LG modes from their intensity distributions. This part of the project is currently **work in progress**.

> **Status:** 🚧 Work in progress

---

## Project overview

Laguerre-Gaussian modes form a family of orthogonal solutions to the paraxial wave equation. They are commonly described by two mode indices:

* `p` — radial mode index,
* `l` — azimuthal mode index.

The goal of this project is to provide a convenient computational pipeline for working with these modes:

```text
Laguerre-Gaussian mode
        │
        ▼
     lgbeam
        │
        ├── Generate LG mode
        ├── Propagate beam
        └── Create mode mixtures
        │
        ▼
   Intensity / field data
        │
        ▼
 Neural Network Classifier
        │
        ▼
 Predicted LG mode (p, l)
```

The generated optical fields can be used to create datasets for training and evaluating machine-learning models.

---

## Repository structure

```text
lg-mode-classifier/
│
├── src/
│   └── lgbeam/
│       └── ...                 # lgbeam library
│
├── examples/
│   └── lgbeam/
│       ├── LG00.py
│       ├── LG02.py
│       ├── LG10.py
│       ├── mixture.py
│       ├── propagation.py
│       └── vortex.py
│
├── docs/
│   └── lgbeam/                 # library documentation
│
├── tests/                      # tests
│
├── Dockerfile
├── pyproject.toml
├── LICENSE
└── README.md
```

---

# 1. `lgbeam`

`lgbeam` is a Python library for generating **Laguerre-Gaussian optical beams**.

The package is currently version `0.1.0` and requires **Python 3.10+**. Its core dependencies are NumPy, SciPy and Matplotlib.

### Installation

Clone the repository:

```bash
git clone https://github.com/mmarze/lg-mode-classifier.git
cd lg-mode-classifier
```

Install the package in editable mode:

```bash
pip install -e .
```

The package can then be imported as:

```python
import lgbeam
```

### Features

The library is being developed around several operations relevant to LG beams, including:

* generation of Laguerre-Gaussian modes,
* beam propagation,
* generation of mode mixtures,
* optical vortex generation,
* visualization of optical fields.

Example scripts for these operations are available in [`examples/lgbeam`](https://github.com/mmarze/lg-mode-classifier/tree/main/examples/lgbeam).

---

## Example

A simple example of generating an LG mode:

```python
import lgbeam

# Generate a Laguerre-Gaussian mode
# Example parameters: radial index p and azimuthal index l
beam = lgbeam.LG(p=0, l=1)
```

> **Note:** The API is still under development and may change between versions.

---

# 2. Neural Network Classifier

The second part of the project focuses on **automatic classification of Laguerre-Gaussian modes using neural networks**.

The general idea is to use simulated optical fields generated with `lgbeam` as training data for a neural network.

```text
        lgbeam
           │
           ▼
   Generate LG modes
           │
           ▼
  Generate training data
           │
           ▼
     Neural network
           │
           ▼
     Mode prediction
           │
           ▼
        (p, l)
```

The classifier is currently **in development**.

---

## Scientific motivation

Laguerre-Gaussian beams are important in several areas of modern optics because of their spatial structure and orbital angular momentum.

Reliable identification of LG modes can be useful in applications involving optical communications, optical metrology, mode multiplexing, beam characterization and computational imaging.

Machine-learning-based approaches have also been investigated for the analysis and classification of LG modes, making automated mode recognition an interesting application for this project.

---

# Development

This project is currently under active development.

The `lgbeam` library is the more mature component, while the neural-network classifier is being developed on top of it.

---

## Testing

Tests for the project are located in:

```text
tests/
```

Run the test suite with:

```bash
pytest
```

---

## Documentation

Documentation for the `lgbeam` package is located in:

```text
docs/lgbeam/
```

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Author

**Marcin Marzejon**

GitHub: [@mmarze](https://github.com/mmarze)

---

## Project status

🚧 **Work in progress**

The project is currently evolving, especially the neural-network component. APIs, model architectures, datasets and training procedures may change as development progresses.

## Tech Stack

**Python · NumPy · SciPy · Matplotlib · PyTorch · pytest · Docker · Git**
