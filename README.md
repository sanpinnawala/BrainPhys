## Learning Mechanistic Subtypes of Neurodegeneration with a Physics-Informed Variational Autoencoder Mixture Model

> Sanduni Pinnawala, Annabelle Hartanto, Ivor J. A. Simpson, Peter A. Wijeratne

This repository provides code, configurations, and environment setup to reproduce experiments from the paper.

![Schematic of BrainPhys](mixture.jpg)

### Prerequisites

- All Python dependencies are listed in [`requirements.txt`](./requirements.txt)
- [Docker](https://www.docker.com/) is recommended for reproducible environment setup

### Setup

1. Clone the repository

```bash
git clone https://github.com/sanpinnawala/BrainPhys.git
cd BrainPhys
```

2. Build Docker image

```bash
docker build -t brainphys .
```

4. Run Docker container

```bash
docker run --gpus all -v /absolute/path/to/project:/app -it brainphys
```

### Running Experiments

1. Change to the model directory (e.g., [`reaction-diffusion-mixture-model`](./reaction-diffusion-mixture-model))

```bash
cd reaction-diffusion-mixture-model
```

2. Create `data`, `checkpoints`, and `artefacts` directories inside the model directory

```bash
mkdir data checkpoints artefacts
```

#### Train 

```bash
python src/train.py --config src/configs/config.yaml
```

#### Test

```bash
python src/test.py --config src/configs/config.yaml
```

### Citation

> TBD

### 📬 Contact

For questions or collaborations, feel free to contact [m.pinnawala@sussex.ac.uk](mailto:m.pinnawala@sussex.ac.uk)