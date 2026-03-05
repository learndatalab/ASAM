# ASAM: Adaptive Spatial Dependency Alignment for Domain Generalization in Multivariate Time-series Sensor Data

**ASAM** is a two-phase **domain generalization (DG)** framework for **multivariate time-series (MTS)** classification that explicitly addresses **sensor-induced spatial distribution shifts**, such as **sensor misalignment (reattachment)** and **sensor permutation (reordering)**.

> Paper: *Harnessing spatial dependency for domain generalization in multivariate time-series sensor data*  
> Authors: Jaehyun Bae, Heesoo Jung, Hogun Park
> Affiliation: Department of Computer Science and Engineering, Sungkyunkwan University


---

##  Overview

Multivariate time-series (MTS) data collected from multiple sensors often experience **distribution shifts across domains** due to factors such as:

- Sensor misalignment (sensor reattachment)
- Sensor permutation (channel reordering)
- Subject-specific variability

Existing domain generalization (DG) methods typically assume a **fixed spatial dependency structure**, which may not hold in real-world MTS sensor environments.

To address this problem, we propose **ASAM (Adaptive Spatial Dependency Alignment in MTS Data for Domain Generalization)**, a two-phase framework that adaptively aligns spatial dependencies across domains.

Key ideas include:

- Sensor positional embeddings to capture sensor roles
- Input-driven graph generation
- A GNN-based domain generalization layer
- Two-view regularization for improved robustness


---

## Key Contributions

- A **domain generalization framework** specifically designed for MTS classification
- A **GNN-based DG layer** that adaptively aligns spatial dependencies across domains
- **Sensor positional embeddings** to model sensor roles
- **Two-view regularization** for sensor independence and temporal consistency
- Extensive evaluation on **four real-world MTS datasets**

---

## Results Summary

ASAM is evaluated on **4 real-world datasets**:

- **Ninapro DB5** (sEMG)
- **SD-gesture** (sEMG, sparse)
- **HHAR** (HAR, smartphone/watch)
- **UCI-HAR** (HAR, smartphone sensors)

It outperforms **13 baselines** including:
- Feature-learning methods (PICCA, TCN, 2SRNN, SimpleAtt, STCN-GR, GNN-SD, InceptionTime)
- DG methods (GLIE, LAG, GSAT, CAL, GREA, DisC)

Additional robustness experiments show strong preservation under **sensor permutation**.

---

## Reproducibility

### Key Hyperparameters
| Hyperparameter | Ninapro | SD-gesture | HHAR | UCI-HAR |
|---|---:|---:|---:|---:|
| learning rate | 0.001 | 0.001 | 0.001 | 0.001 |
| batch size | 512 | 1024 | 512 | 512 |
| λ1 | 1e-2 | 1e-1 | 1e-3 | 1e-3 |
| λ2 | 1e-5 | 1e-4 | 1e-3 | 1e-3 |
| λ3 | 1e-5 | 1e-5 | 1e-5 | 1e-4 |
| σ (Gaussian) | 3.0 | 5.0 | 1.0 | 1.0 |
| F (LSTM dim) | 512 | 512 | 512 | 512 |

### Hardware
- NVIDIA RTX A6000 GPU (paper setting)

---

## Datasets

- **Ninapro DB5**: https://ninapro.hevs.ch/instructions/DB5.html  
- **HHAR**: https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition  
- **UCI-HAR**: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones  

> SD-gesture dataset citation is included in the paper (Lee et al., 2023).

---

## Environment Setting

The code has been tested with the following environment:

- Python 3.9.15

Required packages:

- pytorch==1.12.0  
- numpy==1.23.4  
- scikit-learn==1.2.1  
- scipy==1.9.3  
- pandas==1.5.3  

You can install the dependencies using:

```bash
pip install torch==1.12.0 numpy==1.23.4 scikit-learn==1.2.1 scipy==1.9.3 pandas==1.5.3
```


---

## How to Run the Code

Example: Running the model on the HHAR dataset
```bash
python train_model.py --dataset HAR_SA --channel_electrode 3 --feature_dim 128
```

## Citation

If you use ASAM in your research, please cite:

```bibtex
@article{asam_eswa,
  title     = {Harnessing spatial dependency for domain generalization in multivariate time-series sensor data},
  author    = {Bae, Jaehyun and Jung, Heesoo and Park, Hogun},
  journal   = {Expert Systems with Applications},
  year      = {2026}
}
