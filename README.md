# INRBO-optimized-CNN-BiLSTM-Attention-model
Data-driven airflow velocity prediction for air-jet loom main nozzles
# Predictive Optimization Framework for Air-Jet Loom Nozzle Performance

This repository provides the implementation of a data-driven predictive optimization framework for airflow performance analysis and parameter optimization of air-jet loom nozzles.

The framework integrates deep learning models, hybrid prediction methods, intelligent optimization algorithms, and SHAP-based interpretability analysis to achieve accurate prediction and optimized design of nozzle aerodynamic performance.

---

## Repository Structure

```
Predictive Optimization
│
├── Hybrid model prediction comparison
│   ├── CNN-BiLSTM.py
│   ├── CNN-BiLSTM-Attention-Pytorch.py
│   ├── CNN-GRU.py
│   ├── CNN-GRU-Attention.py
│   ├── CNN-LSTM-Attention.py
│   └── CNN-LSTM-Pytorch.py
│
├── Parameter optimization
│   ├── INRBO.py
│   ├── NRBO.py
│   ├── HEOA.py
│   ├── SCNGO.py
│   └── X-CNN-BiLSTM-Attention.py
│
├── SHAP Analysis Result Figures
│
├── SHAP Analysis.py
│
├── Data.xlsx
│
├── SHAP Analysis Data.xlsx
│
└── Output image generation programs
```

---

# 1. Overview

Air-jet loom nozzles are critical components in high-speed weft insertion systems. Their airflow characteristics directly affect weaving efficiency, energy consumption, and fabric quality.

This project develops a predictive optimization framework based on artificial intelligence methods, including:

- Deep learning-based airflow prediction
- Hybrid model performance comparison
- Intelligent parameter optimization
- SHAP-based model interpretation
- Scientific visualization generation

The proposed framework establishes the nonlinear relationship between nozzle structural parameters and aerodynamic performance.

---

# 2. Main Functions

## 2.1 Hybrid Model Prediction Comparison

Folder:

```
Hybrid model prediction comparison
```

This folder contains different deep learning prediction models:

- CNN-LSTM
- CNN-BiLSTM
- CNN-GRU
- CNN-LSTM-Attention
- CNN-GRU-Attention
- CNN-BiLSTM-Attention

The main functions include:

- Model training
- Performance evaluation
- Prediction comparison
- Result visualization

---

## 2.2 Parameter Optimization

Folder:

```
Parameter optimization
```

This folder contains intelligent optimization algorithms:

- NRBO (Newton-Raphson-Based Optimization)
- INRBO (Improved Newton-Raphson-Based Optimization)
- HEOA (Hunger Games Search Optimization Algorithm)
- SCNGO (Supply Chain Network Game Optimization)

The optimization algorithms are applied to search optimal nozzle structural parameters based on performance objectives.

Optimization objectives include:

- Maximizing airflow velocity
- Reducing air consumption
- Multi-objective optimization

---

## 2.3 SHAP Interpretability Analysis

Files:

```
SHAP Analysis.py

SHAP Analysis Data.xlsx

SHAP Analysis Result Figures
```

SHAP (SHapley Additive exPlanations) is used to interpret the prediction models and quantify the contribution of input parameters.

The analysis includes:

- Feature importance ranking
- SHAP summary plots
- Feature contribution analysis
- Parameter influence interpretation

---

# 3. Dataset Description

## Data.xlsx

The dataset contains nozzle structural parameters and aerodynamic performance data.

Input variables:

- Nozzle geometric parameters
- Structural design parameters

Output variables:

- Airflow velocity
- Air consumption

---

## SHAP Analysis Data.xlsx

This file contains:

- SHAP values
- Feature contribution data
- Model interpretation results

---

# 4. Environment Requirements

Recommended environment:

```
Python >= 3.10
```

Main dependencies:

```
numpy
pandas
scikit-learn
matplotlib
scipy
pytorch
tensorflow
catboost
shap
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 5. Usage

## Step 1: Data Preparation

Prepare the dataset:

```
Data.xlsx
```

---

## Step 2: Model Training

Run prediction models:

```bash
python CNN-BiLSTM-Attention-Pytorch.py
```

Other models can be executed for performance comparison.

---

## Step 3: Parameter Optimization

Run optimization algorithms:

```bash
python INRBO.py
```

The program searches optimal nozzle parameters according to predefined optimization objectives.

---

## Step 4: SHAP Analysis

Run:

```bash
python SHAP Analysis.py
```

The generated SHAP visualization results are saved in:

```
SHAP Analysis Result Figures
```

---

# 6. Visualization Results

The repository provides programs for generating:

- Prediction comparison figures
- Optimization convergence curves
- SHAP analysis figures
- Statistical analysis plots
- Publication-quality images

---

# 7. Citation

If you use this repository in academic research, please cite the corresponding paper:

```
Data-driven prediction and multi-objective optimization of air-jet loom nozzle aerodynamic performance.
```

---

# 8. License

This repository is provided for academic research purposes only.

---

# 9. Contact

For questions regarding this repository, please contact the authors.
