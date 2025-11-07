# DA5401 Assignment 8 — Ensemble Learning for Bike Sharing Demand Prediction

This repository contains my submission for **DA5401 Assignment 8**, where I explore **ensemble regression techniques** to predict hourly bike rental counts (`cnt`) from the **UCI Bike Sharing Dataset**. The assignment compares multiple regression models and ensemble methods, evaluating their performance using **Root Mean Squared Error (RMSE)**.

The notebook demonstrates:
- **Baseline models**: Decision Tree Regressor and Linear Regression
- **Bagging Regressor**: Variance reduction through bootstrap aggregating
- **Gradient Boosting Regressor**: Bias reduction through sequential error correction
- **K-Nearest Neighbors**: Model diversity for stacking
- **Stacking Regressor**: Meta-learning combining multiple base learners

---

## Repository Contents

- `ensemble_learning.ipynb` — Main notebook implementing preprocessing, model training, ensemble methods, and evaluation
- `bike+sharing+dataset/` — Dataset folder (contains `hour.csv`, `day.csv`, and `Readme.txt`)
- `requirements.txt` — Python dependencies
- `.gitignore` — Ignores checkpoints, venv, caches, and zipped datasets
- `README.md` — This file

---

## Author

- **Name**: Mayank Chandak
- **Roll No.**: ME22B224

---

## Problem Overview

The **UCI Bike Sharing Dataset** contains hourly bike rental data from Washington, D.C. during 2011-2012, including weather conditions, temporal features, and rental counts. The objective is to:

- Build and compare multiple regression models (Decision Tree, Linear Regression)
- Implement **Bagging** to reduce model variance
- Implement **Gradient Boosting** to reduce model bias
- Create a **Stacking Regressor** combining diverse base learners
- Evaluate all models using **RMSE** and identify the best-performing approach

---

## Setup & Usage

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook and run:
```bash
jupyter notebook
```
- Open `ensemble_learning.ipynb`
- Ensure the dataset folder `bike+sharing+dataset/` exists alongside the notebook
- Run cells sequentially

### Alternative: Using Conda
```bash
conda create -n da5401-a8 python=3.10 -y
conda activate da5401-a8
pip install -r requirements.txt
jupyter notebook
```

---

## File Structure

```
assignment-8-mnm-21/
├── ensemble_learning.ipynb          # Main analysis notebook
├── bike+sharing+dataset/             # Dataset folder
│   ├── hour.csv                      # Hourly bike sharing data
│   ├── day.csv                       # Daily bike sharing data
│   └── Readme.txt                    # Dataset documentation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignore rules (incl. zipped datasets)
└── README.md                         # This file
```

---

## Dependencies

Key packages (see `requirements.txt`):
- `numpy`, `pandas` — Data manipulation and analysis
- `scikit-learn` — Machine learning models, preprocessing, and evaluation
- `matplotlib` — Data visualization
- `jupyter`, `ipykernel` — Notebook environment

---

## Methodology

### Data Preprocessing
- **Feature Selection**: Remove redundant columns (`instant`, `dteday`, `casual`, `registered`)
- **Feature Engineering**: Separate categorical and numerical features
  - Categorical: `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`
  - Numerical: `temp`, `atemp`, `hum`, `windspeed`
- **Preprocessing Pipelines**:
  - **Tree-based models**: One-hot encoding for categoricals, passthrough for numericals
  - **Linear/KNN models**: One-hot encoding + StandardScaler for numericals

### Models Implemented

1. **Decision Tree Regressor** (Baseline)
   - `max_depth=6` to control overfitting
   - Provides interpretable baseline

2. **Linear Regression** (Baseline)
   - Simple linear model for comparison
   - Requires feature scaling

3. **Bagging Regressor**
   - Base estimator: Decision Tree (`max_depth=6`)
   - `n_estimators=100` (or 300)
   - Reduces variance through bootstrap aggregating

4. **Gradient Boosting Regressor**
   - `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`
   - Sequentially corrects errors to reduce bias

5. **K-Nearest Neighbors Regressor**
   - `n_neighbors=15`, `weights="distance"`
   - Provides model diversity for stacking

6. **Stacking Regressor**
   - Base learners: KNN, Bagging, Gradient Boosting
   - Meta-learner: Ridge Regression (`alpha=1.0`)
   - Combines predictions optimally

### Evaluation
- **Metric**: Root Mean Squared Error (RMSE)
- **Train/Test Split**: 80/20 with fixed random seed
- All preprocessing and modeling done via scikit-learn pipelines

---

## Results Summary (High-Level)

The **Stacking Regressor** achieved the lowest RMSE (~69.95), outperforming all individual models:

| Model | RMSE |
|:------|------:|
| Linear Regression (baseline) | ~102.14 |
| Decision Tree Regressor | ~118.47 |
| Bagging Regressor | ~112.39 |
| Gradient Boosting Regressor | ~72.90 |
| **Stacking Regressor** | **~69.95** |

> Note: Exact numbers depend on random seeds and environment. Re-run the notebook to reproduce.

---

## Key Insights

1. **Ensemble Methods Outperform Baselines**: Both Bagging and Boosting significantly improve upon single Decision Tree performance.

2. **Bias vs Variance Trade-off**: 
   - **Bagging** reduces variance (improvement from 118.47 to 112.39)
   - **Boosting** reduces bias (major improvement to 72.90)

3. **Stacking Leverages Diversity**: Combining diverse learners (KNN, Bagging, GBR) with a meta-learner achieves the best performance by learning optimal combination weights.

4. **Model Diversity Matters**: Including KNN (instance-based) alongside tree-based models improves stacking performance.

---

## Troubleshooting

1. **Import Errors**
   - Ensure dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (≥3.8)

2. **Dataset Path Issues**
   - Ensure `bike+sharing+dataset/hour.csv` exists relative to the notebook
   - Update `DATA_PATH` if dataset is located elsewhere

3. **Memory/Performance**
   - Reduce `n_estimators` in Bagging/Boosting if memory is constrained
   - Set `n_jobs=-1` for parallel processing (already included)

4. **Convergence Warnings**
   - Increase `max_iter` for Linear Regression if needed
   - Ensure proper preprocessing (scaling) for linear models

---

## Dataset Information

- **Source**: UCI Bike Sharing Dataset
- **Target**: `cnt` (total hourly bike rentals)
- **Features**: 
  - Temporal: season, year, month, hour, weekday, workingday, holiday
  - Weather: temperature, feels-like temperature, humidity, windspeed, weather situation
- **Objective**: Predict hourly bike rental demand using ensemble regression methods

---

## References

- Scikit-learn documentation: ensemble methods, pipelines, preprocessing, metrics
- UCI Machine Learning Repository: Bike Sharing Dataset
- Breiman, L. (1996): Bagging Predictors
- Friedman, J. H. (2001): Greedy Function Approximation: A Gradient Boosting Machine
- Wolpert, D. H. (1992): Stacked Generalization

---

## Reproducibility

- Fixed random seed (`RANDOM_STATE = 224`) ensures consistent results
- Run all cells in order; results will be displayed at the end
- Ensure dataset is in the correct location before running
