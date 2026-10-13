dielec-ml-pub\
├── train.py
├── requirements.txt
├── README.md
├── data\
│   ├── built_in_dataset.xlsx
│   ├── external_qin.xlsx
│   ├── external_ye.xlsx
│   └── external_a_new.xlsx
└── results\

# DielectroNet Reproducibility Package

This repository contains the source code, datasets, preprocessing scripts, trained configurations, and evaluation scripts for the DielectroNet platform.

The code reproduces all experiments reported in the manuscript:

> DielectroNet: An Integrated Platform for Data-Driven Property Evaluation and Explainable Analysis of Dielectric Materials

---

## 1. Environment

- Python 3.10
- scikit-learn 1.7.2
- xgboost 2.x
- shap 0.4x
- pandas 2.x
- numpy 1.26
- matplotlib 3.x
- joblib
- openpyxl
- openai (optional, for DeepSeek API)

Install dependencies:

pip install -r requirements.txt

Optional: set the DeepSeek API key to enable LLM-assisted interpretation.

On Linux / macOS:

export DEEPSEEK_API_KEY="your_api_key"

On Windows CMD:

set DEEPSEEK_API_KEY=your_api_key

If the key is not set, the script will skip LLM interpretation and still produce all numerical results and figures.

---

## 2. Data

All datasets are stored in `data/`.

| File | Description |
|---|---|
| `data/built_in_dataset.xlsx` | Built-in dataset used in the main experiments |
| `data/external_qin.xlsx` | External validation dataset (Qin et al.) |
| `data/external_ye.xlsx` | External validation dataset (Ye et al.) |
| `data/external_a_new.xlsx` | Additional external dataset |

Data format:

- The last column is the target (dielectric constant).
- All other numeric columns are features.
- Non-numeric columns are ignored.

To run the code on a different dataset, change `FILE_PATH` in `train.py` to point to the new Excel file, or pass it via the command line.

---

## 3. Preprocessing

The preprocessing pipeline is defined inside `train.py`:

Pipeline([
    ('scaler', StandardScaler()),
    ('model', <regressor>)
])

- StandardScaler is applied to all features.
- No test data is used in preprocessing.
- No feature selection is applied in the final pipeline.

---

## 4. Data splitting

The main performance evaluation uses Nested Repeated 5-fold CV:

- Outer CV: RepeatedKFold(n_splits=5, n_repeats=5, random_state=42) -> 25 outer-fold evaluations.
- Inner CV: KFold(n_splits=3, shuffle=True, random_state=42) -> used only for hyperparameter selection inside each outer training fold.

The test folds are never used for hyperparameter tuning or preprocessing.

All random seeds are fixed to 42:

random.seed(42)
np.random.seed(42)

---

## 5. Models and hyperparameter search spaces

Six models are trained:

| Key | Model | Search space |
|---|---|---|
| `ridge` | Ridge Regression | `alpha`: [0.001, 0.01, 0.1, 1, 10, 100] |
| `rf` | Random Forest | `n_estimators`: [200]; `max_depth`: [None, 15]; `min_samples_leaf`: [1, 3]; `max_features='sqrt'` (fixed) |
| `svr` | Support Vector Regression | `C`: [0.1, 1, 10]; `epsilon`: [0.01, 0.1, 0.2]; `kernel`: ['rbf', 'linear'] |
| `gbrt` | Gradient Boosting | `n_estimators`: [200]; `learning_rate`: [0.1]; `max_depth`: [2, 3]; `min_samples_leaf`: [3, 5] |
| `xgb` | XGBoost | `n_estimators`: [300]; `max_depth`: [3, 5]; `reg_alpha`: [0, 0.2]; `reg_lambda`: [0.5, 1.5] |
| `knn` | K-Nearest Neighbors | `n_neighbors`: [3, 5, 7, 9]; `weights`: ['uniform', 'distance']; `p`: [1, 2] |

Hyperparameters are selected using GridSearchCV with the inner 3-fold CV. The final optimized parameters are reported in `results/<model>_best_params.txt`.

---

## 6. How to run

Run the main training and evaluation script:

python train.py --data data/built_in_dataset.xlsx --out results

The script will:

1. Load the dataset.
2. Run Nested Repeated 5-fold CV for all six models.
3. Save the 25 outer-fold results.
4. Save averaged OOF predictions and per-sample prediction SD.
5. Retrain each model on the full dataset and save the final pipelines.
6. Run SHAP, bubble matrix, and Pareto analysis on the best model.
7. Optionally call DeepSeek for LLM-assisted interpretation.

Outputs are saved to the directory specified by `--out`.

---

## 7. Output files

| File | Description |
|---|---|
| `repeated_5fold_cv_summary.csv` | Mean ± SD of R², RMSE, MAE across 25 outer folds |
| `repeated_5fold_cv_all_results.csv` | Per-fold R², RMSE, MAE for all 25 folds |
| `all_models_25_outer_fold_predictions.csv` | Raw outer-fold predictions for every sample |
| `<model>_averaged_oof_predictions.csv` | Five repeat-level OOF predictions, mean, and SD per sample |
| `all_models_averaged_oof_predictions.xlsx` | Six model-level averaged OOF tables + metrics |
| `all_models_averaged_oof_metrics.csv` | R²/RMSE/MAE from averaged OOF predictions |
| `<model>_averaged_oof_observed_vs_predicted.png/.svg` | Observed vs predicted plots |
| `<best_model>_pipeline.pkl` | Final trained pipeline for platform use |
| `<best_model>_features.pkl` | Feature column names |
| `<best_model>_best_params.txt` | Final optimized hyperparameters |
| `<best_model>_feature_importance_all.csv` | Full feature importance ranking |
| `<best_model>_shap_summary_top12.png/.svg` | SHAP summary plot |
| `<best_model>_corr_bubble_matrix_top12.png/.svg` | Correlation bubble matrix |
| `<best_model>_pareto_top12.png/.svg` | Pareto cumulative contribution plot |
| `<best_model>_meta.pkl` | Metadata for the platform |
| `<best_model>_deepseek_shap.txt` | DeepSeek interpretation of SHAP |
| `<best_model>_deepseek_bubble.txt` | DeepSeek interpretation of bubble matrix |
| `<best_model>_deepseek_pareto.txt` | DeepSeek interpretation of Pareto |

---

## 8. Reproducibility checklist

| Item | Location |
|---|---|
| Data | `data/` |
| Preprocessing | `train.py` |
| Data split | `train.py` (RepeatedKFold + KFold) |
| Random seeds | `train.py` (seed = 42) |
| Hyperparameter search space | `train.py` |
| Final optimized parameters | `results/<model>_best_params.txt` |
| Model performance | `results/repeated_5fold_cv_summary.csv` |
| OOF predictions | `results/all_models_averaged_oof_predictions.xlsx` |
| SHAP / bubble / Pareto | `results/` |

---

## 9. Citation

If you use this code, please cite:

> DielectroNet: An Integrated Platform for Data-Driven Property Evaluation and Explainable Analysis of Dielectric Materials.

---

## 10. Contact

For questions, please open an issue on GitHub or contact the corresponding author.
