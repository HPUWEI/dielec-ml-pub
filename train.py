import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RepeatedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance

import shap
import json
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import warnings
import random
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

client = None
if DEEPSEEK_API_KEY and OpenAI is not None:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 300,
    'savefig.dpi': 600,
})

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "data" / "ye_train_data.xlsx"
SAVE_FOLDER = BASE_DIR / "results" / "built_in_dataset"

FEATURE_DESCRIPTION_QIN = """
Feature abbreviations used in this dataset (7 features for microwave dielectric ceramics):
bg      Band gap (eV): energy gap between valence and conduction bands; governs electronic polarization contribution to dielectric response; wider gap generally means lower dielectric loss
d       Theoretical density (g/cm³): mass per unit volume; reflects atomic packing and compositional heaviness; denser materials often have higher polarizability per volume
fepa    Formation energy per atom (eV/atom): thermodynamic stability indicator; more negative values indicate more stable crystal structures
blm     Average bond length (Å): mean distance between bonded atoms; longer bonds are generally softer and more polarizable, contributing to higher dielectric constant
ppv     Polarizability per unit volume (Å²/Å³): intrinsic ability of the material to polarize per unit volume; the most direct physical driver of dielectric constant
bpv     Number of bonds per unit cell volume (Å⁻³): bond density; higher values indicate more structurally rigid and cross-linked frameworks, affecting both permittivity and mechanical stability
va      Average cell volume per atom (Å³/atom): space each atom occupies on average; inversely related to packing fraction; larger values indicate more open structures with potentially higher ionic polarizability
"""

FEATURE_DESCRIPTION_DATABASE = """
Feature abbreviations used in this dataset:
blm     Average bond length
pm      Molecular dielectric polarizability
blsd    Standard deviation of bond length
pa      Polarizability per atom
sm      Average bond strength
pv      Polarizability per unit volume
ssd     Standard deviation of bond strength
lsd     Standard deviation of lattice constant edge length
com     Average covalency
asd     Standard deviation of lattice constant angle
cosd    Standard deviation of covalency
vm      Unit cell volume
Em      Average bond energy
m       Relative molecular mass
Ev      Average bond energy per unit volume
d       Theoretical density
Esd     Standard deviation of bond energy
va      Unit cell volume per atom
bvsdm   Average of standard deviations of bond valence
Gll     Unit cell volume per molecule
devvm   Average deviation of bond valence from nominal valence
vi      Total ionic volume
pd      Packing fraction
via     Ionic volume per atom
"""

FEATURE_DESCRIPTION_YE = """
emb     B-site electronegativity–Matyonov-Batsanov
arb     Atomic radius of B-site element
eab     B-site electronegativity–absolute
epa     A-site electronegativity–Pauling
ieb     Ionization energies of B-site element
"""

def select_tick_step(file_path):
    name = os.path.basename(file_path).lower()

    if "qin" in name:
        return 4
    if "a_new" in name or "database_new_7features" in name or "built_in_dataset" in name:
        return 50
    if "train_data" in name or "ye" in name:
        return 25

    return 25

def select_feature_description(file_path):
    name = os.path.basename(file_path).lower()

    if "qin" in name:
        return FEATURE_DESCRIPTION_QIN
    if "a_new" in name or "database_new_7features" in name or "built_in_dataset" in name:
        return FEATURE_DESCRIPTION_DATABASE
    if "train_data" in name or "ye" in name:
        return FEATURE_DESCRIPTION_YE

    return ""

FEATURE_DESCRIPTION = select_feature_description(FILE_PATH)

os.makedirs(SAVE_FOLDER, exist_ok=True)

def compute_meta(
    key,
    best_pipeline,
    X,
    y,
    feature_cols,
    cv_summary_row,
    imps_all=None,
    shap_values_for_meta=None,
    shap_feature_names=None
):
    model = best_pipeline.named_steps['model']

    y_full_pred = best_pipeline.predict(X)

    used_features = list(feature_cols)

    meta = {
        'model_name': {
            'ridge': 'Ridge Regression',
            'rf': 'Random Forest',
            'svr': 'Support Vector Regression',
            'gbrt': 'Gradient Boosting',
            'xgb': 'XGBoost',
            'knn': 'K-Nearest Neighbors',
        }.get(key, key),

        'feature_names': used_features,

        'cv_r2_mean': float(cv_summary_row['R2_mean']),
        'cv_r2_sd': float(cv_summary_row['R2_SD']),
        'cv_rmse_mean': float(cv_summary_row['RMSE_mean']),
        'cv_rmse_sd': float(cv_summary_row['RMSE_SD']),
        'cv_mae_mean': float(cv_summary_row['MAE_mean']),
        'cv_mae_sd': float(cv_summary_row['MAE_SD']),

        'r2_full': float(r2_score(y, y_full_pred)),
        'rmse_full': float(np.sqrt(mean_squared_error(y, y_full_pred))),
        'mae_full': float(mean_absolute_error(y, y_full_pred)),

        'best_params': cv_summary_row.get('Final_Best_Params', {}),
        'importances': {},
        'shap_mean': {},
        'top_features_80pct': [],
        'corr_pairs': [],
    }

    if imps_all is not None and len(imps_all) == len(used_features):
        pairs = sorted(
            zip(used_features, [float(v) for v in imps_all]),
            key=lambda x: x[1],
            reverse=True
        )
        meta['importances'] = {n: round(v, 6) for n, v in pairs}

        total = sum(max(v, 0) for _, v in pairs)
        cumsum_val = 0
        top80 = []

        for n, v in pairs:
            cumsum_val += max(v, 0)
            top80.append(n)
            if total > 0 and cumsum_val / total >= 0.8:
                break

        meta['top_features_80pct'] = top80

    elif hasattr(model, 'feature_importances_'):
        raw_imp = model.feature_importances_
        pairs = sorted(
            zip(used_features, raw_imp.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        meta['importances'] = {n: round(v, 6) for n, v in pairs}

        total = sum(max(v, 0) for _, v in pairs)
        cumsum_val = 0
        top80 = []

        for n, v in pairs:
            cumsum_val += max(v, 0)
            top80.append(n)
            if total > 0 and cumsum_val / total >= 0.8:
                break

        meta['top_features_80pct'] = top80

    if shap_values_for_meta is not None and shap_feature_names is not None:
        sv = np.asarray(shap_values_for_meta)

        if sv.ndim == 2:
            mean_abs = np.abs(sv).mean(axis=0)

            meta['shap_mean'] = {
                n: round(float(v), 6)
                for n, v in sorted(
                    zip(shap_feature_names, mean_abs),
                    key=lambda x: x[1],
                    reverse=True
                )[:12]
            }

    try:
        corr = X.corr()
        cols_list = corr.columns.tolist()
        cp = []

        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                v = corr.iloc[i, j]
                if abs(v) >= 0.5:
                    cp.append({
                        'f1': cols_list[i],
                        'f2': cols_list[j],
                        'corr': round(float(v), 3)
                    })

        meta['corr_pairs'] = sorted(
            cp,
            key=lambda x: abs(x['corr']),
            reverse=True
        )[:15]

    except Exception:
        pass

    return meta

def call_deepseek(meta, plot_type, feature_desc=""):
    if client is None:
        return "[DeepSeek 未设置 API key，未执行 AI 解读。]"

    model_name = meta.get('model_name', '')
    feat_names = meta.get('feature_names', [])
    importances = meta.get('importances', {})
    shap_mean = meta.get('shap_mean', {})
    top80 = meta.get('top_features_80pct', [])
    corr_pairs = meta.get('corr_pairs', [])

    cv_r2_mean = meta.get('cv_r2_mean')
    cv_r2_sd = meta.get('cv_r2_sd')
    cv_rmse_mean = meta.get('cv_rmse_mean')
    cv_rmse_sd = meta.get('cv_rmse_sd')
    cv_mae_mean = meta.get('cv_mae_mean')
    cv_mae_sd = meta.get('cv_mae_sd')

    score_str = ""
    if cv_r2_mean is not None:
        score_str = (
            f"Nested repeated 5-fold CV performance: "
            f"R²={cv_r2_mean:.4f} ± {cv_r2_sd:.4f}, "
            f"RMSE={cv_rmse_mean:.4f} ± {cv_rmse_sd:.4f}, "
            f"MAE={cv_mae_mean:.4f} ± {cv_mae_sd:.4f}"
        )

    ctx = f"""You are a materials informatics expert.
A machine learning model was trained to predict dielectric properties of materials.

Model: {model_name}
{score_str}
All input features: {feat_names}
"""

    if feature_desc.strip():
        ctx += f"\nFeature meanings provided by the researcher:\n{feature_desc.strip()}\n"

    if plot_type == 'shap':
        ctx += f"""
Feature importances (sorted): {json.dumps(importances, indent=2)}
SHAP mean |value| ranking: {json.dumps(shap_mean, indent=2)}
Features capturing 80% importance: {top80}
"""

        instruction = """
Write a structured interpretation with exactly these three labeled points,
each 1-2 sentences:

[Key Features] Name the top 2-3 features by their actual names and SHAP values.
State what physical property each may represent in dielectric materials.

[Model Insight] Compare SHAP ranking vs raw importance ranking.
If they differ, state specifically which features differ and what this may indicate
about model dependence or feature interactions.

[Design Guidance] Give one cautious, model-based design hypothesis based on the
observed feature relationships. Do NOT claim that changing a feature will
causally improve the dielectric property. Explicitly distinguish model association
from physical causality.

Use the actual feature names and numbers provided.
Do not explain what SHAP is.
"""

    elif plot_type == 'bubble':
        strong_pos = [
            (p['f1'], p['f2'], p['corr'])
            for p in corr_pairs if p['corr'] > 0
        ][:5]

        strong_neg = [
            (p['f1'], p['f2'], p['corr'])
            for p in corr_pairs if p['corr'] < 0
        ][:5]

        ctx += f"""
Strong positive correlations (|r|≥0.5): {strong_pos}
Strong negative correlations (|r|≥0.5): {strong_neg}
"""

        instruction = """
Write a structured interpretation with exactly these three labeled points,
each 1-2 sentences:

[Strong Correlations] Name the top 2 most strongly correlated feature pairs
with their actual correlation values. State a plausible materials-science reason
why they may co-vary.

[Redundancy] Identify features that carry overlapping statistical information.
Do not claim that one feature can definitely be removed solely from correlation.

[Design Implication] Based on the correlation structure, describe which
combination of features may be informative for predicting the dielectric property.

Use the actual feature names and correlation values provided.
Do not explain what a correlation matrix is.
"""

    elif plot_type == 'pareto':
        ctx += f"""
Feature importances (Pareto sorted): {json.dumps(importances, indent=2)}
Features capturing 80% importance: {top80} ({len(top80)} out of {len(feat_names)} total)
"""

        instruction = """
Write a structured interpretation with exactly these three labeled points,
each 1-2 sentences:

[Dominance] State how many features account for 80% of predictive importance
and name them specifically with their importance values. Comment on whether
the importance is concentrated or distributed.

[Physical Meaning] For the top 2-3 features by name, explain why they may be
associated with dielectric-property prediction.

[Simplification] State which features in the long tail are candidates for
further investigation as lower-priority variables. Do not claim that removing
them will necessarily preserve predictive performance unless supported by
a separate retraining experiment.

Use the actual feature names and importance values provided.
Do not explain what a Pareto chart is.
"""

    else:
        instruction = "Briefly interpret this machine learning feature analysis chart."

    prompt = ctx + "\n" + instruction

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in materials informatics. "
                        "Be specific and data-driven. Always reference actual "
                        "feature names from the data provided."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=900
        )

        result = response.choices[0].message.content

    except Exception as e:
        result = f"[DeepSeek 调用失败: {e}]"

    return result

def make_bold(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    if ax.get_legend():
        for text in ax.get_legend().get_texts():
            text.set_fontweight('bold')

def process_excel_local(file_path):

    global FEATURE_DESCRIPTION
    FEATURE_DESCRIPTION = select_feature_description(file_path)

    df = pd.read_excel(file_path)
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] < 2:
        print("错误: Excel 至少需要两列数值")
        return

    feature_cols = list(df.columns[:-1])
    target_col = df.columns[-1]

    X = df[feature_cols].copy()
    y = df[target_col].values

    models = {

        'ridge': (
            Ridge(random_state=42),
            {
                'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            }
        ),

        'rf': (
            RandomForestRegressor(
                random_state=42,
                max_features='sqrt'
            ),
            {
                'model__n_estimators': [200],
                'model__max_depth': [None, 15],
                'model__min_samples_leaf': [1, 3]
            }
        ),

        'svr': (
            SVR(),
            {
                'model__C': [0.1, 1, 10],
                'model__epsilon': [0.01, 0.1, 0.2],
                'model__kernel': ['rbf', 'linear']
            }
        ),

        'gbrt': (
            GradientBoostingRegressor(random_state=42),
            {
                'model__n_estimators': [200],
                'model__learning_rate': [0.1],
                'model__max_depth': [2, 3],
                'model__min_samples_leaf': [3, 5]
            }
        ),

        'xgb': (
            XGBRegressor(
                random_state=42,
                eval_metric='rmse'
            ),
            {
                'model__n_estimators': [300],
                'model__max_depth': [3, 5],
                'model__reg_alpha': [0, 0.2],
                'model__reg_lambda': [0.5, 1.5]
            }
        ),

        'knn': (
            KNeighborsRegressor(),
            {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2]
            }
        ),
    }

    outer_cv = RepeatedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42
    )

    inner_cv = KFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    all_cv_results = []
    all_oof_prediction_rows = []
    cv_summary = []

    final_models = {}

    oof_predictions = {}

    scores = {}

    for key, (model, grid_params) in models.items():

        fold_results = []
        model_oof = []
        model_oof_matrix = np.full((len(X), 5), np.nan, dtype=float)

        for fold_id, (train_idx, val_idx) in enumerate(
            outer_cv.split(X),
            start=1
        ):

            repeat_id = (fold_id - 1) // 5 + 1
            fold_in_repeat = (fold_id - 1) % 5 + 1

            X_train_cv = X.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]

            y_train_cv = y[train_idx]
            y_val_cv = y[val_idx]

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            grid = GridSearchCV(
                estimator=pipe,
                param_grid=grid_params,
                cv=inner_cv,
                scoring='r2',
                n_jobs=-1,
                refit=True
            )

            grid.fit(X_train_cv, y_train_cv)

            best_model_cv = grid.best_estimator_

            y_val_pred = best_model_cv.predict(X_val_cv)

            r2_val = r2_score(y_val_cv, y_val_pred)
            rmse_val = np.sqrt(
                mean_squared_error(y_val_cv, y_val_pred)
            )
            mae_val = mean_absolute_error(
                y_val_cv,
                y_val_pred
            )

            result_row = {
                'Model': key,
                'Repeat': repeat_id,
                'Fold': fold_in_repeat,
                'Outer_Fold_ID': fold_id,
                'R2': r2_val,
                'RMSE': rmse_val,
                'MAE': mae_val,
            }

            fold_results.append(result_row)
            all_cv_results.append(result_row)

            model_oof.append({
                'indices': np.asarray(val_idx),
                'y_true': np.asarray(y_val_cv),
                'y_pred': np.asarray(y_val_pred),
                'repeat': repeat_id,
                'fold': fold_in_repeat
            })

            model_oof_matrix[val_idx, repeat_id - 1] = y_val_pred

            for sample_idx, observed, predicted in zip(
                val_idx, y_val_cv, y_val_pred
            ):
                all_oof_prediction_rows.append({
                    'Model': key,
                    'Sample_Index': int(sample_idx),
                    'Repeat': repeat_id,
                    'Fold': fold_in_repeat,
                    'Outer_Fold_ID': fold_id,
                    'Observed': float(observed),
                    'OOF_Predicted': float(predicted)
                })

        fold_df = pd.DataFrame(fold_results)

        r2_mean = fold_df['R2'].mean()
        r2_sd = fold_df['R2'].std(ddof=1)

        rmse_mean = fold_df['RMSE'].mean()
        rmse_sd = fold_df['RMSE'].std(ddof=1)

        mae_mean = fold_df['MAE'].mean()
        mae_sd = fold_df['MAE'].std(ddof=1)

        final_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        final_grid = GridSearchCV(
            estimator=final_pipe,
            param_grid=grid_params,
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1,
            refit=True
        )

        final_grid.fit(X, y)

        final_model = final_grid.best_estimator_
        final_models[key] = final_model

        final_best_params = final_grid.best_params_

        joblib.dump(
            final_model,
            os.path.join(
                SAVE_FOLDER,
                f'{key}_pipeline.pkl'
            )
        )

        joblib.dump(
            feature_cols,
            os.path.join(
                SAVE_FOLDER,
                f'{key}_features.pkl'
            )
        )

        with open(
            os.path.join(SAVE_FOLDER, f'{key}_best_params.txt'),
            'w',
            encoding='utf-8'
        ) as f:
            f.write(str(final_best_params))

        cv_summary.append({
            'Model': key,
            'R2_mean': r2_mean,
            'R2_SD': r2_sd,
            'RMSE_mean': rmse_mean,
            'RMSE_SD': rmse_sd,
            'MAE_mean': mae_mean,
            'MAE_SD': mae_sd,
            'Final_Best_Params': str(final_best_params)
        })

        scores[key] = r2_mean
        oof_predictions[key] = {
            'folds': model_oof,
            'matrix': model_oof_matrix
        }

    all_cv_df = pd.DataFrame(all_cv_results)
    summary_df = pd.DataFrame(cv_summary)

    all_cv_csv = os.path.join(
        SAVE_FOLDER,
        "repeated_5fold_cv_all_results.csv"
    )

    summary_csv = os.path.join(
        SAVE_FOLDER,
        "repeated_5fold_cv_summary.csv"
    )

    all_cv_df.to_csv(
        all_cv_csv,
        index=False,
        encoding='utf-8-sig'
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding='utf-8-sig'
    )

    excel_path = os.path.join(
        SAVE_FOLDER,
        "repeated_5fold_cv_results.xlsx"
    )

    with pd.ExcelWriter(excel_path) as writer:
        all_cv_df.to_excel(
            writer,
            sheet_name="All_25_Folds",
            index=False
        )
        summary_df.to_excel(
            writer,
            sheet_name="Mean_SD",
            index=False
        )

    best_key = max(scores, key=scores.get)
    best_pipeline = final_models[best_key]

    best_summary_row = summary_df[
        summary_df['Model'] == best_key
    ].iloc[0].to_dict()

    all_oof_prediction_df = pd.DataFrame(all_oof_prediction_rows)
    all_oof_prediction_df.to_csv(
        os.path.join(
            SAVE_FOLDER,
            "all_models_25_outer_fold_predictions.csv"
        ),
        index=False,
        encoding='utf-8-sig'
    )

    averaged_oof_tables = {}
    oof_metrics_rows = []

    from matplotlib.ticker import MultipleLocator

    model_display_names = {
        'ridge': 'RR',
        'rf': 'RF',
        'svr': 'SVR',
        'gbrt': 'GBRT',
        'xgb': 'XGBoost',
        'knn': 'KNN',
    }

    for key in models.keys():
        pred_matrix = oof_predictions[key]['matrix']

        if np.isnan(pred_matrix).any():
            raise RuntimeError(
                f"{key} 的 OOF prediction 不完整：并非每个样品都有 5 次重复预测。"
            )

        pred_mean = np.mean(pred_matrix, axis=1)
        pred_sd = np.std(pred_matrix, axis=1, ddof=1)

        model_oof_r2 = r2_score(y, pred_mean)
        model_oof_rmse = np.sqrt(mean_squared_error(y, pred_mean))
        model_oof_mae = mean_absolute_error(y, pred_mean)

        oof_table = pd.DataFrame({
            'Sample_Index': np.arange(len(X)),
            'Observed': np.asarray(y, dtype=float),
            'OOF_Predicted_Repeat_1': pred_matrix[:, 0],
            'OOF_Predicted_Repeat_2': pred_matrix[:, 1],
            'OOF_Predicted_Repeat_3': pred_matrix[:, 2],
            'OOF_Predicted_Repeat_4': pred_matrix[:, 3],
            'OOF_Predicted_Repeat_5': pred_matrix[:, 4],
            'OOF_Predicted_Mean': pred_mean,
            'OOF_Prediction_SD': pred_sd,
        })

        averaged_oof_tables[key] = oof_table

        oof_table.to_csv(
            os.path.join(
                SAVE_FOLDER,
                f"{key}_averaged_oof_predictions.csv"
            ),
            index=False,
            encoding='utf-8-sig'
        )

        oof_metrics_rows.append({
            'Model': key,
            'OOF_R2_from_Averaged_Predictions': model_oof_r2,
            'OOF_RMSE_from_Averaged_Predictions': model_oof_rmse,
            'OOF_MAE_from_Averaged_Predictions': model_oof_mae,
            'Mean_Prediction_SD_Across_Samples': float(np.mean(pred_sd)),
            'Median_Prediction_SD_Across_Samples': float(np.median(pred_sd)),
        })

        fig, ax = plt.subplots(figsize=(6.5, 4.0))

        residual = np.abs(pred_mean - y)
        res_norm = residual / residual.max() if residual.max() > 0 else residual

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "yellow_purple", ["#FFD400", "#7A00CC"]
        )

        sc = ax.scatter(
            y,
            pred_mean,
            c=1 - res_norm,                   
            cmap=cmap,
            s=65,
            alpha=0.88,
            edgecolor='black',
            linewidth=0.45,
            vmin=0, vmax=1
        )

        if key == 'knn':
            cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
                                                                                      
            cbar.ax.tick_params(labelsize=12)
            for label in cbar.ax.get_yticklabels():
                label.set_fontweight('normal')

        all_vals = np.concatenate([y, pred_mean])
        min_val = float(np.min(all_vals))
        max_val = float(np.max(all_vals))
        padding = 0.05 * (max_val - min_val)
        if padding == 0:
            padding = 1.0

        lims = [min_val - padding, max_val + padding]

        ax.plot(
            lims,
            lims,
            linestyle='--',
            color='black',
            linewidth=1.8
        )

        ax.set_xlim(lims)
        ax.set_ylim(lims)

        tick_step = select_tick_step(FILE_PATH)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))

        ax.set_xlabel(
            'Experimental εᵣ',
            fontsize=20,
            fontweight='normal'
        )
        ax.set_ylabel(
            'Predicted εᵣ',
            fontsize=20,
            fontweight='normal'
        )

        ax.text(
            0.04,
            0.96,
            model_display_names[key],
            transform=ax.transAxes,
            fontsize=18,
            ha='left',
            va='top'
        )

        metric_text = '\n'.join((
            f'R² = {model_oof_r2:.3f}',
            f'RMSE = {model_oof_rmse:.3f}',
            f'MAE = {model_oof_mae:.3f}',
        ))

        ax.text(
            0.96,
            0.05,
            metric_text,
            transform=ax.transAxes,
            fontsize=18,
            ha='right',
            va='bottom',
            linespacing=1.45
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

        ax.tick_params(width=1.5, labelsize=20)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('normal')

        plt.subplots_adjust(
            left=0.17,
            right=0.87,
            top=0.95,
            bottom=0.17
        )

        fig.savefig(
            os.path.join(
                SAVE_FOLDER,
                f"{key}_averaged_oof_observed_vs_predicted.png"
            ),
            dpi=300
        )
        fig.savefig(
            os.path.join(
                SAVE_FOLDER,
                f"{key}_averaged_oof_observed_vs_predicted.svg"
            ),
            format='svg'
        )
        plt.close(fig)

    oof_metrics_df = pd.DataFrame(oof_metrics_rows)
    oof_metrics_df.to_csv(
        os.path.join(
            SAVE_FOLDER,
            "all_models_averaged_oof_metrics.csv"
        ),
        index=False,
        encoding='utf-8-sig'
    )

    averaged_oof_excel = os.path.join(
        SAVE_FOLDER,
        "all_models_averaged_oof_predictions.xlsx"
    )

    with pd.ExcelWriter(averaged_oof_excel) as writer:
        for key, table in averaged_oof_tables.items():
            table.to_excel(
                writer,
                sheet_name=key.upper(),
                index=False
            )
        oof_metrics_df.to_excel(
            writer,
            sheet_name='OOF_Metrics',
            index=False
        )

    best_key = max(scores, key=scores.get)
    best_pipeline = final_models[best_key]

    best_summary_row = summary_df[
        summary_df['Model'] == best_key
    ].iloc[0].to_dict()

    best_oof_table = averaged_oof_tables[best_key]
    best_oof_r2 = float(
        oof_metrics_df.loc[
            oof_metrics_df['Model'] == best_key,
            'OOF_R2_from_Averaged_Predictions'
        ].iloc[0]
    )
    best_oof_rmse = float(
        oof_metrics_df.loc[
            oof_metrics_df['Model'] == best_key,
            'OOF_RMSE_from_Averaged_Predictions'
        ].iloc[0]
    )
    best_oof_mae = float(
        oof_metrics_df.loc[
            oof_metrics_df['Model'] == best_key,
            'OOF_MAE_from_Averaged_Predictions'
        ].iloc[0]
    )

    scaler = best_pipeline.named_steps['scaler']
    model = best_pipeline.named_steps['model']

    all_features = feature_cols

    X_all_trans = scaler.transform(X)

    n_features = X_all_trans.shape[1]

    if n_features != len(all_features):
        used_features = [
            f"feature_{i}"
            for i in range(n_features)
        ]

    else:
        used_features = all_features

    imps = None

    if hasattr(model, "feature_importances_"):
        try:
            imps = model.feature_importances_

            if len(imps) != n_features:
                imps = None

        except Exception:
            imps = None

    if imps is None:
        try:
            sample_size = min(
                500,
                X_all_trans.shape[0]
            )

            sample_idx = np.random.choice(
                X_all_trans.shape[0],
                sample_size,
                replace=False
            )

            result = permutation_importance(
                model,
                X_all_trans[sample_idx],
                y[sample_idx],
                n_repeats=5,
                random_state=42,
                n_jobs=1
            )

            imps = result.importances_mean

        except Exception as e:

            imps = X_all_trans.std(axis=0)

    imps = np.asarray(imps, dtype=float)

    n_top = min(
        12,
        n_features
    )

    top_indices = np.argsort(
        imps
    )[::-1][:n_top]

    top_features = [
        used_features[i]
        for i in top_indices
    ]

    top_imps = imps[top_indices]

    shap_values_top = None

    if shap is not None:

        try:

            bg_idx = np.random.choice(
                X_all_trans.shape[0],
                min(
                    200,
                    X_all_trans.shape[0]
                ),
                replace=False
            )

            X_background = X_all_trans[bg_idx]

            sample_idx = np.random.choice(
                X_all_trans.shape[0],
                min(
                    200,
                    X_all_trans.shape[0]
                ),
                replace=False
            )

            X_sample = X_all_trans[sample_idx]

            model_name = (
                type(model)
                .__name__
                .lower()
            )

            is_tree = any(
                t in model_name
                for t in [
                    'randomforest',
                    'gradientboosting',
                    'xgboost',
                    'lgbm',
                    'catboost'
                ]
            )

            if is_tree:

                explainer = shap.TreeExplainer(
                    model
                )

                shap_values_obj = (
                    explainer.shap_values(
                        X_sample
                    )
                )

                if isinstance(
                    shap_values_obj,
                    list
                ):
                    shap_values_obj = (
                        shap_values_obj[0]
                    )

            else:

                bg_small = X_background[
                    :min(
                        100,
                        X_background.shape[0]
                    )
                ]

                explainer = shap.KernelExplainer(
                    model.predict,
                    bg_small
                )

                shap_values_obj = (
                    explainer.shap_values(
                        X_sample,
                        nsamples=200
                    )
                )

                if isinstance(
                    shap_values_obj,
                    list
                ):
                    shap_values_obj = (
                        shap_values_obj[0]
                    )

            shap_values_obj = np.asarray(
                shap_values_obj
            )

            if shap_values_obj.ndim == 2:

                shap_values_top = (
                    shap_values_obj[:, top_indices]
                )

                X_sample_top = (
                    X_sample[:, top_indices]
                )

                plt.figure(
                    figsize=(6.5, 6)
                )

                shap.summary_plot(
                    shap_values_top,
                    X_sample_top,
                    feature_names=top_features,
                    show=False
                )

                cbar = plt.gcf().axes[-1]

                cbar.tick_params(
                    labelsize=18
                )

                cbar.set_ylabel(
                    "Feature value",
                    fontsize=22,
                    labelpad=2
                )

                cbar.yaxis.set_label_coords(
                    1.9,
                    0.5
                )

                shap_ax = plt.gca()

                shap_ax.tick_params(
                    labelsize=15
                )

                shap_ax.set_xlabel(
                    'SHAP value',
                    fontsize=20
                )

                plt.subplots_adjust(
                    left=0.23,
                    right=0.93,
                    bottom=0.14,
                    top=0.98
                )

                shap_png = os.path.join(
                    SAVE_FOLDER,
                    f"{best_key}_shap_summary_top12.png"
                )

                shap_svg = os.path.join(
                    SAVE_FOLDER,
                    f"{best_key}_shap_summary_top12.svg"
                )

                plt.savefig(
                    shap_png,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.02
                )

                plt.savefig(
                    shap_svg,
                    format='svg',
                    bbox_inches='tight',
                    pad_inches=0.02
                )

                plt.close()

            else:
                pass

        except Exception:
            pass

    X_top_all = X_all_trans[
        :,
        top_indices
    ]

    X_top_df = pd.DataFrame(
        X_top_all,
        columns=top_features
    )

    corr = X_top_df.corr()
    n = len(top_features)

    fig, ax = plt.subplots(
        figsize=(
            max(8, n * 0.8),
            max(6, n * 0.8)
        )
    )

    pos_color = "#e60e1c"
    neg_color = "#0e52e6"

    for i in range(n):

        for j in range(n):

            if i <= j:
                continue

            value = corr.iloc[i, j]

            radius = abs(value) * 1500

            color = (
                pos_color
                if value > 0
                else neg_color
            )

            ax.scatter(
                j,
                i,
                s=radius,
                color=color,
                alpha=0.85,
                edgecolor='white',
                linewidth=0.6
            )

    ax.set_xticks(
        np.arange(n)
    )

    ax.set_yticks(
        np.arange(n)
    )

    ax.set_xticklabels(
        top_features,
        rotation=90,
        fontsize=26
    )

    ax.set_yticklabels(
        top_features,
        fontsize=26
    )

    ax.set_xlim(
        -0.5,
        n - 0.5
    )

    ax.set_ylim(
        n - 0.5,
        -0.5
    )

    ax.set_aspect("equal")

    import matplotlib.patches as mpatches

    pos_patch = mpatches.Patch(
        color=pos_color,
        label='Positive'
    )

    neg_patch = mpatches.Patch(
        color=neg_color,
        label='Negative'
    )

    ax.legend(
        handles=[
            pos_patch,
            neg_patch
        ],
        loc='upper right',
        fontsize=20
    )

    plt.tight_layout()

    bubble_path = os.path.join(
        SAVE_FOLDER,
        f"{best_key}_corr_bubble_matrix_top12.png"
    )

    bubble_svg_path = os.path.join(
        SAVE_FOLDER,
        f"{best_key}_corr_bubble_matrix_top12.svg"
    )

    plt.savefig(
        bubble_path,
        dpi=300
    )

    plt.savefig(
        bubble_svg_path,
        format='svg'
    )

    plt.close()

    order = np.argsort(
        top_imps
    )[::-1]

    imp_sorted = top_imps[order]

    feat_sorted = [
        top_features[i]
        for i in order
    ]

    total_imp = (
        imp_sorted.sum()
    )

    if total_imp != 0:
        cum_ratio = (
            np.cumsum(imp_sorted)
            / total_imp
        )
    else:
        cum_ratio = np.zeros(
            len(imp_sorted)
        )

    fig, ax1 = plt.subplots(
        figsize=(6.5, 4.5)
    )

    ax1.bar(
        range(len(imp_sorted)),
        imp_sorted,
        alpha=0.85,
        color='steelblue'
    )

    ax1.set_ylabel(
        "Importance",
        color='b',
        fontsize=20
    )

    ax1.tick_params(
        axis='y',
        labelcolor='b',
        labelsize=18
    )

    ax2 = ax1.twinx()

    ax2.plot(
        range(len(imp_sorted)),
        cum_ratio,
        marker='o',
        color='red',
        linewidth=2.5
    )

    ax2.set_ylabel(
        "Cumulative Contribution",
        color='r',
        fontsize=20
    )

    ax2.set_ylim(
        0,
        1
    )

    ax2.tick_params(
        axis='y',
        labelcolor='r'
    )

    ax1.set_xticks(
        range(len(feat_sorted))
    )

    ax1.set_xticklabels(
        feat_sorted,
        rotation=90,
        ha='right',
        fontsize=18
    )

    plt.tight_layout()

    pareto_path = os.path.join(
        SAVE_FOLDER,
        f"{best_key}_pareto_top12.png"
    )

    pareto_svg_path = os.path.join(
        SAVE_FOLDER,
        f"{best_key}_pareto_top12.svg"
    )

    plt.savefig(
        pareto_path,
        dpi=300
    )

    plt.savefig(
        pareto_svg_path,
        format='svg'
    )

    plt.close()

    df_summary = pd.DataFrame({
        "feature": used_features,
        "importance": imps,
        "rank": np.argsort(
            np.argsort(-imps)
        ) + 1
    })

    df_summary = df_summary.sort_values(
        "importance",
        ascending=False
    )

    feature_csv = os.path.join(
        SAVE_FOLDER,
        f"{best_key}_feature_importance_all.csv"
    )

    df_summary.to_csv(
        feature_csv,
        index=False,
        encoding='utf-8-sig'
    )

    meta = compute_meta(
        best_key,
        best_pipeline,
        X,
        y,
        feature_cols,
        best_summary_row,
        imps_all=imps,
        shap_values_for_meta=shap_values_top,
        shap_feature_names=top_features
    )

    meta_path = os.path.join(
        SAVE_FOLDER,
        f"{best_key}_meta.pkl"
    )

    joblib.dump(
        meta,
        meta_path
    )

    deepseek_outputs = {
        'shap': call_deepseek(
            meta,
            'shap',
            FEATURE_DESCRIPTION
        ),
        'bubble': call_deepseek(
            meta,
            'bubble',
            FEATURE_DESCRIPTION
        ),
        'pareto': call_deepseek(
            meta,
            'pareto',
            FEATURE_DESCRIPTION
        ),
    }

    for plot_type, result in deepseek_outputs.items():
        with open(
            os.path.join(
                SAVE_FOLDER,
                f"{best_key}_deepseek_{plot_type}.txt"
            ),
            'w',
            encoding='utf-8'
        ) as f:
            f.write(result if result is not None else '')

    readme_path = os.path.join(SAVE_FOLDER, "README_results.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(
            "DielectroNet Nested Repeated 5-fold CV Results\n"
            "===============================================\n\n"
            "1. repeated_5fold_cv_summary.csv / repeated_5fold_cv_results.xlsx\n"
            "   Six-model Mean ± SD from 25 outer-fold evaluations.\n\n"
            "2. repeated_5fold_cv_all_results.csv\n"
            "   R2, RMSE and MAE for all 25 outer folds of all six models.\n\n"
            "3. all_models_25_outer_fold_predictions.csv\n"
            "   Raw outer-fold predictions for every sample, repeat and fold.\n\n"
            "4. <model>_averaged_oof_predictions.csv\n"
            "   For each sample: five repeat-level OOF predictions, their mean, "
            "and the sample-level prediction SD across the five repeats.\n\n"
            "5. all_models_averaged_oof_predictions.xlsx\n"
            "   The six model-level averaged OOF tables plus OOF metrics.\n\n"
            "6. <model>_averaged_oof_observed_vs_predicted.png/.svg\n"
            "   Observed vs predicted plots based on averaged OOF predictions.\n"
            "   The dashed diagonal is the ideal prediction line.\n\n"
            "7. all_models_averaged_oof_metrics.csv\n"
            "   R2/RMSE/MAE calculated from the averaged OOF predictions "
            "for each model, plus summary statistics of sample-level prediction SD.\n\n"
            "8. <best_model>_pipeline.pkl / *_features.pkl / *_best_params.txt\n"
            "   Final full-data models and their selected parameters for platform use.\n\n"
            "9. Feature-analysis files, *_meta.pkl and *_deepseek_*.txt\n"
            "   Generated for the model selected by the highest mean outer-fold R2.\n"
        )

    print(f"结果保存目录: {os.path.abspath(SAVE_FOLDER)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(BASE_DIR / "data" / "ye_train_data.xlsx"),
        help="Path to the input Excel dataset. The target must be the last numeric column."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for output files. Defaults to results/<dataset_name>."
    )
    args = parser.parse_args()

    FILE_PATH = Path(args.data).expanduser().resolve()
    SAVE_FOLDER = Path(args.output).expanduser().resolve() if args.output else BASE_DIR / "results" / FILE_PATH.stem
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    if FILE_PATH.is_file():
        process_excel_local(FILE_PATH)
    else:
        raise FileNotFoundError(f"Dataset file not found: {FILE_PATH}")
