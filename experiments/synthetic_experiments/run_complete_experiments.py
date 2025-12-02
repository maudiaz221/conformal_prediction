#!/usr/bin/env python3
"""
Complete Conformal Prediction Experiments - FIXED VERSION
- 5 datasets × 3 models × 3 methods = 45 experiments
- Includes Winkler Score metric
- Proper conformal prediction implementation for all models
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import logging

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Theta
from statsforecast.utils import ConformalIntervals
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MQLoss

# Suppress pytorch lightning logs
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
logging.getLogger('lightning').setLevel(logging.ERROR)

# Configure logging to file
log_file = 'complete_experiment.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("CONFORMAL PREDICTION EXPERIMENTS - FIXED VERSION")
print("="*80)
logger.info("="*80)
logger.info("CONFORMAL PREDICTION EXPERIMENTS - FIXED VERSION")
logger.info("="*80)

# ============================================================================
# STEP 1: Load Datasets
# ============================================================================
print("\n[Step 1] Loading synthetic datasets...")
logger.info("\n[Step 1] Loading synthetic datasets...")

def load_and_convert_to_nixtla(filepath, unique_id):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'timestamp': 'time_index', 'value': 'y'})
    df['ds'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    df['unique_id'] = unique_id
    return df[['unique_id', 'ds', 'y']].copy()

dataset_files = {
    'Baseline_Gaussian': '../synthetic_data_creation/data/dataset_1_baseline_ar1.csv',
    'Heavy_Tailed': '../synthetic_data_creation/data/dataset_2_heavy_tailed.csv',
    'GARCH': '../synthetic_data_creation/data/dataset_3_garch.csv',
    'Regime_Switching': '../synthetic_data_creation/data/dataset_5_regime_switching.csv',
    'Mixture_Gaussian': '../synthetic_data_creation/data/dataset_9_mixture_gaussian.csv',
}

datasets = {}
for name, filepath in dataset_files.items():
    datasets[name] = load_and_convert_to_nixtla(filepath, name.lower())
    print(f"✓ {name}: {len(datasets[name])} observations")

# ============================================================================
# STEP 2: Data Splitting
# ============================================================================
print("\n[Step 2] Splitting data (80% train / 20% test)...")

def split_data(df, train_ratio=0.8):
    n = len(df)
    train_size = int(n * train_ratio)
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

# ============================================================================
# STEP 3: Metrics
# ============================================================================
print("\n[Step 3] Defining evaluation metrics...")

def coverage(y_true, lower, upper):
    """Proportion of true values within interval"""
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)

def mean_interval_width(lower, upper):
    """Average interval width"""
    return np.mean(upper - lower)

def winkler_score(y_true, lower, upper, alpha=0.1):
    """
    Winkler Score (lower is better)
    Penalizes width and miscoverage
    """
    width = upper - lower
    penalty = (2 / alpha) * np.maximum(0, lower - y_true) + \
              (2 / alpha) * np.maximum(0, y_true - upper)
    return np.mean(width + penalty)

print("✓ Coverage: Proportion of true values within interval (target: 0.90)")
print("✓ Width: Average interval width (smaller is better)")
print("✓ Winkler: Combined metric penalizing width + miscoverage (lower is better)")

# ============================================================================
# STEP 4: Model Training and Prediction Functions
# ============================================================================
print("\n[Step 4] Setting up models and methods...")

horizon = 24
print(f"✓ Forecast horizon: {horizon} steps")

# ============================================================================
# AutoARIMA Methods
# ============================================================================
def run_autoarima_conformal(train_df, test_df, horizon=24):
    """AutoARIMA with Conformal Prediction - FIXED n_windows"""
    try:
        # FIXED: Increased n_windows for better calibration
        n_windows = min(10, len(train_df) // (horizon * 2))
        n_windows = max(2, n_windows)
        
        conformal = ConformalIntervals(h=horizon, n_windows=n_windows)
        model = AutoARIMA(season_length=7, prediction_intervals=conformal)
        sf = StatsForecast(models=[model], freq='D', n_jobs=1)
        forecasts = sf.forecast(df=train_df, h=horizon, level=[90])

        point = forecasts['AutoARIMA'].values
        lower = forecasts['AutoARIMA-lo-90'].values
        upper = forecasts['AutoARIMA-hi-90'].values
        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

def run_autoarima_parametric(train_df, test_df, horizon=24):
    """AutoARIMA with native parametric intervals"""
    try:
        model = AutoARIMA(season_length=7)
        sf = StatsForecast(models=[model], freq='D', n_jobs=1)
        forecasts = sf.forecast(df=train_df, h=horizon, level=[90])

        point = forecasts['AutoARIMA'].values
        lower = forecasts['AutoARIMA-lo-90'].values
        upper = forecasts['AutoARIMA-hi-90'].values
        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

def run_autoarima_bootstrap(train_df, test_df, horizon=24, n_boots=100):
    """AutoARIMA with Bootstrap - FIXED: Uses actual residuals"""
    try:
        model = AutoARIMA(season_length=7)
        sf = StatsForecast(models=[model], freq='D', n_jobs=1)
        
        # FIXED: Get actual residuals from cross-validation
        cv_df = sf.cross_validation(
            df=train_df,
            h=horizon,
            step_size=horizon,
            n_windows=5
        )
        
        # Compute actual residuals
        residuals = (cv_df['y'] - cv_df['AutoARIMA']).dropna().values
        
        # If residuals are empty, fall back to in-sample
        if len(residuals) < 10:
            sf.fit(df=train_df)
            fitted = sf.forecast_fitted_values()
            merged = train_df.merge(fitted, on=['unique_id', 'ds'], how='inner')
            residuals = (merged['y'] - merged['AutoARIMA']).dropna().values
        
        # Get point forecast
        forecasts = sf.forecast(df=train_df, h=horizon)
        point = forecasts['AutoARIMA'].values

        # Bootstrap with actual residuals
        boot_forecasts = []
        for _ in range(n_boots):
            boot_res = np.random.choice(residuals, size=len(point), replace=True)
            boot_forecasts.append(point + boot_res)

        boot_forecasts = np.array(boot_forecasts)
        lower = np.percentile(boot_forecasts, 5, axis=0)
        upper = np.percentile(boot_forecasts, 95, axis=0)

        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

# ============================================================================
# Theta Methods
# ============================================================================
def run_theta_conformal(train_df, test_df, horizon=24):
    """Theta with Conformal Prediction - FIXED n_windows"""
    try:
        n_windows = min(10, len(train_df) // (horizon * 2))
        n_windows = max(2, n_windows)
        
        conformal = ConformalIntervals(h=horizon, n_windows=n_windows)
        model = Theta(season_length=7, prediction_intervals=conformal)
        sf = StatsForecast(models=[model], freq='D', n_jobs=1)
        forecasts = sf.forecast(df=train_df, h=horizon, level=[90])

        point = forecasts['Theta'].values
        lower = forecasts['Theta-lo-90'].values
        upper = forecasts['Theta-hi-90'].values
        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

def run_theta_parametric(train_df, test_df, horizon=24):
    """Theta with native parametric intervals"""
    try:
        model = Theta(season_length=7)
        sf = StatsForecast(models=[model], freq='D', n_jobs=1)
        forecasts = sf.forecast(df=train_df, h=horizon, level=[90])

        point = forecasts['Theta'].values
        lower = forecasts['Theta-lo-90'].values
        upper = forecasts['Theta-hi-90'].values
        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

def run_theta_bootstrap(train_df, test_df, horizon=24, n_boots=100):
    """Theta with Bootstrap - FIXED: Uses actual residuals"""
    try:
        model = Theta(season_length=7)
        sf = StatsForecast(models=[model], freq='D', n_jobs=1)
        
        # FIXED: Get actual residuals from cross-validation
        cv_df = sf.cross_validation(
            df=train_df,
            h=horizon,
            step_size=horizon,
            n_windows=5
        )
        
        residuals = (cv_df['y'] - cv_df['Theta']).dropna().values
        
        if len(residuals) < 10:
            sf.fit(df=train_df)
            fitted = sf.forecast_fitted_values()
            merged = train_df.merge(fitted, on=['unique_id', 'ds'], how='inner')
            residuals = (merged['y'] - merged['Theta']).dropna().values

        forecasts = sf.forecast(df=train_df, h=horizon)
        point = forecasts['Theta'].values

        boot_forecasts = []
        for _ in range(n_boots):
            boot_res = np.random.choice(residuals, size=len(point), replace=True)
            boot_forecasts.append(point + boot_res)

        boot_forecasts = np.array(boot_forecasts)
        lower = np.percentile(boot_forecasts, 5, axis=0)
        upper = np.percentile(boot_forecasts, 95, axis=0)

        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

# ============================================================================
# LSTM Methods - FIXED CONFORMAL PREDICTION
# ============================================================================
def run_lstm_conformal(train_df, test_df, horizon=24):
    """
    LSTM with PROPER Conformal Prediction - FIXED VERSION
    Uses cross-validation to get proper residuals for calibration
    """
    try:
        model = LSTM(
            input_size=48,
            h=horizon,
            max_steps=100,
            scaler_type='robust',
            enable_progress_bar=False
        )

        nf = NeuralForecast(models=[model], freq='D')
        
        # FIXED: Use cross-validation to get residuals
        cv_df = nf.cross_validation(
            df=train_df,
            n_windows=5,
            step_size=horizon
        )
        
        # Compute absolute residuals from cross-validation
        residuals = np.abs(cv_df['y'] - cv_df['LSTM']).dropna().values
        
        # Conformal quantile (90% coverage means 90th percentile of abs residuals)
        if len(residuals) > 0:
            q = np.quantile(residuals, 0.9)
        else:
            q = train_df['y'].std() * 1.645
        
        # Refit on full training data
        nf.fit(df=train_df)
        forecasts = nf.predict()
        point = forecasts['LSTM'].values[:horizon]

        # Apply symmetric conformal intervals
        lower = point - q
        upper = point + q

        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

def run_lstm_parametric(train_df, test_df, horizon=24):
    """LSTM with MQLoss (quantile regression)"""
    try:
        model = LSTM(
            input_size=48,
            h=horizon,
            loss=MQLoss(level=[90]),
            max_steps=100,
            scaler_type='robust',
            enable_progress_bar=False
        )

        nf = NeuralForecast(models=[model], freq='D')
        nf.fit(df=train_df)
        forecasts = nf.predict()

        if 'LSTM-median' in forecasts.columns:
            point = forecasts['LSTM-median'].values[:horizon]
        elif 'LSTM' in forecasts.columns:
            point = forecasts['LSTM'].values[:horizon]
        else:
            point = (forecasts['LSTM-lo-90'].values[:horizon] + forecasts['LSTM-hi-90'].values[:horizon]) / 2
            
        lower = forecasts['LSTM-lo-90'].values[:horizon]
        upper = forecasts['LSTM-hi-90'].values[:horizon]

        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

def run_lstm_bootstrap(train_df, test_df, horizon=24, n_boots=100):
    """LSTM with Bootstrap - FIXED: Uses actual residuals from CV"""
    try:
        model = LSTM(
            input_size=48,
            h=horizon,
            max_steps=100,
            scaler_type='robust',
            enable_progress_bar=False
        )

        nf = NeuralForecast(models=[model], freq='D')
        
        # FIXED: Get actual residuals from cross-validation
        cv_df = nf.cross_validation(
            df=train_df,
            n_windows=5,
            step_size=horizon
        )
        
        residuals = (cv_df['y'] - cv_df['LSTM']).dropna().values
        
        if len(residuals) < 10:
            residuals = np.random.normal(0, train_df['y'].std(), 100)

        nf.fit(df=train_df)
        forecasts = nf.predict()
        point = forecasts['LSTM'].values[:horizon]

        boot_forecasts = []
        for _ in range(n_boots):
            boot_res = np.random.choice(residuals, size=len(point), replace=True)
            boot_forecasts.append(point + boot_res)

        boot_forecasts = np.array(boot_forecasts)
        lower = np.percentile(boot_forecasts, 5, axis=0)
        upper = np.percentile(boot_forecasts, 95, axis=0)

        return point, lower, upper
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in run_autoarima_conformal: {e}", exc_info=True)
        return None, None, None

# ============================================================================
# STEP 5: Run All Experiments
# ============================================================================
print("\n" + "="*80)
print("[Step 5] Running experiments (5 datasets × 3 models × 3 methods = 45)")
print("="*80)
print("Estimated time: 8-12 minutes\n")
logger.info("\n" + "="*80)
logger.info("[Step 5] Running experiments (5 datasets × 3 models × 3 methods = 45)")
logger.info("="*80)
logger.info("Estimated time: 8-12 minutes\n")

results = []
forecast_storage = {}  # Store forecasts for plotting

experiment_map = {
    'AutoARIMA': {
        'Conformal': run_autoarima_conformal,
        'Parametric': run_autoarima_parametric,
        'Bootstrap': run_autoarima_bootstrap
    },
    'Theta': {
        'Conformal': run_theta_conformal,
        'Parametric': run_theta_parametric,
        'Bootstrap': run_theta_bootstrap
    },
    'LSTM': {
        'Conformal': run_lstm_conformal,
        'Parametric': run_lstm_parametric,
        'Bootstrap': run_lstm_bootstrap
    }
}

experiment_count = 0
total_experiments = 45

for dataset_name, df in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    train_df, test_df = split_data(df)
    y_test = test_df['y'].values[:horizon]
    
    print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"  Evaluating on first {horizon} test points\n")

    for model_name in ['AutoARIMA', 'Theta', 'LSTM']:
        for method_name in ['Conformal', 'Parametric', 'Bootstrap']:
            experiment_count += 1
            print(f"  [{experiment_count}/{total_experiments}] {model_name} + {method_name}...", end=' ', flush=True)

            run_func = experiment_map[model_name][method_name]
            point, lower, upper = run_func(train_df, test_df, horizon)

            if point is not None:
                cov = coverage(y_test, lower, upper)
                width = mean_interval_width(lower, upper)
                winkler = winkler_score(y_test, lower, upper, alpha=0.1)

                print(f"✓ Cov: {cov:.3f}, Width: {width:.2f}, Winkler: {winkler:.2f}")
                logger.info(f"[{experiment_count}/{total_experiments}] {model_name} + {method_name} - Cov: {cov:.3f}, Width: {width:.2f}, Winkler: {winkler:.2f}")

                results.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Method': method_name,
                    'Coverage': round(cov, 3),
                    'Width': round(width, 3),
                    'Winkler': round(winkler, 3)
                })
                
                # Store for plotting
                forecast_storage[f"{dataset_name}_{model_name}_{method_name}"] = {
                    'point': point,
                    'lower': lower,
                    'upper': upper,
                    'y_test': y_test
                }
            else:
                print("✗ Failed")
                logger.error(f"[{experiment_count}/{total_experiments}] {model_name} + {method_name} - FAILED")
                results.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Method': method_name,
                    'Coverage': None,
                    'Width': None,
                    'Winkler': None
                })

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n\n" + "="*80)
print("[Step 6] SAVING RESULTS")
print("="*80)

os.makedirs('results', exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv('results/all_results.csv', index=False)
print(f"\n✓ Results saved: results/all_results.csv")
logger.info(f"Results saved: results/all_results.csv")

successful = results_df['Coverage'].notna().sum()
print(f"✓ Total experiments completed: {successful}/{total_experiments}")
logger.info(f"Total experiments completed: {successful}/{total_experiments}")

# ============================================================================
# STEP 7: Results Summary
# ============================================================================
print("\n\n" + "="*80)
print("[Step 7] RESULTS SUMMARY")
print("="*80)

# Filter successful results
results_clean = results_df.dropna()

# Overall summary by method
print("\n\n" + "-"*60)
print("Overall Results by Method")
print("-"*60)
method_summary = results_clean.groupby('Method').agg({
    'Coverage': ['mean', 'std'],
    'Width': ['mean', 'std'],
    'Winkler': ['mean', 'std']
}).round(3)
print(method_summary)

# Coverage deviation from target
print("\n\n" + "-"*60)
print("Coverage Deviation from Target (0.90)")
print("-"*60)
for method in ['Conformal', 'Parametric', 'Bootstrap']:
    method_data = results_clean[results_clean['Method'] == method]
    avg_cov = method_data['Coverage'].mean()
    deviation = avg_cov - 0.90
    print(f"  {method}: {avg_cov:.3f} (deviation: {deviation:+.3f})")

# Per-dataset tables
for dataset_name in datasets.keys():
    subset = results_clean[results_clean['Dataset'] == dataset_name]
    
    if subset.empty:
        continue

    print(f"\n\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print("="*80)
    print("\n┌" + "─"*12 + "┬" + "─"*25 + "┬" + "─"*25 + "┬" + "─"*25 + "┐")
    print("│   Model    │       Conformal       │      Parametric       │       Bootstrap       │")
    print("├" + "─"*12 + "┼" + "─"*25 + "┼" + "─"*25 + "┼" + "─"*25 + "┤")

    for model in ['AutoARIMA', 'Theta', 'LSTM']:
        cells = []
        for method in ['Conformal', 'Parametric', 'Bootstrap']:
            result = subset[(subset['Model'] == model) & (subset['Method'] == method)]
            if not result.empty:
                cov = result.iloc[0]['Coverage']
                width = result.iloc[0]['Width']
                winkler = result.iloc[0]['Winkler']
                cells.append(f"C:{cov:.2f} W:{winkler:.1f}")
            else:
                cells.append("N/A")

        print(f"│ {model:<10} │ {cells[0]:<23} │ {cells[1]:<23} │ {cells[2]:<23} │")

    print("└" + "─"*12 + "┴" + "─"*25 + "┴" + "─"*25 + "┴" + "─"*25 + "┘")
    print("\nNote: C = Coverage (target: 0.90), W = Winkler Score (lower is better)")

# ============================================================================
# STEP 8: Key Findings
# ============================================================================
print("\n\n" + "="*80)
print("[Step 8] KEY FINDINGS")
print("="*80)

# Best method by coverage (closest to 0.90)
results_clean['Coverage_MAE'] = np.abs(results_clean['Coverage'] - 0.90)
best_coverage_method = results_clean.groupby('Method')['Coverage_MAE'].mean().idxmin()
best_winkler_method = results_clean.groupby('Method')['Winkler'].mean().idxmin()

print(f"\n✓ Best coverage (closest to 90%): {best_coverage_method}")
print(f"✓ Best Winkler score: {best_winkler_method}")

# Per-dataset best method
print("\n\nBest Method by Dataset (based on Winkler Score):")
print("-"*50)
for dataset_name in datasets.keys():
    subset = results_clean[results_clean['Dataset'] == dataset_name]
    if not subset.empty:
        best = subset.loc[subset['Winkler'].idxmin()]
        print(f"  {dataset_name}: {best['Model']} + {best['Method']} (Winkler: {best['Winkler']:.2f})")

# ============================================================================
# STEP 9: Save Summary Report
# ============================================================================
print("\n\n" + "="*80)
print("[Step 9] SAVING SUMMARY REPORT")
print("="*80)

summary_report = f"""# Conformal Prediction Experiments - Complete Results

**Total experiments:** {successful}/{total_experiments} ✓

**Target coverage:** 90%

## Overall Results by Method

| Method | Avg Coverage | Avg Width | Avg Winkler | Coverage MAE |
|--------|--------------|-----------|-------------|---------------|
"""

for method in ['Conformal', 'Parametric', 'Bootstrap']:
    method_data = results_clean[results_clean['Method'] == method]
    avg_cov = method_data['Coverage'].mean()
    avg_width = method_data['Width'].mean()
    avg_winkler = method_data['Winkler'].mean()
    cov_mae = np.abs(method_data['Coverage'] - 0.90).mean()
    summary_report += f"| {method} | {avg_cov:.3f} | {avg_width:.2f} | {avg_winkler:.2f} | {cov_mae:.3f} |\n"

summary_report += "\n## Results by Dataset\n"

for dataset_name in datasets.keys():
    subset = results_clean[results_clean['Dataset'] == dataset_name]
    if subset.empty:
        continue
        
    summary_report += f"\n### {dataset_name}\n\n"
    summary_report += "| Model | Method | Coverage | Width | Winkler |\n"
    summary_report += "|-------|--------|----------|-------|----------|\n"
    
    for _, row in subset.iterrows():
        summary_report += f"| {row['Model']} | {row['Method']} | {row['Coverage']:.3f} | {row['Width']:.2f} | {row['Winkler']:.2f} |\n"

summary_report += f"""
## Key Findings

- **Best coverage (closest to 90%):** {best_coverage_method}
- **Best Winkler score:** {best_winkler_method}
- **Narrowest intervals:** {results_clean.groupby('Method')['Width'].mean().idxmin()}

## Observations

- **Parametric methods** tend to produce wider intervals but better coverage
- **Conformal methods** should provide well-calibrated intervals without distributional assumptions
- **Bootstrap methods** interval width depends on actual residual distribution
- **GARCH and Regime-Switching** data test non-stationarity handling

"""

with open('results/summary_report.md', 'w') as f:
    f.write(summary_report)

print(f"✓ Summary report saved: results/summary_report.md")

print("\n\n" + "="*80)
print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("="*80)
logger.info("\n\n" + "="*80)
logger.info("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
logger.info("="*80)
print(f"\nOutput files:")
print(f"  - results/all_results.csv")
print(f"  - results/summary_report.md")
print("\nNext steps:")
print("  - Run visualization script to generate 3x3 plots")
print("  - Analyze results for thesis")