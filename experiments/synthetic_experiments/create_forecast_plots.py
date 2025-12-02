#!/usr/bin/env python3
"""
Generate forecast plots with prediction intervals for synthetic datasets
Shows actual values, point forecasts, and shaded prediction intervals
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

print("="*80)
print("GENERATING FORECAST PLOTS WITH PREDICTION INTERVALS - SYNTHETIC DATASETS")
print("="*80)

# ============================================================================
# Load and prepare data
# ============================================================================
print("\n[Step 1] Loading synthetic datasets...")

def load_and_convert_to_nixtla(filepath, unique_id):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'timestamp': 'time_index', 'value': 'y'})
    df['ds'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    df['unique_id'] = unique_id
    return df[['unique_id', 'ds', 'y']].copy()

dataset_files = {
    'Baseline_Gaussian': '../../synthetic_data_creation/data/dataset_1_baseline_ar1.csv',
    'Heavy_Tailed': '../../synthetic_data_creation/data/dataset_2_heavy_tailed.csv',
    'GARCH': '../../synthetic_data_creation/data/dataset_3_garch.csv',
}

datasets = {}
for name, filepath in dataset_files.items():
    datasets[name] = load_and_convert_to_nixtla(filepath, name.lower())
    print(f"✓ {name}: {len(datasets[name])} observations")

def split_data(df, train_ratio=0.8):
    n = len(df)
    train_size = int(n * train_ratio)
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

horizon = 24

# ============================================================================
# Model functions (simplified from main experiment)
# ============================================================================

def run_autoarima_conformal(train_df, horizon=24):
    n_windows = min(10, len(train_df) // (horizon * 2))
    n_windows = max(2, n_windows)
    conformal = ConformalIntervals(h=horizon, n_windows=n_windows)
    model = AutoARIMA(season_length=7, prediction_intervals=conformal)
    sf = StatsForecast(models=[model], freq='D', n_jobs=1)
    forecasts = sf.forecast(df=train_df, h=horizon, level=[90])
    return forecasts['AutoARIMA'].values, forecasts['AutoARIMA-lo-90'].values, forecasts['AutoARIMA-hi-90'].values

def run_autoarima_parametric(train_df, horizon=24):
    model = AutoARIMA(season_length=7)
    sf = StatsForecast(models=[model], freq='D', n_jobs=1)
    forecasts = sf.forecast(df=train_df, h=horizon, level=[90])
    return forecasts['AutoARIMA'].values, forecasts['AutoARIMA-lo-90'].values, forecasts['AutoARIMA-hi-90'].values

def run_autoarima_bootstrap(train_df, horizon=24, n_boots=100):
    model = AutoARIMA(season_length=7)
    sf = StatsForecast(models=[model], freq='D', n_jobs=1)
    cv_df = sf.cross_validation(df=train_df, h=horizon, step_size=horizon, n_windows=5)
    residuals = (cv_df['y'] - cv_df['AutoARIMA']).dropna().values
    if len(residuals) < 10:
        sf.fit(df=train_df)
        fitted = sf.forecast_fitted_values()
        merged = train_df.merge(fitted, on=['unique_id', 'ds'], how='inner')
        residuals = (merged['y'] - merged['AutoARIMA']).dropna().values
    forecasts = sf.forecast(df=train_df, h=horizon)
    point = forecasts['AutoARIMA'].values
    boot_forecasts = []
    for _ in range(n_boots):
        boot_res = np.random.choice(residuals, size=len(point), replace=True)
        boot_forecasts.append(point + boot_res)
    boot_forecasts = np.array(boot_forecasts)
    lower = np.percentile(boot_forecasts, 5, axis=0)
    upper = np.percentile(boot_forecasts, 95, axis=0)
    return point, lower, upper

def run_theta_conformal(train_df, horizon=24):
    n_windows = min(10, len(train_df) // (horizon * 2))
    n_windows = max(2, n_windows)
    conformal = ConformalIntervals(h=horizon, n_windows=n_windows)
    model = Theta(season_length=7, prediction_intervals=conformal)
    sf = StatsForecast(models=[model], freq='D', n_jobs=1)
    forecasts = sf.forecast(df=train_df, h=horizon, level=[90])
    return forecasts['Theta'].values, forecasts['Theta-lo-90'].values, forecasts['Theta-hi-90'].values

def run_theta_parametric(train_df, horizon=24):
    model = Theta(season_length=7)
    sf = StatsForecast(models=[model], freq='D', n_jobs=1)
    forecasts = sf.forecast(df=train_df, h=horizon, level=[90])
    return forecasts['Theta'].values, forecasts['Theta-lo-90'].values, forecasts['Theta-hi-90'].values

def run_theta_bootstrap(train_df, horizon=24, n_boots=100):
    model = Theta(season_length=7)
    sf = StatsForecast(models=[model], freq='D', n_jobs=1)
    cv_df = sf.cross_validation(df=train_df, h=horizon, step_size=horizon, n_windows=5)
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

def run_lstm_conformal(train_df, horizon=24):
    model = LSTM(input_size=48, h=horizon, max_steps=100, scaler_type='robust', enable_progress_bar=False)
    nf = NeuralForecast(models=[model], freq='D')
    cv_df = nf.cross_validation(df=train_df, n_windows=5, step_size=horizon)
    residuals = np.abs(cv_df['y'] - cv_df['LSTM']).dropna().values
    q = np.quantile(residuals, 0.9) if len(residuals) > 0 else train_df['y'].std() * 1.645
    nf.fit(df=train_df)
    forecasts = nf.predict()
    point = forecasts['LSTM'].values[:horizon]
    return point, point - q, point + q

def run_lstm_parametric(train_df, horizon=24):
    model = LSTM(input_size=48, h=horizon, loss=MQLoss(level=[90]), max_steps=100, scaler_type='robust', enable_progress_bar=False)
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

def run_lstm_bootstrap(train_df, horizon=24, n_boots=100):
    model = LSTM(input_size=48, h=horizon, max_steps=100, scaler_type='robust', enable_progress_bar=False)
    nf = NeuralForecast(models=[model], freq='D')
    cv_df = nf.cross_validation(df=train_df, n_windows=5, step_size=horizon)
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

# ============================================================================
# Generate forecasts and plots
# ============================================================================
print("\n[Step 2] Generating forecasts and creating plots...")

os.makedirs('figures/forecast_plots', exist_ok=True)

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

method_colors = {
    'Conformal': '#2ecc71',
    'Parametric': '#3498db',
    'Bootstrap': '#e74c3c'
}

plot_count = 0
total_plots = len(datasets) * len(experiment_map) * 3  # 3 datasets × 3 models × 3 methods = 27

for dataset_name, df in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    train_df, test_df = split_data(df)
    y_test = test_df['y'].values[:horizon]

    # Get the last 50 points of training data for context in plots
    context_size = 50
    y_context = train_df['y'].values[-context_size:]

    for model_name, methods in experiment_map.items():
        print(f"\n  Model: {model_name}")

        # Create a figure with 3 subplots (one per method)
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(f'{dataset_name} - {model_name} Prediction Intervals',
                     fontsize=16, fontweight='bold')

        for idx, (method_name, run_func) in enumerate(methods.items()):
            plot_count += 1
            print(f"    [{plot_count}/{total_plots}] {method_name}...", end=' ', flush=True)

            ax = axes[idx]

            try:
                # Generate forecast
                point, lower, upper = run_func(train_df, horizon)

                # Prepare x-axis
                x_context = np.arange(-context_size, 0)
                x_forecast = np.arange(0, horizon)

                # Plot context (last part of training data)
                ax.plot(x_context, y_context, 'o-', color='gray', alpha=0.6,
                       label='Historical', markersize=3, linewidth=1.5)

                # Plot actual test values
                ax.plot(x_forecast, y_test, 'o-', color='black',
                       label='Actual', markersize=5, linewidth=2, zorder=5)

                # Plot point forecast
                ax.plot(x_forecast, point, 's--', color=method_colors[method_name],
                       label='Forecast', markersize=4, linewidth=2, zorder=4)

                # Plot prediction interval (shaded area)
                ax.fill_between(x_forecast, lower, upper,
                               color=method_colors[method_name], alpha=0.3,
                               label='90% Prediction Interval')

                # Add vertical line separating training and test
                ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.5)

                # Formatting
                ax.set_xlabel('Time Steps', fontsize=11, fontweight='bold')
                ax.set_ylabel('Value', fontsize=11, fontweight='bold')
                ax.set_title(f'{method_name}', fontsize=13, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Calculate coverage for annotation
                covered = np.sum((y_test >= lower) & (y_test <= upper))
                coverage = covered / len(y_test)
                width = np.mean(upper - lower)

                # Add text box with metrics
                textstr = f'Coverage: {coverage:.1%}\nAvg Width: {width:.2f}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)

                print("✓")

            except Exception as e:
                print(f"✗ Error: {e}")
                ax.text(0.5, 0.5, f'Error generating forecast\n{str(e)[:50]}',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        safe_name = dataset_name.replace(' ', '_').replace('/', '_')
        filename = f'figures/forecast_plots/{safe_name}_{model_name}_intervals.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

print("\n" + "="*80)
print("ALL FORECAST PLOTS COMPLETE!")
print("="*80)
print("\nGenerated files in: figures/forecast_plots/")
print(f"Total plots created: {plot_count}")
