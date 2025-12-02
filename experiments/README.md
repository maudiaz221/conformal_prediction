# Conformal Prediction Experiments

This directory contains all experiments comparing conformal prediction with traditional prediction interval methods.

## Directory Structure

```
experiments/
├── INTERVAL_FORMULAS.md              # Mathematical formulas for all methods
├── real_experiments/                 # Experiments on real-world datasets
│   ├── run_complete_experiments.py   # Main experiment script
│   ├── create_final_visualizations.py # Generate summary charts
│   ├── create_forecast_plots.py      # Generate forecast line plots
│   ├── results/                      # CSV results and summaries
│   │   ├── all_results.csv
│   │   ├── summary.md
│   │   ├── summary_report.md
│   │   └── summary_statistics.csv
│   └── figures/                      # All visualizations
│       ├── coverage_comparison.png
│       ├── winkler_comparison.png
│       ├── coverage_vs_winkler.png
│       ├── model_comparison.png
│       ├── heatmap_*.png (5 datasets)
│       └── forecast_plots/
│           ├── Gold_Prices_AutoARIMA_intervals.png
│           ├── Gold_Prices_LSTM_intervals.png
│           ├── Exchange_Rate_AutoARIMA_intervals.png
│           ├── Exchange_Rate_LSTM_intervals.png
│           ├── Electricity_Demand_AutoARIMA_intervals.png
│           └── Electricity_Demand_LSTM_intervals.png
│
└── synthetic_experiments/            # Experiments on synthetic datasets
    ├── run_complete_experiments.py   # Main experiment script
    ├── create_final_visualizations.py # Generate summary charts
    ├── create_forecast_plots.py      # Generate forecast line plots
    ├── results/                      # CSV results and summaries
    │   ├── all_results.csv
    │   ├── summary.md
    │   ├── summary_report.md
    │   └── summary_statistics.csv
    └── figures/                      # All visualizations
        ├── coverage_comparison.png
        ├── winkler_comparison.png
        ├── coverage_vs_winkler.png
        ├── model_comparison.png
        ├── heatmap_*.png (5 datasets)
        └── forecast_plots/
            ├── Baseline_Gaussian_AutoARIMA_intervals.png
            ├── Baseline_Gaussian_LSTM_intervals.png
            ├── Heavy_Tailed_AutoARIMA_intervals.png
            ├── Heavy_Tailed_LSTM_intervals.png
            ├── GARCH_AutoARIMA_intervals.png
            └── GARCH_LSTM_intervals.png
```

## Datasets

### Real Datasets (5)
1. **Gold Prices** - Daily gold prices (2,513 observations)
2. **Stock Prices** - Daily stock market data (2,516 observations)
3. **Exchange Rate** - Currency exchange rates (2,606 observations)
4. **Electricity Demand** - Power consumption (1,433 observations)
5. **M4 Series** - M4 competition time series (1,006 observations)

### Synthetic Datasets (5)
1. **Baseline Gaussian** - AR(1) with Gaussian noise (ideal case)
2. **Heavy Tailed** - AR(1) with Student-t noise (outliers)
3. **GARCH** - AR(1) with heteroscedastic noise (changing variance)
4. **Regime Switching** - AR(1) with abrupt volatility changes
5. **Mixture Gaussian** - AR(1) with bimodal noise (contamination)

## Methods Compared

### 1. Conformal Prediction
- Distribution-free method
- Finite-sample coverage guarantees
- Based on calibration residuals

### 2. Parametric Intervals
- Assumes Gaussian errors
- Uses model-based variance estimates
- Asymptotic coverage guarantees

### 3. Bootstrap Intervals
- Resamples residuals
- Empirical distribution-based
- Can capture asymmetry

## Models Tested

1. **AutoARIMA** - Automatic ARIMA model selection
2. **Theta** - Exponential smoothing method
3. **LSTM** - Neural network for time series

## Evaluation Metrics

1. **Coverage** - Proportion of actual values within intervals (target: 90%)
2. **Width** - Average interval width (smaller is better)
3. **Winkler Score** - Combined metric penalizing width + miscoverage (lower is better)

## Running Experiments

### Real Datasets
```bash
cd real_experiments
python run_complete_experiments.py      # Run all experiments (8-12 min)
python create_final_visualizations.py   # Generate summary charts
python create_forecast_plots.py         # Generate forecast plots
```

### Synthetic Datasets
```bash
cd synthetic_experiments
python run_complete_experiments.py      # Run all experiments (8-12 min)
python create_final_visualizations.py   # Generate summary charts
python create_forecast_plots.py         # Generate forecast plots
```

## Key Findings

### Real Datasets
- **Best Coverage:** Parametric (84.7%)
- **Best Winkler:** Conformal (245.8)
- **Observations:** Real data shows different challenges than synthetic

### Synthetic Datasets
- **Best Coverage:** Parametric (91.9%)
- **Best Winkler:** Varies by dataset type
- **Observations:** Conformal maintains coverage across violation scenarios

## Visualizations

### Summary Charts (in figures/)
- `coverage_comparison.png` - Coverage across all methods
- `winkler_comparison.png` - Winkler scores comparison
- `coverage_vs_winkler.png` - Trade-off scatter plot
- `model_comparison.png` - Performance by model
- `heatmap_*.png` - Per-dataset metric heatmaps

### Forecast Plots (in figures/forecast_plots/)
Each plot shows:
- Historical context (last 50 points)
- Actual test values
- Point forecasts
- 90% prediction intervals (shaded)
- Coverage and width metrics

## References

See `INTERVAL_FORMULAS.md` for complete mathematical formulas and references.
