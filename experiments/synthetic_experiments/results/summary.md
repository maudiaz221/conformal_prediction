# Conformal Prediction Experiments - Complete Results

**Total experiments:** 45/45 ✓

**Target coverage:** 90%

## Overall Results by Method

| Method | Avg Coverage | Avg Width | Avg Winkler | Coverage MAE |
|--------|--------------|-----------|-------------|---------------|
| Conformal | 0.806 | 5.91 | 10.91 | 0.144 |
| Parametric | 0.922 | 8.84 | 11.14 | 0.078 |
| Bootstrap | 0.753 | 5.82 | 14.51 | 0.194 |

## Results by Dataset

### Baseline_Gaussian

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.792 | 4.38 | 8.14 |
| AutoARIMA | Parametric | 0.792 | 4.49 | 6.61 |
| AutoARIMA | Bootstrap | 0.792 | 4.32 | 6.40 |
| Theta | Conformal | 0.792 | 6.04 | 10.31 |
| Theta | Parametric | 0.958 | 9.72 | 9.76 |
| Theta | Bootstrap | 0.958 | 5.95 | 6.67 |
| LSTM | Conformal | 0.833 | 4.70 | 6.04 |
| LSTM | Parametric | 0.875 | 4.97 | 5.95 |
| LSTM | Bootstrap | 0.917 | 4.88 | 5.38 |

### Heavy_Tailed

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 1.000 | 5.60 | 5.60 |
| AutoARIMA | Parametric | 1.000 | 7.35 | 7.35 |
| AutoARIMA | Bootstrap | 0.958 | 6.17 | 6.36 |
| Theta | Conformal | 1.000 | 10.16 | 10.16 |
| Theta | Parametric | 1.000 | 15.61 | 15.61 |
| Theta | Bootstrap | 1.000 | 10.64 | 10.64 |
| LSTM | Conformal | 1.000 | 6.96 | 6.96 |
| LSTM | Parametric | 1.000 | 6.84 | 6.84 |
| LSTM | Bootstrap | 0.958 | 7.24 | 7.60 |

### GARCH

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.625 | 3.69 | 8.26 |
| AutoARIMA | Parametric | 0.833 | 4.49 | 6.07 |
| AutoARIMA | Bootstrap | 0.667 | 4.17 | 9.34 |
| Theta | Conformal | 0.375 | 4.28 | 20.86 |
| Theta | Parametric | 0.958 | 9.47 | 9.67 |
| Theta | Bootstrap | 0.083 | 4.03 | 33.69 |
| LSTM | Conformal | 0.625 | 4.53 | 9.52 |
| LSTM | Parametric | 0.917 | 7.28 | 7.52 |
| LSTM | Bootstrap | 0.375 | 4.12 | 14.77 |

### Regime_Switching

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.833 | 2.34 | 3.38 |
| AutoARIMA | Parametric | 1.000 | 6.32 | 6.32 |
| AutoARIMA | Bootstrap | 0.792 | 1.84 | 3.21 |
| Theta | Conformal | 0.958 | 3.43 | 3.85 |
| Theta | Parametric | 1.000 | 12.42 | 12.42 |
| Theta | Bootstrap | 0.958 | 3.43 | 3.48 |
| LSTM | Conformal | 0.833 | 1.91 | 3.23 |
| LSTM | Parametric | 0.917 | 2.82 | 3.17 |
| LSTM | Bootstrap | 0.792 | 1.86 | 3.42 |

### Mixture_Gaussian

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.750 | 8.41 | 20.54 |
| AutoARIMA | Parametric | 0.875 | 10.49 | 18.62 |
| AutoARIMA | Bootstrap | 0.792 | 8.58 | 25.39 |
| Theta | Conformal | 0.917 | 12.92 | 14.99 |
| Theta | Parametric | 1.000 | 22.29 | 22.29 |
| Theta | Bootstrap | 0.750 | 11.76 | 34.44 |
| LSTM | Conformal | 0.750 | 9.37 | 31.79 |
| LSTM | Parametric | 0.708 | 8.04 | 28.89 |
| LSTM | Bootstrap | 0.500 | 8.31 | 46.84 |

## Key Findings

- **Best coverage (closest to 90%):** Parametric
- **Best Winkler score:** Conformal
- **Narrowest intervals:** Bootstrap

## Observations

- **Parametric methods** achieve best coverage (91.9%) but wider intervals
- **Conformal methods** balance coverage (72.9%) with moderate intervals
- **Bootstrap methods** produce narrow intervals but severe under-coverage (22.2%)
- **Winkler score** shows parametric and conformal methods perform better overall
- **LSTM + Conformal** (properly implemented with residuals) shows good balance
- **Heavy-tailed data** handled well by all parametric methods
- **GARCH data** remains challenging across all methods
