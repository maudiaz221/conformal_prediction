# Conformal Prediction Experiments - Real Datasets Results

**Total experiments:** 45/45 ✓

**Target coverage:** 90%

## Overall Results by Method

| Method | Avg Coverage | Avg Width | Avg Winkler | Coverage MAE |
|--------|--------------|-----------|-------------|---------------|
| Conformal | 0.789 | 196.15 | 245.83 | 0.177 |
| Parametric | 0.847 | 215.71 | 267.91 | 0.162 |
| Bootstrap | 0.728 | 181.60 | 252.68 | 0.200 |

## Results by Dataset

### Gold_Prices

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.500 | 155.25 | 297.41 |
| AutoARIMA | Parametric | 0.500 | 160.09 | 281.85 |
| AutoARIMA | Bootstrap | 0.917 | 207.08 | 246.12 |
| Theta | Conformal | 0.542 | 156.34 | 259.96 |
| Theta | Parametric | 0.500 | 159.08 | 326.46 |
| Theta | Bootstrap | 0.833 | 203.96 | 218.40 |
| LSTM | Conformal | 0.417 | 191.03 | 535.48 |
| LSTM | Parametric | 1.000 | 452.18 | 452.18 |
| LSTM | Bootstrap | 0.375 | 191.10 | 641.58 |

### Stock_Prices

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.958 | 572.20 | 602.64 |
| AutoARIMA | Parametric | 0.667 | 365.67 | 501.43 |
| AutoARIMA | Bootstrap | 0.917 | 563.69 | 651.66 |
| Theta | Conformal | 0.875 | 548.37 | 638.53 |
| Theta | Parametric | 0.500 | 370.49 | 726.37 |
| Theta | Bootstrap | 0.917 | 563.68 | 698.45 |
| LSTM | Conformal | 1.000 | 1046.64 | 1046.64 |
| LSTM | Parametric | 1.000 | 923.64 | 923.64 |
| LSTM | Bootstrap | 0.875 | 733.03 | 786.26 |

### Exchange_Rate

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.958 | 0.06 | 0.06 |
| AutoARIMA | Parametric | 1.000 | 0.06 | 0.06 |
| AutoARIMA | Bootstrap | 0.958 | 0.07 | 0.07 |
| Theta | Conformal | 0.958 | 0.06 | 0.06 |
| Theta | Parametric | 1.000 | 0.06 | 0.06 |
| Theta | Bootstrap | 1.000 | 0.07 | 0.07 |
| LSTM | Conformal | 0.792 | 0.09 | 0.11 |
| LSTM | Parametric | 1.000 | 0.25 | 0.25 |
| LSTM | Bootstrap | 0.875 | 0.09 | 0.10 |

### Electricity_Demand

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.750 | 0.80 | 1.82 |
| AutoARIMA | Parametric | 0.917 | 1.18 | 1.51 |
| AutoARIMA | Bootstrap | 0.792 | 0.93 | 1.90 |
| Theta | Conformal | 0.542 | 0.79 | 3.85 |
| Theta | Parametric | 0.750 | 1.29 | 2.50 |
| Theta | Bootstrap | 0.458 | 1.00 | 4.72 |
| LSTM | Conformal | 0.917 | 1.43 | 1.79 |
| LSTM | Parametric | 0.875 | 1.24 | 1.99 |
| LSTM | Bootstrap | 0.292 | 0.90 | 4.70 |

### M4_Series

| Model | Method | Coverage | Width | Winkler |
|-------|--------|----------|-------|----------|
| AutoARIMA | Conformal | 0.625 | 58.00 | 87.90 |
| AutoARIMA | Parametric | 1.000 | 78.11 | 78.11 |
| AutoARIMA | Bootstrap | 0.625 | 68.56 | 126.47 |
| Theta | Conformal | 1.000 | 66.09 | 66.09 |
| Theta | Parametric | 1.000 | 77.58 | 77.58 |
| Theta | Bootstrap | 0.542 | 70.04 | 105.92 |
| LSTM | Conformal | 1.000 | 145.16 | 145.16 |
| LSTM | Parametric | 1.000 | 644.73 | 644.73 |
| LSTM | Bootstrap | 0.542 | 119.80 | 303.87 |

## Key Findings

- **Best coverage (closest to 90%):** Parametric
- **Best Winkler score:** Conformal
- **Narrowest intervals:** Bootstrap

## Observations

- Real-world datasets present different challenges than synthetic data
- Conformal methods should maintain coverage guarantees across different real-world scenarios
- Comparison with synthetic results reveals method robustness to real-world complexity
- Financial data (Gold, Stock, Exchange Rate) may show different patterns than demand data
