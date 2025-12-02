# Prediction Interval Formulas

This document describes the mathematical formulas for the three prediction interval methods used in the experiments.

---

## 1. Parametric Intervals

**Assumption:** Forecast errors follow a known distribution (typically Gaussian).

### Formula

For a forecast at time $t+h$ with point prediction $\hat{y}_{t+h}$:

$$
\text{PI}_{1-\alpha}(t+h) = \left[\hat{y}_{t+h} - z_{\alpha/2} \cdot \hat{\sigma}_{t+h}, \quad \hat{y}_{t+h} + z_{\alpha/2} \cdot \hat{\sigma}_{t+h}\right]
$$

Where:
- $\hat{y}_{t+h}$ = point forecast for horizon $h$
- $z_{\alpha/2}$ = quantile of the standard normal distribution (e.g., $z_{0.05} = 1.645$ for 90% coverage)
- $\hat{\sigma}_{t+h}$ = estimated forecast standard deviation at horizon $h$

### For Different Models

**AutoARIMA & Theta:**
$$
\hat{\sigma}_{t+h}^2 = \sigma^2 \sum_{i=0}^{h-1} \psi_i^2
$$

Where $\psi_i$ are the MA($\infty$) coefficients from the ARIMA representation, and $\sigma^2$ is the innovation variance.

**LSTM (with Quantile Regression):**

The model directly predicts quantiles using MQLoss:

$$
\begin{align}
\hat{y}_{t+h}^{(0.05)} &= f_\theta(x_t; q=0.05) \\
\hat{y}_{t+h}^{(0.95)} &= f_\theta(x_t; q=0.95)
\end{align}
$$

Then:
$$
\text{PI}_{90\%}(t+h) = \left[\hat{y}_{t+h}^{(0.05)}, \quad \hat{y}_{t+h}^{(0.95)}\right]
$$

### Strengths & Weaknesses

✅ **Strengths:**
- Theoretically grounded when assumptions hold
- Computationally efficient
- Provides smooth intervals across horizons

❌ **Weaknesses:**
- Requires distributional assumptions (typically Gaussianity)
- May undercover when errors are heavy-tailed or heteroscedastic
- Model misspecification affects both point and interval estimates

---

## 2. Conformal Prediction Intervals

**Assumption:** Exchangeability of data (weaker than i.i.d., distribution-free).

### Method Overview

Conformal prediction uses a calibration set to compute a conformity score, then applies this score to new predictions.

### Algorithm

**Step 1: Split data**
- Training set: $\mathcal{D}_{\text{train}} = \{(x_1, y_1), \ldots, (x_n, y_n)\}$
- Calibration set: $\mathcal{D}_{\text{cal}} = \{(x_{n+1}, y_{n+1}), \ldots, (x_{n+m}, y_{n+m})\}$

**Step 2: Train model on training set**
$$
\hat{f} = \text{train}(\mathcal{D}_{\text{train}})
$$

**Step 3: Compute conformity scores on calibration set**

For each calibration point $i \in \{n+1, \ldots, n+m\}$:
$$
s_i = |y_i - \hat{f}(x_i)|
$$

**Step 4: Compute conformal quantile**

For desired coverage level $(1-\alpha)$:
$$
q_{\alpha} = \text{Quantile}\left(\{s_{n+1}, \ldots, s_{n+m}\}, \frac{\lceil (m+1)(1-\alpha) \rceil}{m}\right)
$$

**Step 5: Construct prediction interval**

For new prediction at $x_{t+h}$:
$$
\text{PI}_{1-\alpha}(t+h) = \left[\hat{f}(x_{t+h}) - q_{\alpha}, \quad \hat{f}(x_{t+h}) + q_{\alpha}\right]
$$

### Implementation in Experiments

**For AutoARIMA & Theta (using ConformalIntervals):**

Uses time series cross-validation with $n_{\text{windows}}$ splits:

```python
conformal = ConformalIntervals(h=horizon, n_windows=n_windows)
```

Internally:
1. Creates $n_{\text{windows}}$ train/calibration splits
2. For each split, computes forecast errors on calibration window
3. Pools all calibration errors: $\{e_1, e_2, \ldots, e_K\}$ where $K = n_{\text{windows}} \times h$
4. Computes $q_{0.9} = \text{Quantile}(\{|e_1|, |e_2|, \ldots, |e_K|\}, 0.9)$
5. Intervals: $[\hat{y}_{t+h} - q_{0.9}, \hat{y}_{t+h} + q_{0.9}]$

**For LSTM (custom implementation):**

Uses cross-validation to get residuals:

```python
cv_df = nf.cross_validation(df=train_df, n_windows=5, step_size=horizon)
residuals = |cv_df['y'] - cv_df['LSTM']|
q = np.quantile(residuals, 0.9)
```

Then applies symmetric intervals:
$$
\text{PI}_{90\%}(t+h) = [\hat{y}_{t+h} - q, \hat{y}_{t+h} + q]
$$

### Theoretical Guarantee

Under exchangeability, conformal prediction provides **finite-sample coverage guarantee**:

$$
\mathbb{P}\left(Y_{t+h} \in \text{PI}_{1-\alpha}(t+h)\right) \geq 1 - \alpha
$$

This holds **regardless of the model quality** or **data distribution**.

### Strengths & Weaknesses

✅ **Strengths:**
- Distribution-free (no parametric assumptions)
- Finite-sample coverage guarantees
- Works with any base forecasting model
- Adapts to model errors automatically

❌ **Weaknesses:**
- Requires calibration data (reduces training set size)
- Assumes exchangeability (may not hold with concept drift)
- Symmetric intervals (doesn't capture asymmetry in errors)
- Width depends on model accuracy (poor models → wide intervals)

---

## 3. Bootstrap Intervals

**Assumption:** Residuals are representative of future errors.

### Method Overview

Bootstrap resamples historical residuals to construct empirical prediction intervals.

### Algorithm

**Step 1: Fit model and compute residuals**

Train model on $\mathcal{D}_{\text{train}}$:
$$
\hat{f} = \text{train}(\mathcal{D}_{\text{train}})
$$

Compute residuals (preferably from cross-validation to avoid overfitting):
$$
\hat{\varepsilon}_i = y_i - \hat{f}(x_i), \quad i = 1, \ldots, n
$$

**Step 2: Generate point forecast**
$$
\hat{y}_{t+h} = \hat{f}(x_{t+h})
$$

**Step 3: Bootstrap resampling**

For $b = 1, \ldots, B$ (typically $B = 100$ or more):

1. Sample residuals with replacement:
   $$
   \varepsilon_1^{(b)}, \varepsilon_2^{(b)}, \ldots, \varepsilon_h^{(b)} \sim \text{Resample}(\{\hat{\varepsilon}_1, \ldots, \hat{\varepsilon}_n\})
   $$

2. Generate bootstrap forecast path:
   $$
   \hat{y}_{t+j}^{(b)} = \hat{y}_{t+j} + \varepsilon_j^{(b)}, \quad j = 1, \ldots, h
   $$

**Step 4: Compute interval from bootstrap distribution**

For horizon $h$:
$$
\text{PI}_{1-\alpha}(t+h) = \left[\text{Quantile}\left(\{\hat{y}_{t+h}^{(1)}, \ldots, \hat{y}_{t+h}^{(B)}\}, \frac{\alpha}{2}\right), \quad \text{Quantile}\left(\{\hat{y}_{t+h}^{(1)}, \ldots, \hat{y}_{t+h}^{(B)}\}, 1-\frac{\alpha}{2}\right)\right]
$$

For 90% coverage:
$$
\text{PI}_{90\%}(t+h) = \left[\text{Quantile}_5(\{\hat{y}_{t+h}^{(b)}\}), \quad \text{Quantile}_{95}(\{\hat{y}_{t+h}^{(b)}\})\right]
$$

### Implementation in Experiments

```python
# Get residuals from cross-validation
cv_df = model.cross_validation(df=train_df, h=horizon, n_windows=5)
residuals = (cv_df['y'] - cv_df['model']).values

# Point forecast
point = model.forecast(df=train_df, h=horizon)

# Bootstrap
boot_forecasts = []
for _ in range(n_boots):  # n_boots = 100
    boot_res = np.random.choice(residuals, size=horizon, replace=True)
    boot_forecasts.append(point + boot_res)

# Intervals
boot_forecasts = np.array(boot_forecasts)  # Shape: (100, horizon)
lower = np.percentile(boot_forecasts, 5, axis=0)
upper = np.percentile(boot_forecasts, 95, axis=0)
```

### Strengths & Weaknesses

✅ **Strengths:**
- Distribution-free (uses empirical distribution of residuals)
- Can capture asymmetry in residuals
- Flexible and model-agnostic
- Can account for parameter uncertainty with model refitting

❌ **Weaknesses:**
- Assumes residuals are i.i.d. or exchangeable
- Computationally expensive (requires $B$ resamples)
- May undercover if residuals are heteroscedastic
- No finite-sample coverage guarantees (asymptotic only)
- Width depends on quality of residuals from historical fit

---

## Comparison Table

| Method | Distributional Assumptions | Coverage Guarantee | Computational Cost | Captures Asymmetry |
|--------|---------------------------|-------------------|-------------------|-------------------|
| **Parametric** | Strong (Gaussian errors) | Asymptotic (if assumptions hold) | Low | No (symmetric) |
| **Conformal** | None (exchangeability only) | **Finite-sample** | Medium | No (symmetric) |
| **Bootstrap** | Weak (i.i.d. residuals) | Asymptotic | High ($B$ resamples) | **Yes** |

---

## Evaluation Metrics

All intervals are evaluated using:

### 1. Coverage
$$
\text{Coverage} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{y_t \in [\hat{L}_t, \hat{U}_t]\}
$$

**Target:** 0.90 (90% coverage)

### 2. Mean Interval Width
$$
\text{Width} = \frac{1}{T} \sum_{t=1}^{T} (\hat{U}_t - \hat{L}_t)
$$

**Preference:** Smaller is better (sharper intervals)

### 3. Winkler Score
$$
\text{Winkler}(y_t, \hat{L}_t, \hat{U}_t) = (\hat{U}_t - \hat{L}_t) + \frac{2}{\alpha}(\hat{L}_t - y_t)\mathbb{1}\{y_t < \hat{L}_t\} + \frac{2}{\alpha}(y_t - \hat{U}_t)\mathbb{1}\{y_t > \hat{U}_t\}
$$

Where $\alpha = 0.1$ for 90% intervals.

**Interpretation:**
- Rewards narrow intervals (first term)
- Penalizes under-coverage (second and third terms)
- **Lower is better**

**Average Winkler Score:**
$$
\overline{\text{Winkler}} = \frac{1}{T} \sum_{t=1}^{T} \text{Winkler}(y_t, \hat{L}_t, \hat{U}_t)
$$

---

## References

1. **Conformal Prediction:**
   - Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
   - Angelopoulos, A. N., & Bates, S. (2021). "A gentle introduction to conformal prediction and distribution-free uncertainty quantification." arXiv:2107.07511.

2. **Bootstrap Intervals:**
   - Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC press.
   - Stine, R. A. (1987). "Estimating properties of autoregressive forecasts." *Journal of the American Statistical Association*, 82(400), 1072-1078.

3. **Parametric Intervals:**
   - Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.
   - Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

4. **Winkler Score:**
   - Winkler, R. L. (1972). "A decision-theoretic approach to interval estimation." *Journal of the American Statistical Association*, 67(337), 187-191.
