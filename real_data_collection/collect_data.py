#!/usr/bin/env python3
"""
Real-world time series data collection script
Collects 5 datasets for conformal prediction thesis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
from datetime import datetime
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Metadata storage
metadata = {}

print("="*80)
print("REAL-WORLD TIME SERIES DATA COLLECTION")
print("="*80)

# ============================================================================
# Dataset 1: Stock Price (S&P 500)
# ============================================================================
print("\n[1/5] Collecting S&P 500 stock prices...")

try:
    sp500 = yf.download('^GSPC', start='2015-01-01', end='2025-01-01', progress=False)

    # Extract closing prices - reset index to make timestamp a column
    sp500 = sp500.reset_index()
    df_stock = sp500[['Date', 'Close']].copy()
    df_stock.columns = ['timestamp', 'value']

    # Handle missing values
    df_stock = df_stock.dropna()

    # Save to CSV
    df_stock.to_csv('data/stock_prices.csv', index=False)

    # Metadata
    metadata['stock_prices'] = {
        'name': 'S&P 500 Index',
        'source': 'Yahoo Finance',
        'ticker': '^GSPC',
        'url': 'https://finance.yahoo.com/quote/%5EGSPC',
        'description': 'Daily closing prices of S&P 500 index',
        'characteristics': 'Non-stationary, volatile, GARCH effects, regime changes',
        'length': len(df_stock),
        'date_range': f"{df_stock['timestamp'].min()} to {df_stock['timestamp'].max()}",
        'mean': float(df_stock['value'].mean()),
        'std': float(df_stock['value'].std()),
        'min': float(df_stock['value'].min()),
        'max': float(df_stock['value'].max())
    }

    print(f"✓ Collected {len(df_stock)} observations")
    print(f"  Date range: {df_stock['timestamp'].min()} to {df_stock['timestamp'].max()}")
    print(f"  Mean: {df_stock['value'].mean():.2f}, Std: {df_stock['value'].std():.2f}")

except Exception as e:
    print(f"✗ Error collecting stock data: {e}")

# ============================================================================
# Dataset 2: Exchange Rate (USD to EUR)
# ============================================================================
print("\n[2/5] Collecting USD to EUR exchange rate...")

try:
    eurusd = yf.download('EURUSD=X', start='2015-01-01', end='2025-01-01', progress=False)

    # Extract closing prices - reset index to make timestamp a column
    eurusd = eurusd.reset_index()
    df_fx = eurusd[['Date', 'Close']].copy()
    df_fx.columns = ['timestamp', 'value']

    # Handle missing values
    df_fx = df_fx.ffill()

    # Save to CSV
    df_fx.to_csv('data/exchange_rate.csv', index=False)

    # Metadata
    metadata['exchange_rate'] = {
        'name': 'EUR/USD Exchange Rate',
        'source': 'Yahoo Finance',
        'ticker': 'EURUSD=X',
        'url': 'https://finance.yahoo.com/quote/EURUSD=X',
        'description': 'Daily EUR/USD exchange rate',
        'characteristics': 'Mean-reverting, heteroscedastic, policy-driven regime shifts',
        'length': len(df_fx),
        'date_range': f"{df_fx['timestamp'].min()} to {df_fx['timestamp'].max()}",
        'mean': float(df_fx['value'].mean()),
        'std': float(df_fx['value'].std()),
        'min': float(df_fx['value'].min()),
        'max': float(df_fx['value'].max())
    }

    print(f"✓ Collected {len(df_fx)} observations")
    print(f"  Date range: {df_fx['timestamp'].min()} to {df_fx['timestamp'].max()}")
    print(f"  Mean: {df_fx['value'].mean():.4f}, Std: {df_fx['value'].std():.4f}")

except Exception as e:
    print(f"✗ Error collecting exchange rate data: {e}")

# ============================================================================
# Dataset 3: Gold Prices
# ============================================================================
print("\n[3/5] Collecting gold prices...")

try:
    gold = yf.download('GC=F', start='2015-01-01', end='2025-01-01', progress=False)

    # Extract closing prices - reset index to make timestamp a column
    gold = gold.reset_index()
    df_gold = gold[['Date', 'Close']].copy()
    df_gold.columns = ['timestamp', 'value']

    # Handle missing values
    df_gold = df_gold.ffill()

    # Save to CSV
    df_gold.to_csv('data/gold_prices.csv', index=False)

    # Metadata
    metadata['gold_prices'] = {
        'name': 'Gold Futures Prices',
        'source': 'Yahoo Finance',
        'ticker': 'GC=F',
        'url': 'https://finance.yahoo.com/quote/GC=F',
        'description': 'Daily gold futures closing prices',
        'characteristics': 'Safe-haven asset, crisis-reactive, trend + volatility clustering',
        'length': len(df_gold),
        'date_range': f"{df_gold['timestamp'].min()} to {df_gold['timestamp'].max()}",
        'mean': float(df_gold['value'].mean()),
        'std': float(df_gold['value'].std()),
        'min': float(df_gold['value'].min()),
        'max': float(df_gold['value'].max())
    }

    print(f"✓ Collected {len(df_gold)} observations")
    print(f"  Date range: {df_gold['timestamp'].min()} to {df_gold['timestamp'].max()}")
    print(f"  Mean: {df_gold['value'].mean():.2f}, Std: {df_gold['value'].std():.2f}")

except Exception as e:
    print(f"✗ Error collecting gold data: {e}")

# ============================================================================
# Dataset 4: Electricity Demand (UCI Dataset)
# ============================================================================
print("\n[4/5] Collecting electricity demand data...")

try:
    # Download UCI household power consumption dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'

    print("  Downloading UCI electricity dataset (may take a moment)...")

    # Try to download and process
    import io
    import zipfile

    response = requests.get(url, timeout=60)

    if response.status_code == 200:
        # Extract zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open('household_power_consumption.txt') as f:
                df_elec = pd.read_csv(f, sep=';', parse_dates={'timestamp': ['Date', 'Time']},
                                     na_values=['?'])

        # Use Global_active_power column
        df_elec = df_elec[['timestamp', 'Global_active_power']].copy()
        df_elec.columns = ['timestamp', 'value']

        # Remove missing values
        df_elec = df_elec.dropna()

        # Aggregate to daily (too many hourly observations)
        df_elec['date'] = pd.to_datetime(df_elec['timestamp']).dt.date
        df_elec_daily = df_elec.groupby('date')['value'].mean().reset_index()
        df_elec_daily.columns = ['timestamp', 'value']

        # Take a subset for manageability
        df_elec_daily = df_elec_daily.iloc[:2000]

        # Save to CSV
        df_elec_daily.to_csv('data/electricity_demand.csv', index=False)

        # Metadata
        metadata['electricity_demand'] = {
            'name': 'Household Electric Power Consumption',
            'source': 'UCI Machine Learning Repository',
            'url': 'https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption',
            'description': 'Daily aggregated household electricity consumption (Global Active Power)',
            'characteristics': 'Strong seasonality, weather-dependent variance',
            'length': len(df_elec_daily),
            'date_range': f"{df_elec_daily['timestamp'].min()} to {df_elec_daily['timestamp'].max()}",
            'mean': float(df_elec_daily['value'].mean()),
            'std': float(df_elec_daily['value'].std()),
            'min': float(df_elec_daily['value'].min()),
            'max': float(df_elec_daily['value'].max())
        }

        print(f"✓ Collected {len(df_elec_daily)} observations (daily aggregated)")
        print(f"  Date range: {df_elec_daily['timestamp'].min()} to {df_elec_daily['timestamp'].max()}")
        print(f"  Mean: {df_elec_daily['value'].mean():.4f}, Std: {df_elec_daily['value'].std():.4f}")
    else:
        print(f"✗ Failed to download UCI dataset (status code: {response.status_code})")
        print("  Creating synthetic electricity-like data as fallback...")

        # Fallback: create synthetic seasonal data
        n = 2000
        t = np.arange(n)
        trend = 1.2 + 0.0001 * t
        seasonal_daily = 0.3 * np.sin(2 * np.pi * t / 7)  # weekly
        seasonal_yearly = 0.4 * np.sin(2 * np.pi * t / 365)  # yearly
        noise = np.random.normal(0, 0.1, n)

        value = trend + seasonal_daily + seasonal_yearly + noise

        df_elec_daily = pd.DataFrame({
            'timestamp': pd.date_range('2019-01-01', periods=n, freq='D'),
            'value': value
        })

        df_elec_daily.to_csv('data/electricity_demand.csv', index=False)

        metadata['electricity_demand'] = {
            'name': 'Synthetic Electricity-like Data (Fallback)',
            'source': 'Generated',
            'description': 'Synthetic data with seasonal patterns (fallback due to download issue)',
            'characteristics': 'Weekly and yearly seasonality',
            'length': len(df_elec_daily),
            'date_range': f"{df_elec_daily['timestamp'].min()} to {df_elec_daily['timestamp'].max()}",
            'mean': float(df_elec_daily['value'].mean()),
            'std': float(df_elec_daily['value'].std()),
            'min': float(df_elec_daily['value'].min()),
            'max': float(df_elec_daily['value'].max())
        }

        print(f"✓ Created fallback dataset: {len(df_elec_daily)} observations")

except Exception as e:
    print(f"✗ Error with electricity data: {e}")
    print("  Creating fallback synthetic data...")

    # Fallback
    n = 2000
    t = np.arange(n)
    trend = 1.2 + 0.0001 * t
    seasonal_daily = 0.3 * np.sin(2 * np.pi * t / 7)
    seasonal_yearly = 0.4 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 0.1, n)

    value = trend + seasonal_daily + seasonal_yearly + noise

    df_elec_daily = pd.DataFrame({
        'timestamp': pd.date_range('2019-01-01', periods=n, freq='D'),
        'value': value
    })

    df_elec_daily.to_csv('data/electricity_demand.csv', index=False)

    metadata['electricity_demand'] = {
        'name': 'Synthetic Electricity-like Data (Fallback)',
        'source': 'Generated',
        'description': 'Synthetic data with seasonal patterns',
        'characteristics': 'Weekly and yearly seasonality',
        'length': len(df_elec_daily),
        'mean': float(df_elec_daily['value'].mean()),
        'std': float(df_elec_daily['value'].std())
    }

# ============================================================================
# Dataset 5: M4 Competition Dataset
# ============================================================================
print("\n[5/5] Collecting M4 competition dataset...")

try:
    # Try to download a daily M4 series from GitHub
    url = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Daily-train.csv'

    print("  Downloading M4 daily dataset...")

    df_m4_all = pd.read_csv(url, nrows=10)  # Read just first few to see structure

    # Now read full and select one series
    df_m4_all = pd.read_csv(url)

    # Take the first series with enough data
    series_id = None
    for idx, row in df_m4_all.iterrows():
        values = row.iloc[1:].dropna().values  # Skip the ID column
        if len(values) >= 1000:
            series_id = row.iloc[0]
            series_values = values
            break

    if series_id:
        # Create time series
        df_m4 = pd.DataFrame({
            'timestamp': pd.date_range(start='2010-01-01', periods=len(series_values), freq='D'),
            'value': series_values
        })

        df_m4.to_csv('data/m4_series.csv', index=False)

        metadata['m4_series'] = {
            'name': f'M4 Competition Series {series_id}',
            'source': 'M4 Competition',
            'url': 'https://github.com/Mcompetitions/M4-methods',
            'description': f'M4 competition daily series (ID: {series_id})',
            'characteristics': 'Benchmark dataset for forecasting competitions',
            'length': len(df_m4),
            'date_range': f"{df_m4['timestamp'].min()} to {df_m4['timestamp'].max()}",
            'mean': float(df_m4['value'].mean()),
            'std': float(df_m4['value'].std()),
            'min': float(df_m4['value'].min()),
            'max': float(df_m4['value'].max())
        }

        print(f"✓ Collected series {series_id}: {len(df_m4)} observations")
        print(f"  Date range: {df_m4['timestamp'].min()} to {df_m4['timestamp'].max()}")
        print(f"  Mean: {df_m4['value'].mean():.2f}, Std: {df_m4['value'].std():.2f}")
    else:
        raise Exception("No suitable M4 series found")

except Exception as e:
    print(f"✗ Error collecting M4 data: {e}")
    print("  Creating synthetic benchmark-like data as fallback...")

    # Fallback: AR(1) with trend
    n = 1500
    np.random.seed(42)
    trend = 100 + 0.05 * np.arange(n)
    y = np.zeros(n)
    y[0] = trend[0]
    for t in range(1, n):
        y[t] = trend[t] + 0.7 * (y[t-1] - trend[t-1]) + np.random.normal(0, 5)

    df_m4 = pd.DataFrame({
        'timestamp': pd.date_range('2010-01-01', periods=n, freq='D'),
        'value': y
    })

    df_m4.to_csv('data/m4_series.csv', index=False)

    metadata['m4_series'] = {
        'name': 'Synthetic Benchmark Data (Fallback)',
        'source': 'Generated',
        'description': 'AR(1) process with trend (fallback)',
        'characteristics': 'Trend + autocorrelation',
        'length': len(df_m4),
        'mean': float(df_m4['value'].mean()),
        'std': float(df_m4['value'].std())
    }

    print(f"✓ Created fallback dataset: {len(df_m4)} observations")

# ============================================================================
# Save Metadata
# ============================================================================
print("\n" + "="*80)
print("Saving metadata...")

with open('data/datasets_info.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print("✓ Metadata saved to data/datasets_info.json")

# ============================================================================
# Create Visualization
# ============================================================================
print("\nCreating visualization...")

fig, axes = plt.subplots(5, 1, figsize=(14, 16))

datasets_to_plot = [
    ('data/stock_prices.csv', 'S&P 500 Stock Prices', 'blue'),
    ('data/exchange_rate.csv', 'EUR/USD Exchange Rate', 'green'),
    ('data/gold_prices.csv', 'Gold Futures Prices', 'gold'),
    ('data/electricity_demand.csv', 'Electricity Demand', 'red'),
    ('data/m4_series.csv', 'M4 Benchmark Series', 'purple')
]

for idx, (file, title, color) in enumerate(datasets_to_plot):
    try:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        axes[idx].plot(df['timestamp'], df['value'], linewidth=0.8, color=color)
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Date', fontsize=9)
        axes[idx].set_ylabel('Value', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(labelsize=8)
    except Exception as e:
        axes[idx].text(0.5, 0.5, f'Error loading {title}',
                      ha='center', va='center', transform=axes[idx].transAxes)

plt.suptitle('Real-World Time Series Datasets for Conformal Prediction',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/all_real_datasets.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to data/all_real_datasets.png")

print("\n" + "="*80)
print("DATA COLLECTION COMPLETE!")
print("="*80)
print(f"\nCollected {len(metadata)} datasets:")
for key, info in metadata.items():
    print(f"  - {info['name']}: {info['length']} observations")

print("\nAll files saved to: real_data_collection/data/")
