"""
Feature Engineering Module for Forex RL Trading Bot
Implements comprehensive feature extraction including technical indicators,
currency strength, fractals, and temporal features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Major currencies for strength calculation
MAJORS = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]


# Add vectorized percentile helper
def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    FAST rolling percentile using manual loop with pre-allocated array.
    Computes percentile of current value within trailing window.
    """
    s = series.values.astype(np.float64)
    n = len(s)
    result = np.zeros(n, dtype=np.float64)
    
    # Manual loop - faster than pandas .apply()
    for i in range(window-1, n):
        window_data = s[i-window+1:i+1]
        current_value = s[i]
        # Count how many values in window are <= current value
        rank = np.sum(window_data <= current_value) - 1
        percentile = rank / (window - 1 + 1e-9)
        result[i] = np.clip(percentile, 0.0, 1.0)
    
    return pd.Series(result, index=series.index)


def _rolling_lr_slope(y: pd.Series, window: int) -> pd.Series:
    """
    FAST vectorized rolling linear regression slope using manual loop.
    Much faster than pandas .apply() due to pre-computed constants.
    
    Args:
        y: Series to compute slope on
        window: Rolling window size
        
    Returns:
        Series with slope values
    """
    W = int(window)
    if W < 2:
        return pd.Series(0.0, index=y.index)
    
    # Convert to numpy array
    arr = y.values.astype(np.float64)
    n = len(arr)
    
    # Pre-compute x values and their statistics (constant for all windows)
    x = np.arange(W, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = (x_centered ** 2).sum()
    
    if x_var < 1e-12:
        return pd.Series(0.0, index=y.index)
    
    # Initialize result array
    slopes = np.zeros(n, dtype=np.float64)
    
    # Manual loop - still much faster than pandas .apply()
    # because we pre-computed x statistics
    for i in range(W-1, n):
        window_data = arr[i-W+1:i+1]
        y_mean = window_data.mean()
        y_centered = window_data - y_mean
        cov = (x_centered * y_centered).sum()
        slopes[i] = cov / x_var
    
    return pd.Series(slopes, index=y.index)


# =========================================================================
# Panoptic Indicator Helper Functions
# =========================================================================

def _rolling_spearman(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling Spearman rank correlation of price vs time (trend linearity).
    Mirrors the Panoptic Correlation Gauge.
    Output: -1 (perfect downtrend) to +1 (perfect uptrend), 0 = no linear trend.
    """
    arr = series.values.astype(np.float64)
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    W = int(window)
    if W < 3:
        return pd.Series(result, index=series.index)
    
    for i in range(W - 1, n):
        window_data = arr[i - W + 1: i + 1]
        # Rank the price values
        order = np.argsort(np.argsort(window_data)).astype(np.float64) + 1.0
        # Time ranks are simply 1..W
        time_ranks = np.arange(1, W + 1, dtype=np.float64)
        # Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
        d = order - time_ranks
        sum_d2 = (d * d).sum()
        result[i] = 1.0 - (6.0 * sum_d2) / (W * (W * W - 1.0))
    
    return pd.Series(result, index=series.index)


def _rolling_mad(series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling median and MAD (Median Absolute Deviation).
    Returns (median, mad) pair.
    """
    arr = series.values.astype(np.float64)
    n = len(arr)
    medians = np.zeros(n, dtype=np.float64)
    mads = np.zeros(n, dtype=np.float64)
    W = int(window)
    
    for i in range(W - 1, n):
        window_data = arr[i - W + 1: i + 1]
        med = np.median(window_data)
        medians[i] = med
        mads[i] = np.median(np.abs(window_data - med))
    
    return (pd.Series(medians, index=series.index),
            pd.Series(mads, index=series.index))


def _tanh_compress(x: np.ndarray) -> np.ndarray:
    """Tanh compression, clipping input to avoid overflow."""
    x_safe = np.clip(x, -10.0, 10.0)
    return np.tanh(x_safe)


def _compute_cci(close: pd.Series, high: pd.Series, low: pd.Series,
                 period: int) -> pd.Series:
    """Commodity Channel Index (CCI)."""
    tp = (high + low + close) / 3.0
    tp_sma = tp.rolling(period, min_periods=1).mean()
    tp_mad = tp.rolling(period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (tp - tp_sma) / (0.015 * tp_mad + 1e-10)
    return cci


def _compute_ppi(close: pd.Series, high: pd.Series, low: pd.Series,
                 period: int) -> pd.Series:
    """
    Price Position Index — percentile rank of current close within
    trailing HLC distribution (3x sampling per bar).
    Mirrors the Panoptic Price Density indicator.
    Output: 0-100.
    """
    c = close.values.astype(np.float64)
    h = high.values.astype(np.float64)
    l = low.values.astype(np.float64)
    n = len(c)
    result = np.full(n, 50.0, dtype=np.float64)
    W = int(period)
    
    for i in range(W, n):
        # Build HLC distribution from trailing window (excluding current bar)
        dist = np.empty(W * 3, dtype=np.float64)
        for j in range(W):
            idx = i - W + j
            dist[j * 3] = h[idx]
            dist[j * 3 + 1] = l[idx]
            dist[j * 3 + 2] = c[idx]
        dist.sort()
        # Rank of current close
        rank = np.searchsorted(dist, c[i], side='right')
        result[i] = (rank / len(dist)) * 100.0
    
    return pd.Series(result, index=close.index)


def compute_currency_strengths(pair_dfs: Dict[str, pd.DataFrame],
                               currencies: List[str] = None,
                               window: int = 24,
                               lags: int = 3,
                               use_ema_strength: bool = False,
                               ema_span: int = 12) -> pd.DataFrame:
    """
    Compute currency strength index across multiple currency pairs.

    Args:
        pair_dfs: Dictionary of {pair: dataframe} with datetime index and 'close' column
        currencies: List of currencies to compute strengths for (defaults to MAJORS)
        window: Rolling window for strength calculation
        lags: Number of lag features to create
        use_ema_strength: If True, apply EMA smoothing before z-score normalization
        ema_span: EMA span for smoothing (default 12)

    Returns:
        DataFrame with currency strength features including lags
    """
    if currencies is None:
        currencies = MAJORS
    # 1) Calculate hourly log returns for each pair
    rets = {}
    for pair, df in pair_dfs.items():
        if len(pair) >= 6:  # Ensure valid pair format
            r = np.log(df['close']).diff()
            rets[pair] = r.rename(pair)

    if not rets:
        return pd.DataFrame()

    R = pd.concat(rets.values(), axis=1).dropna()

    # 2) Calculate signed returns per currency (base +, quote -)
    def signed_ret(curr):
        cols = []
        for pair in R.columns:
            if len(pair) >= 6:
                base, quote = pair[:3], pair[3:6]  # Handle pairs like EURUSD
                s = +1 if curr == base else (-1 if curr == quote else 0)
                if s != 0:
                    cols.append(s * R[pair])

        if not cols:
            return pd.Series(index=R.index, dtype=float, name=f"strength_{curr}")

        return pd.concat(cols, axis=1).mean(axis=1)

    # 3) Calculate strength for each specified currency
    strengths = {}
    for curr in currencies:
        sr = signed_ret(curr)
        if len(sr) > 0 and not sr.isna().all():
            # Rolling mean to denoise (or EMA if enabled)
            if use_ema_strength:
                m = sr.ewm(span=ema_span, adjust=False, min_periods=1).mean()
            else:
                m = sr.rolling(window, min_periods=1).mean()
            # z-score normalization
            z = (m - m.rolling(window * 4, min_periods=window).mean()) / \
                (m.rolling(window * 4, min_periods=window).std(ddof=1) + 1e-9)
            strengths[curr] = z.rename(f"strength_{curr}")

    if not strengths:
        return pd.DataFrame()

    S = pd.concat(strengths.values(), axis=1)

    # 4) Add lag features (configurable number)
    for curr in list(strengths.keys()):
        for lag in range(1, lags + 1):
            S[f"strength_{curr}_lag{lag}"] = S[f"strength_{curr}"].shift(lag)

    return S.dropna()


class FeatureEngineer:
    """
    Comprehensive feature engineering for Forex trading.
    """

    def __init__(self,
                 short_window: int = 5,
                 medium_window: int = 20,
                 long_window: int = 50,
                 atr_period: int = 14,
                 rsi_period: int = 14,
                 lr_window: int = 10,
                 fractal_window: int = 5):
        """
        Initialize feature engineer with configurable windows.

        Args:
            short_window: Window for short-term percentile calculation
            medium_window: Window for medium-term percentile calculation
            long_window: Window for long-term percentile calculation
            atr_period: Period for ATR calculation
            rsi_period: Period for RSI calculation
            lr_window: Window for linear regression slope
            fractal_window: Window for fractal detection (typically 5)
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.lr_window = lr_window
        self.fractal_window = fractal_window

    def compute_all_features(self, df: pd.DataFrame,
                            currency_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute all features for a given OHLC dataframe.

        Args:
            df: DataFrame with OHLC data and timestamp
            currency_data: Optional dict of {pair: dataframe} for currency strength

        Returns:
            DataFrame with all computed features
        """
        df = df.copy()

        # Basic technical indicators
        df['atr'] = self.compute_atr(df)
        df['rsi'] = self.compute_rsi(df)

        # Percentile features (vectorized)
        df['percentile_short'] = rolling_percentile(df['close'], self.short_window)
        df['percentile_medium'] = rolling_percentile(df['close'], self.medium_window)
        df['percentile_long'] = rolling_percentile(df['close'], self.long_window)

        # Currency strength features (if multi-pair data provided)
        if currency_data:
            try:
                strength_df = compute_currency_strengths(currency_data, window=24)
                if not strength_df.empty:
                    # Ensure indices match for join
                    strength_df = strength_df.reindex(df.index, method='ffill')
                    df = df.join(strength_df, how='left')
            except Exception as e:
                # If currency strength fails, fill with zeros
                print(f"Warning: Currency strength computation failed: {e}")
                for curr in ['EUR', 'USD']:
                    df[f'strength_{curr}'] = 0.0
                    df[f'strength_{curr}_lag1'] = 0.0
                    df[f'strength_{curr}_lag2'] = 0.0
                    df[f'strength_{curr}_lag3'] = 0.0

        # Temporal features
        df = self.add_temporal_features(df)
        df = self.add_cyclical_time(df)
        # Drop raw time columns in favor of cyclical encoding
        for col in ['hour_of_day', 'day_of_week', 'day_of_year']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Leak-proof fractals (strictly past data, no future peeking)
        df = self._compute_fractals_safe(df, w=self.fractal_window)

        # Linear regression slope
        df['lr_slope'] = self.compute_lr_slope(df)
        
        # PATCH 8: Meta-features for regime detection (account-invariant)
        df = self.add_regime_features(df)

        # Panoptic indicator features (from optimized Pine Script indicators)
        df = self.add_panoptic_features(df)

        # Fill NaN values
        df = df.ffill().bfill().fillna(0)

        return df

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Average True Range (ATR).

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()

        return atr

    def compute_rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with RSI values
        """
        close = df['close']
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def compute_percentile(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        DEPRECATED: Use rolling_percentile() function instead for better performance.
        Compute percentile of current price against last N HLC prices.

        Args:
            df: DataFrame with OHLC data
            window: Lookback window

        Returns:
            Series with percentile values (0-100)
        """
        # Use the fast version instead
        return rolling_percentile(df['close'], window) * 100.0  # Convert to 0-100 scale

    def compute_currency_strength(self, currency_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute currency strength across multiple pairs.

        The strength of each currency is calculated by:
        1. Computing log returns for all pairs containing that currency
        2. Adding returns if currency is base, subtracting if quote
        3. Averaging all adjusted returns
        4. Normalizing via z-score

        Args:
            currency_data: Dict of {pair_name: df} with OHLC data

        Returns:
            DataFrame with currency strength features and lagged versions
        """
        # Extract all unique currencies from pair names
        currencies = set()
        for pair in currency_data.keys():
            if len(pair) == 6:  # Standard format like EURUSD
                currencies.add(pair[:3])
                currencies.add(pair[3:6])

        currencies = sorted(list(currencies))

        # Align all dataframes to common index
        common_index = None
        for df in currency_data.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        # Calculate log returns for each pair
        returns_data = {}
        for pair, df in currency_data.items():
            df = df.loc[common_index]
            returns_data[pair] = np.log(df['close'] / df['close'].shift(1))

        # Calculate strength for each currency
        strength_df = pd.DataFrame(index=common_index)

        for currency in currencies:
            currency_returns = []

            for pair, returns in returns_data.items():
                if len(pair) == 6:
                    base = pair[:3]
                    quote = pair[3:6]

                    if base == currency:
                        currency_returns.append(returns)
                    elif quote == currency:
                        currency_returns.append(-returns)

            if currency_returns:
                # Average all returns for this currency
                avg_return = pd.concat(currency_returns, axis=1).mean(axis=1)

                # Normalize via z-score
                mean = avg_return.rolling(window=20, min_periods=1).mean()
                std = avg_return.rolling(window=20, min_periods=1).std()
                strength = (avg_return - mean) / (std + 1e-10)

                strength_df[f'strength_{currency}'] = strength

                # Add lagged versions
                strength_df[f'strength_{currency}_lag1'] = strength.shift(1)
                strength_df[f'strength_{currency}_lag2'] = strength.shift(2)
                strength_df[f'strength_{currency}_lag3'] = strength.shift(3)

        return strength_df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features: time of day, day of week, day of year.

        Args:
            df: DataFrame with 'time' column

        Returns:
            DataFrame with added temporal features
        """
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

            # Time of day (0-23 hours, normalized to 0-1)
            df['hour_of_day'] = df['time'].dt.hour / 23.0

            # Day of week (0-6, normalized to 0-1)
            df['day_of_week'] = df['time'].dt.dayofweek / 6.0

            # Day of year (0-365, normalized to 0-1)
            df['day_of_year'] = df['time'].dt.dayofyear / 365.0
        else:
            # If no time column, use index if it's datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df['hour_of_day'] = df.index.hour / 23.0
                df['day_of_week'] = df.index.dayofweek / 6.0
                df['day_of_year'] = df.index.dayofyear / 365.0
            else:
                df['hour_of_day'] = 0.5
                df['day_of_week'] = 0.5
                df['day_of_year'] = 0.5

        return df
    
    def add_cyclical_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time features (sin/cos encoding for wrapping).
        This helps learning by encoding temporal continuity (e.g., hour 23 is near hour 0).
        
        Args:
            df: DataFrame with datetime index or 'time' column
            
        Returns:
            DataFrame with cyclical time features added
        """
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        h = idx.hour.to_numpy()
        dow = idx.dayofweek.to_numpy()
        doy = idx.dayofyear.to_numpy()
        
        # Cyclical encoding with sin/cos
        df['hour_sin'] = np.sin(2*np.pi*h/24)
        df['hour_cos'] = np.cos(2*np.pi*h/24)
        df['dow_sin'] = np.sin(2*np.pi*dow/7)
        df['dow_cos'] = np.cos(2*np.pi*dow/7)
        df['doy_sin'] = np.sin(2*np.pi*doy/365)
        df['doy_cos'] = np.cos(2*np.pi*doy/365)
        
        return df

    def compute_top_fractal(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the most recent top fractal above current price.

        A top fractal is a high that is higher than N bars before and after it.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with top fractal values
        """
        n = self.fractal_window // 2
        fractals = []

        for i in range(len(df)):
            fractal_value = np.nan
            current_price = df.iloc[i]['close']

            # Look back for fractals
            for j in range(i - 1, max(0, i - 100), -1):  # Look back up to 100 bars
                if j < n or j >= len(df) - n:
                    continue

                # Check if this is a fractal
                high = df.iloc[j]['high']
                is_fractal = True

                for k in range(j - n, j + n + 1):
                    if k != j and df.iloc[k]['high'] >= high:
                        is_fractal = False
                        break

                if is_fractal and high > current_price:
                    fractal_value = high
                    break

            fractals.append(fractal_value)

        return pd.Series(fractals, index=df.index)

    def compute_bottom_fractal(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the most recent bottom fractal below current price.

        A bottom fractal is a low that is lower than N bars before and after it.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with bottom fractal values
        """
        n = self.fractal_window // 2
        fractals = []

        for i in range(len(df)):
            fractal_value = np.nan
            current_price = df.iloc[i]['close']

            # Look back for fractals
            for j in range(i - 1, max(0, i - 100), -1):  # Look back up to 100 bars
                if j < n or j >= len(df) - n:
                    continue

                # Check if this is a fractal
                low = df.iloc[j]['low']
                is_fractal = True

                for k in range(j - n, j + n + 1):
                    if k != j and df.iloc[k]['low'] <= low:
                        is_fractal = False
                        break

                if is_fractal and low < current_price:
                    fractal_value = low
                    break

            fractals.append(fractal_value)

        return pd.Series(fractals, index=df.index)

    def compute_lr_slope(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute linear regression slope of HLC prices over window.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with slope values
        """
        # slope of HLC average over lr_window
        hlc = (df['high'] + df['low'] + df['close']) / 3.0
        return _rolling_lr_slope(hlc.astype(float), self.lr_window)

    def get_feature_names(self, include_currency_strength: bool = True,
                         currencies: List[str] = None) -> List[str]:
        """
        Get list of all feature names.

        Args:
            include_currency_strength: Whether to include currency strength features
            currencies: List of currency codes if including strength features

        Returns:
            List of feature names
        """
        features = [
            'open', 'high', 'low', 'close',
            'atr', 'rsi',
            'percentile_short', 'percentile_medium', 'percentile_long',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',  # Cyclical time
            'top_fractal_confirmed', 'bottom_fractal_confirmed',  # Use CONFIRMED fractals (causal)
            'lr_slope',
            # Panoptic indicators
            'spearman_corr_8', 'spearman_corr_20', 'corr_velocity',
            'slope_velocity', 'slope_acceleration', 'slope_quiet_zone',
            'cci_norm', 'divergence_bull', 'divergence_bear',
            'ppi_raw', 'ppi_fast', 'ppi_slow',
            'dist_nearest_sr', 'sr_zone_position',
            # PATCH 8: Regime meta-features
            'realized_vol_24h_z', 'realized_vol_96h_z',
            'trend_24h', 'trend_96h', 'is_trending'
        ]

        if include_currency_strength:
            # Default to EUR/USD for primary pair trading
            if currencies is None:
                currencies = ['EUR', 'USD']

            for currency in currencies:
                features.extend([
                    f'strength_{currency}',
                    f'strength_{currency}_lag1',
                    f'strength_{currency}_lag2',
                    f'strength_{currency}_lag3'
                ])

        return features

    def _compute_fractals_safe(self, df: pd.DataFrame, w: int = 5) -> pd.DataFrame:
        """
        Compute fractals using strictly past data (no future peeking).
        At time t, only uses data from [t-(w-1)...t], ensuring causality.
        
        Args:
            df: DataFrame with OHLC data
            w: Window size (default 5)
            
        Returns:
            DataFrame with leak-proof confirmed fractals
        """
        H = df['high'].values
        L = df['low'].values
        top = np.full(len(df), np.nan, dtype=np.float64)
        bottom = np.full(len(df), np.nan, dtype=np.float64)
        k = w // 2  # Center offset (e.g., 2 for w=5)
        
        # First valid index is w-1 (need w bars of history)
        for t in range(w - 1, len(df)):
            # Strictly trailing window [t-(w-1)...t]
            win_h = H[t - (w - 1):t + 1]
            win_l = L[t - (w - 1):t + 1]
            
            # Center index is k bars in the past
            c = t - k
            if c >= 0:
                if H[c] == win_h.max():
                    top[t] = H[c]
                if L[c] == win_l.min():
                    bottom[t] = L[c]
        
        # Fill forward and add to dataframe
        df['top_fractal_confirmed'] = pd.Series(top, index=df.index).ffill()
        df['bottom_fractal_confirmed'] = pd.Series(bottom, index=df.index).ffill()
        
        return df

    def add_panoptic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features derived from Panoptic indicator suite.
        All features are single-series (no external data needed) and causal.
        
        Indicators implemented:
        1. Correlation Gauge  — Spearman trend linearity + velocity
        2. Slope Accelerator  — MAD-normalized slope velocity + acceleration
        3. Divergence Engine   — CCI + divergence detection
        4. Price Density (PPI) — HLC distribution percentile
        5. Structure Engine    — K-means S/R clusters + distance
        """
        close = df['close'].astype(np.float64)
        high = df['high'].astype(np.float64)
        low = df['low'].astype(np.float64)
        
        # === 1. CORRELATION GAUGE ===
        # Spearman rank correlation of price vs time (trend linearity)
        corr_8 = _rolling_spearman(close, 8)
        corr_20 = _rolling_spearman(close, 20)
        # Signal = EMA(2) of raw correlation
        corr_signal = corr_20.ewm(span=2, adjust=False).mean()
        # Velocity = change in signal × 10 (rate of trend change)
        corr_velocity = (corr_signal - corr_signal.shift(1)).fillna(0) * 10.0
        
        df['spearman_corr_8'] = corr_8
        df['spearman_corr_20'] = corr_20
        df['corr_velocity'] = corr_velocity
        
        # === 2. SLOPE ACCELERATOR ===
        # Macro: LR slope of EMA(close, 1) = close, period 8
        slope_macro = _rolling_lr_slope(close, 8)
        # MAD-normalize and tanh-compress → velocity
        macro_med, macro_mad = _rolling_mad(slope_macro, 500)
        macro_mad_safe = macro_mad.clip(lower=1e-10)
        macro_z = (slope_macro - macro_med) / macro_mad_safe
        slope_velocity = pd.Series(
            _tanh_compress(macro_z.values * 0.3),
            index=df.index
        )
        
        # Micro: LR slope period 2, change rate
        slope_micro = _rolling_lr_slope(close, 2)
        micro_change = (slope_micro - slope_micro.shift(1)).fillna(0) * 100.0
        micro_smooth = micro_change.ewm(span=3, adjust=False).mean()
        micro_med, micro_mad = _rolling_mad(micro_smooth, 500)
        micro_mad_safe = micro_mad.clip(lower=1e-10)
        micro_z = (micro_smooth - micro_med) / micro_mad_safe
        slope_acceleration = pd.Series(
            _tanh_compress(micro_z.values * 0.3) * 0.8,
            index=df.index
        )
        
        # Signal line
        slope_signal = slope_velocity.ewm(span=2, adjust=False).mean()
        
        # Quiet zone detection: abs(velocity) <= 20th percentile
        abs_vel = slope_velocity.abs()
        quiet_threshold = abs_vel.rolling(250, min_periods=50).quantile(0.20)
        slope_quiet = (abs_vel <= quiet_threshold).astype(np.float32)
        
        df['slope_velocity'] = slope_velocity
        df['slope_acceleration'] = slope_acceleration
        df['slope_quiet_zone'] = slope_quiet
        
        # === 3. DIVERGENCE ENGINE ===
        # CCI at optimized period (45 for short-term preset)
        cci_45 = _compute_cci(close, high, low, 45)
        # Normalize CCI to roughly -1..+1 range for neural net friendliness
        df['cci_norm'] = (cci_45 / 200.0).clip(-2.0, 2.0)
        
        # Divergence detection using fractal pivots on CCI
        # Use confirmed pivots (5-bar left, 5-bar right = 10-bar lag)
        cci_arr = cci_45.values
        n = len(cci_arr)
        div_bull = np.zeros(n, dtype=np.float32)
        div_bear = np.zeros(n, dtype=np.float32)
        f_bars = 5
        max_lookback = 230
        
        # Track last pivot high/low on CCI
        last_ph_cci = np.nan
        last_ph_price = np.nan
        last_ph_bar = -9999
        last_pl_cci = np.nan
        last_pl_price = np.nan
        last_pl_bar = -9999
        
        h_arr = high.values
        l_arr = low.values
        
        for i in range(f_bars, n - f_bars):
            # Check for pivot high on CCI
            is_ph = True
            for j in range(1, f_bars + 1):
                if cci_arr[i] <= cci_arr[i - j] or cci_arr[i] <= cci_arr[i + j]:
                    is_ph = False
                    break
            if is_ph:
                if not np.isnan(last_ph_cci) and (i - last_ph_bar) <= max_lookback:
                    # Bearish divergence: price HH, CCI LH
                    if h_arr[i] > last_ph_price and cci_arr[i] < last_ph_cci:
                        div_bear[i + f_bars] = 1.0  # Signal on confirmation bar
                last_ph_cci = cci_arr[i]
                last_ph_price = h_arr[i]
                last_ph_bar = i
            
            # Check for pivot low on CCI
            is_pl = True
            for j in range(1, f_bars + 1):
                if cci_arr[i] >= cci_arr[i - j] or cci_arr[i] >= cci_arr[i + j]:
                    is_pl = False
                    break
            if is_pl:
                if not np.isnan(last_pl_cci) and (i - last_pl_bar) <= max_lookback:
                    # Bullish divergence: price LL, CCI HL
                    if l_arr[i] < last_pl_price and cci_arr[i] > last_pl_cci:
                        div_bull[i + f_bars] = 1.0
                last_pl_cci = cci_arr[i]
                last_pl_price = l_arr[i]
                last_pl_bar = i
        
        df['divergence_bull'] = div_bull
        df['divergence_bear'] = div_bear
        
        # === 4. PRICE DENSITY (PPI) ===
        # Percentile rank in HLC distribution (3x sampling)
        ppi_raw = _compute_ppi(close, high, low, 15)
        ppi_fast = ppi_raw.ewm(span=2, adjust=False).mean()
        ppi_slow = ppi_raw.ewm(span=28, adjust=False).mean()
        
        # Normalize to 0-1 for neural net
        df['ppi_raw'] = ppi_raw / 100.0
        df['ppi_fast'] = ppi_fast / 100.0
        df['ppi_slow'] = ppi_slow / 100.0
        
        # === 5. STRUCTURE ENGINE (K-Means S/R) ===
        # Simplified: find pivot highs/lows, cluster with K-means, compute distance
        atr = df.get('atr')
        if atr is None:
            atr = self.compute_atr(df)
        
        k_clusters = 3  # Short-term preset
        pivot_memory_size = 100
        update_freq = 50
        
        # Collect pivots (3-bar left, 3-bar right)
        dist_nearest = np.zeros(n, dtype=np.float64)
        sr_position = np.zeros(n, dtype=np.float64)  # -1 below, 0 in zone, +1 above
        
        pivot_prices = []
        for i in range(3, n - 3):
            # Pivot high
            if h_arr[i] > h_arr[i-1] and h_arr[i] > h_arr[i-2] and h_arr[i] > h_arr[i-3] \
               and h_arr[i] > h_arr[i+1] and h_arr[i] > h_arr[i+2] and h_arr[i] > h_arr[i+3]:
                pivot_prices.append((i, h_arr[i]))
            # Pivot low
            if l_arr[i] < l_arr[i-1] and l_arr[i] < l_arr[i-2] and l_arr[i] < l_arr[i-3] \
               and l_arr[i] < l_arr[i+1] and l_arr[i] < l_arr[i+2] and l_arr[i] < l_arr[i+3]:
                pivot_prices.append((i, l_arr[i]))
        
        # Simple K-means on pivot prices, updated every `update_freq` bars
        c_arr = close.values
        atr_arr = atr.values
        active_levels = np.array([])
        
        for i in range(50, n):
            if i % update_freq == 0 or len(active_levels) == 0:
                # Gather recent pivots (within last pivot_memory_size pivots before bar i)
                recent = [p for bar_idx, p in pivot_prices if bar_idx <= i]
                recent = recent[-pivot_memory_size:]
                
                if len(recent) >= k_clusters:
                    pts = np.array(recent, dtype=np.float64)
                    # K-means: initialize evenly spaced
                    p_min, p_max = pts.min(), pts.max()
                    centroids = np.linspace(p_min, p_max, k_clusters)
                    
                    for _ in range(5):  # 5 iterations
                        # Assign each pivot to nearest centroid
                        assignments = np.argmin(
                            np.abs(pts[:, None] - centroids[None, :]), axis=1
                        )
                        # Update centroids
                        for ci in range(k_clusters):
                            mask = assignments == ci
                            if mask.any():
                                centroids[ci] = pts[mask].mean()
                    
                    active_levels = np.sort(centroids)
            
            if len(active_levels) > 0:
                current_atr = max(atr_arr[i], 1e-10) if i < len(atr_arr) else 1e-10
                dists = active_levels - c_arr[i]
                abs_dists = np.abs(dists)
                nearest_idx = np.argmin(abs_dists)
                # ATR-normalized distance to nearest level
                dist_nearest[i] = dists[nearest_idx] / current_atr
                # Zone position: within 0.12 * ATR = inside zone
                zone_h = current_atr * 0.12
                if abs_dists[nearest_idx] <= zone_h:
                    sr_position[i] = 0.0  # Inside zone
                elif dists[nearest_idx] > 0:
                    sr_position[i] = 1.0  # Below nearest (level is above)
                else:
                    sr_position[i] = -1.0  # Above nearest (level is below)
        
        df['dist_nearest_sr'] = np.clip(dist_nearest, -5.0, 5.0)
        df['sr_zone_position'] = sr_position
        
        return df

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PATCH 8: Add regime meta-features for market state detection.
        All features are account-invariant (purely market-based signals).
        
        Features added:
        - realized_vol_24h_z: Realized volatility z-score over 24h
        - realized_vol_96h_z: Realized volatility z-score over 96h
        - trend_24h: Sign of lr_slope over 24h
        - trend_96h: Sign of lr_slope over 96h
        - is_trending: Boolean flag for strong trend + high vol regime
        
        Args:
            df: DataFrame with OHLC and existing features
            
        Returns:
            DataFrame with regime features added
        """
        # Realized volatility (std of log returns)
        log_ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # 24h and 96h realized vol
        vol_24h = log_ret.rolling(24, min_periods=12).std()
        vol_96h = log_ret.rolling(96, min_periods=48).std()
        
        # Z-score normalization of volatility (clip to ±5)
        vol_24h_mean = vol_24h.rolling(96, min_periods=48).mean()
        vol_24h_std = vol_24h.rolling(96, min_periods=48).std() + 1e-9
        realized_vol_24h_z = ((vol_24h - vol_24h_mean) / vol_24h_std).clip(-5, 5)
        
        vol_96h_mean = vol_96h.rolling(192, min_periods=96).mean()
        vol_96h_std = vol_96h.rolling(192, min_periods=96).std() + 1e-9
        realized_vol_96h_z = ((vol_96h - vol_96h_mean) / vol_96h_std).clip(-5, 5)
        
        # Trend proxy using lr_slope over different horizons
        hlc = (df['high'] + df['low'] + df['close']) / 3.0
        lr_slope_24h = _rolling_lr_slope(hlc, 24)
        lr_slope_96h = _rolling_lr_slope(hlc, 96)
        
        # Sign of slope (normalized to -1, 0, +1)
        trend_24h = np.sign(lr_slope_24h)
        trend_96h = np.sign(lr_slope_96h)
        
        # Market phase detection: is_trending
        # Trending if strong slope (>80th percentile) AND elevated vol (>60th percentile)
        # Use simple threshold instead of slow rolling quantile
        slope_96h_abs = np.abs(lr_slope_96h)
        
        # Fast approximation: use mean + std for thresholds instead of quantiles
        # ~80th percentile ≈ mean + 0.84*std, ~60th percentile ≈ mean + 0.25*std
        slope_mean = slope_96h_abs.rolling(192, min_periods=96).mean()
        slope_std = slope_96h_abs.rolling(192, min_periods=96).std() + 1e-9
        slope_q80_approx = slope_mean + 0.84 * slope_std  # ~80th percentile
        
        vol_mean = vol_24h.rolling(96, min_periods=48).mean()
        vol_std = vol_24h.rolling(96, min_periods=48).std() + 1e-9
        vol_q60_approx = vol_mean + 0.25 * vol_std  # ~60th percentile
        
        is_trending = ((slope_96h_abs > slope_q80_approx) & 
                       (vol_24h > vol_q60_approx)).astype(float)
        
        # Add to dataframe
        df['realized_vol_24h_z'] = realized_vol_24h_z
        df['realized_vol_96h_z'] = realized_vol_96h_z
        df['trend_24h'] = trend_24h
        df['trend_96h'] = trend_96h
        df['is_trending'] = is_trending
        
        return df

