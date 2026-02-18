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

