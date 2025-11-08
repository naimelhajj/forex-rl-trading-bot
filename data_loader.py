"""
Data Loading and Preprocessing Module
Handles historical data loading, multi-pair synchronization, and data preparation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class DataLoader:
    """
    Handles loading and preprocessing of Forex data.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing historical data files
        """
        self.data_dir = Path(data_dir)
        self.data_cache = {}
        
    def load_csv(self, filepath: str, 
                 time_col: str = 'time',
                 parse_dates: bool = True) -> pd.DataFrame:
        """
        Load OHLC data from CSV file.
        
        Expected columns: time, open, high, low, close, volume (optional)
        
        Args:
            filepath: Path to CSV file
            time_col: Name of time column
            parse_dates: Whether to parse dates
            
        Returns:
            DataFrame with OHLC data
        """
        df = pd.read_csv(filepath)
        
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        
        if parse_dates and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def load_multiple_pairs(self, pair_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple currency pairs and synchronize timestamps.
        
        Args:
            pair_files: Dict of {pair_name: filepath}
            
        Returns:
            Dict of {pair_name: DataFrame} with synchronized timestamps
        """
        data = {}
        
        # Load all pairs
        for pair, filepath in pair_files.items():
            df = self.load_csv(filepath)
            data[pair] = df
            print(f"Loaded {pair}: {len(df)} rows")
        
        # Find common timestamp range
        common_index = None
        for df in data.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        print(f"\nCommon timestamp range: {len(common_index)} rows")
        
        # Synchronize all dataframes
        synchronized_data = {}
        for pair, df in data.items():
            synchronized_data[pair] = df.loc[common_index]
        
        return synchronized_data
    
    def generate_sample_data(self, 
                            pair: str = "EURUSD",
                            n_bars: int = 10000,
                            start_date: str = "2023-01-01",
                            freq: str = "1H",
                            base_price: float = 1.1000,
                            volatility: float = 0.0005) -> pd.DataFrame:
        """
        Generate synthetic OHLC data for testing.
        
        Args:
            pair: Currency pair name
            n_bars: Number of bars to generate
            start_date: Start date
            freq: Frequency (e.g., '1H', '1D')
            base_price: Starting price
            volatility: Price volatility
            
        Returns:
            DataFrame with synthetic OHLC data
        """
        # use lowercase hourly freq to avoid FutureWarning
        dates = pd.date_range(start=start_date, periods=n_bars, freq='h')
        
        # Generate random walk for close prices
        returns = np.random.normal(0, volatility, n_bars)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        data = []
        for i, close in enumerate(close_prices):
            # Add some intrabar volatility
            high = close * (1 + abs(np.random.normal(0, volatility * 0.5)))
            low = close * (1 - abs(np.random.normal(0, volatility * 0.5)))
            open_price = close_prices[i-1] if i > 0 else base_price
            
            data.append({
                'time': dates[i],
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': np.random.randint(100, 1000)
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('time')
        
        return df
    
    def generate_multiple_pairs(self, 
                               pairs: List[str],
                               n_bars: int = 10000,
                               start_date: str = "2023-01-01",
                               freq: str = "1H") -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for multiple pairs with correlated movements.
        
        Args:
            pairs: List of currency pair names
            n_bars: Number of bars to generate
            start_date: Start date
            freq: Frequency
            
        Returns:
            Dict of {pair_name: DataFrame}
        """
        # Base prices for common pairs
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.3000,
            'USDJPY': 110.00,
            'USDCHF': 0.9200,
            'AUDUSD': 0.7500,
            'USDCAD': 1.2500,
            'NZDUSD': 0.7000,
            'EURJPY': 121.00,
            'EURGBP': 0.8500,
            'GBPJPY': 143.00,
        }
        
        data = {}
        for pair in pairs:
            base_price = base_prices.get(pair, 1.0)
            df = self.generate_sample_data(
                pair=pair,
                n_bars=n_bars,
                start_date=start_date,
                freq=freq,
                base_price=base_price
            )
            data[pair] = df
        
        return data
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        return train_df, val_df, test_df
    
    def normalize_features(self, df: pd.DataFrame, 
                          feature_cols: List[str],
                          method: str = 'zscore') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features using specified method.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to normalize
            method: Normalization method ('zscore', 'minmax')
            
        Returns:
            Tuple of (normalized_df, normalization_params)
        """
        df = df.copy()
        params = {}
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / (std + 1e-10)
                params[col] = {'mean': mean, 'std': std, 'method': 'zscore'}
                
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-10)
                params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
        return df, params
    
    def apply_normalization(self, df: pd.DataFrame, 
                           params: Dict) -> pd.DataFrame:
        """
        Apply pre-computed normalization parameters to new data.
        
        Args:
            df: DataFrame to normalize
            params: Normalization parameters from normalize_features
            
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        for col, param in params.items():
            if col not in df.columns:
                continue
            
            if param['method'] == 'zscore':
                df[col] = (df[col] - param['mean']) / (param['std'] + 1e-10)
            elif param['method'] == 'minmax':
                df[col] = (df[col] - param['min']) / (param['max'] - param['min'] + 1e-10)
        
        return df
    
    def resample_to_business_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to business days (for fitness calculation).
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Resampled DataFrame
        """
        # Resample to business days and forward fill
        df_resampled = df.resample('B').last().ffill()
        return df_resampled


if __name__ == "__main__":
    print("Data Loader Module")
    print("=" * 50)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Generate sample data for multiple pairs
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY']
    print(f"\nGenerating sample data for {len(pairs)} pairs...")
    
    data = loader.generate_multiple_pairs(pairs, n_bars=1000)
    
    for pair, df in data.items():
        print(f"\n{pair}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Sample:\n{df.head(3)}")
    
    # Test data splitting
    print("\n\nTesting data split...")
    train, val, test = loader.split_data(data['EURUSD'])
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

