"""
Data preprocessing utilities for machine learning.

This module provides functions for cleaning, transforming, and preparing
data from SQL Server for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for SQL Server data.
    """
    
    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_names: Optional[List[str]] = None
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop', 'forward_fill')
            columns: Specific columns to process (None for all)
            
        Returns:
            DataFrame with missing values handled
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.columns.tolist()
        
        for col in columns:
            if col not in df_copy.columns:
                continue
                
            if df_copy[col].isnull().sum() == 0:
                continue
                
            if strategy == 'mean' and df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median' and df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode().iloc[0], inplace=True)
            elif strategy == 'forward_fill':
                df_copy[col].fillna(method='ffill', inplace=True)
            elif strategy == 'drop':
                df_copy.dropna(subset=[col], inplace=True)
            
        logger.info(f"Missing values handled using {strategy} strategy for {len(columns)} columns")
        return df_copy
    
    def encode_categorical_variables(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode
            method: Encoding method ('onehot', 'label')
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                continue
                
            if method == 'onehot':
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse_output=False, drop='first')
                    
                # Fit and transform or just transform
                encoded_data = self.encoders[col].fit_transform(df_copy[[col]])
                feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]]
                
                # Create DataFrame with encoded columns
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_copy.index)
                
                # Drop original column and concatenate encoded columns
                df_copy = df_copy.drop(columns=[col])
                df_copy = pd.concat([df_copy, encoded_df], axis=1)
                
            elif method == 'label':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                
                df_copy[col] = self.encoders[col].fit_transform(df_copy[col].astype(str))
        
        logger.info(f"Categorical encoding completed for {len(columns)} columns using {method} method")
        return df_copy
    
    def scale_numerical_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns to scale
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            DataFrame with scaled numerical features
        """
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                continue
                
            if method == 'standard':
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                df_copy[col] = self.scalers[col].fit_transform(df_copy[[col]]).flatten()
                
            elif method == 'minmax':
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                df_copy[col] = self.scalers[col].fit_transform(df_copy[[col]]).flatten()
        
        logger.info(f"Feature scaling completed for {len(columns)} columns using {method} method")
        return df_copy
    
    def remove_outliers(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from numerical columns.
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns to process
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df_copy = df.copy()
        initial_rows = len(df_copy)
        
        for col in columns:
            if col not in df_copy.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                df_copy = df_copy[z_scores <= threshold]
        
        removed_rows = initial_rows - len(df_copy)
        logger.info(f"Outlier removal completed. Removed {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%)")
        return df_copy
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        df_copy = df.copy()
        
        # Sort by Symbol and Date to ensure proper calculation
        if 'Symbol' in df_copy.columns and 'Date' in df_copy.columns:
            df_copy = df_copy.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Enhanced technical analysis features for stock trading
        df_copy = self._create_technical_indicators(df_copy)
        
        # Date-based features if datetime columns exist
        datetime_columns = df_copy.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            df_copy[f'{col}_year'] = df_copy[col].dt.year
            df_copy[f'{col}_month'] = df_copy[col].dt.month
            df_copy[f'{col}_day'] = df_copy[col].dt.day
            df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
            df_copy[f'{col}_quarter'] = df_copy[col].dt.quarter
        
        # Interaction features for numerical columns
        numerical_columns = df_copy.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_columns) >= 2:
            # Create some interaction features (be careful not to create too many)
            for i, col1 in enumerate(numerical_columns[:3]):  # Limit to first 3 columns
                for col2 in numerical_columns[i+1:4]:  # Limit interactions
                    if col1 != col2:
                        df_copy[f'{col1}_x_{col2}'] = df_copy[col1] * df_copy[col2]
        
        logger.info(f"Feature engineering completed. Added {len(df_copy.columns) - len(df.columns)} new features")
        return df_copy
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators for stock trading analysis.
        
        Args:
            df: Input DataFrame with stock price data
            
        Returns:
            DataFrame with technical indicators added
        """
        df_copy = df.copy()
        
        # Check if we have the necessary columns
        required_cols = ['Close', 'Volume']
        if not all(col in df_copy.columns for col in required_cols):
            logger.warning("Missing required columns for technical indicators. Skipping technical analysis.")
            return df_copy
        
        # Group by symbol to calculate indicators per stock
        if 'Symbol' in df_copy.columns:
            df_copy = df_copy.groupby('Symbol').apply(self._calculate_indicators_per_symbol).reset_index(drop=True)
        else:
            df_copy = self._calculate_indicators_per_symbol(df_copy)
        
        return df_copy
    
    def _calculate_indicators_per_symbol(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a single symbol's data.
        
        Args:
            group_df: DataFrame containing data for a single symbol
            
        Returns:
            DataFrame with indicators calculated
        """
        df = group_df.copy()
        
        # Simple Moving Averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df['Close'].ewm(span=5).mean()
        df['ema_10'] = df['Close'].ewm(span=10).mean()
        df['ema_20'] = df['Close'].ewm(span=20).mean()
        df['ema_50'] = df['Close'].ewm(span=50).mean()
        
        # MACD Calculation
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Price vs Moving Average ratios
        df['price_vs_sma20'] = df['Close'] / df['sma_20']
        df['price_vs_sma50'] = df['Close'] / df['sma_50']
        df['price_vs_ema20'] = df['Close'] / df['ema_20']
        
        # Moving Average relationships
        df['sma20_vs_sma50'] = df['sma_20'] / df['sma_50']
        df['ema20_vs_ema50'] = df['ema_20'] / df['ema_50']
        df['sma5_vs_sma20'] = df['sma_5'] / df['sma_20']
        
        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_sma_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Price momentum features
        df['price_momentum_5'] = df['Close'] / df['Close'].shift(5)
        df['price_momentum_10'] = df['Close'] / df['Close'].shift(10)
        
        # Volatility features
        df['price_volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
        df['price_volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Trend strength indicators
        df['trend_strength_10'] = df['Close'].rolling(window=10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() != 0 else 0
        )
        
        # High-low spreads
        if 'High' in df.columns and 'Low' in df.columns:
            df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
            df['hl_spread_sma_5'] = df['hl_spread'].rolling(window=5).mean()
        
        return df
    
    def prepare_for_ml(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for machine learning by splitting into train/test sets.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of test set
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
        )
        
        logger.info(f"Data split completed. Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a comprehensive summary of the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data summary statistics
    """
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numerical_summary': df.describe().to_dict(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'numerical_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    return summary


def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect common data quality issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with detected issues
    """
    issues = {
        'missing_values': {},
        'duplicates': len(df) - len(df.drop_duplicates()),
        'high_cardinality_columns': {},
        'low_variance_columns': [],
        'potential_outliers': {}
    }
    
    # Missing values
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 0:
            issues['missing_values'][col] = f"{missing_pct:.2f}%"
    
    # High cardinality categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        if unique_count > len(df) * 0.9:  # More than 90% unique values
            issues['high_cardinality_columns'][col] = unique_count
    
    # Low variance numerical columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if df[col].var() < 0.01:  # Very low variance
            issues['low_variance_columns'].append(col)
    
    # Potential outliers using IQR method
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        if outliers > 0:
            issues['potential_outliers'][col] = f"{outliers} ({outliers/len(df)*100:.2f}%)"
    
    return issues
