"""
Unit tests for data utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from utils.data_utils import DataLoader, DataCleaner, ip_to_int, merge_with_geolocation


@pytest.mark.unit
@pytest.mark.data
class TestDataLoader:
    """Test DataLoader class."""
    
    def test_init(self, temp_data_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(data_path=temp_data_dir + '/')
        assert loader.data_path == temp_data_dir + '/'
    
    def test_load_fraud_data_success(self, data_loader):
        """Test successful fraud data loading."""
        df = data_loader.load_fraud_data()
        assert not df.empty
        assert 'user_id' in df.columns
        assert 'class' in df.columns
    
    def test_load_fraud_data_file_not_found(self):
        """Test fraud data loading when file doesn't exist."""
        loader = DataLoader(data_path="nonexistent/")
        df = loader.load_fraud_data()
        assert df.empty
    
    def test_load_ip_country_data_success(self, data_loader):
        """Test successful IP country data loading."""
        df = data_loader.load_ip_country_data()
        assert not df.empty
        assert 'country' in df.columns
    
    def test_load_creditcard_data_file_not_found(self, data_loader):
        """Test credit card data loading when file doesn't exist."""
        df = data_loader.load_creditcard_data()
        assert df.empty


@pytest.mark.unit
@pytest.mark.data
class TestDataCleaner:
    """Test DataCleaner class."""
    
    def test_handle_missing_values_drop(self, data_cleaner):
        """Test dropping missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [1, np.nan, 3, 4]
        })
        result = data_cleaner.handle_missing_values(df, strategy='drop')
        assert len(result) == 2  # Only rows without NaN
        assert result.isnull().sum().sum() == 0
    
    def test_handle_missing_values_mean(self, data_cleaner):
        """Test filling missing values with mean."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [1, np.nan, 3, 4]
        })
        result = data_cleaner.handle_missing_values(df, strategy='mean')
        assert result.isnull().sum().sum() == 0
        assert result['A'].iloc[2] == df['A'].mean()
    
    def test_remove_duplicates(self, data_cleaner):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [1, 2, 2, 3]
        })
        result = data_cleaner.remove_duplicates(df)
        assert len(result) == 3  # One duplicate removed
    
    def test_correct_data_types(self, data_cleaner, sample_fraud_data):
        """Test data type correction."""
        result = data_cleaner.correct_data_types(sample_fraud_data)
        assert result['source'].dtype.name == 'category'
        assert result['browser'].dtype.name == 'category'
        assert result['sex'].dtype.name == 'category'
        assert result['class'].dtype == int


@pytest.mark.unit
@pytest.mark.data
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_ip_to_int_valid(self):
        """Test IP to integer conversion with valid IP."""
        result = ip_to_int("192.168.1.1")
        expected = (192 << 24) + (168 << 16) + (1 << 8) + 1
        assert result == expected
    
    def test_ip_to_int_invalid(self):
        """Test IP to integer conversion with invalid IP."""
        result = ip_to_int("invalid.ip")
        assert result == 0
    
    def test_merge_with_geolocation(self, sample_fraud_data):
        """Test geolocation merging."""
        # Create sample IP country data
        ip_country_df = pd.DataFrame({
            'lower_bound_ip_address': [0, 1000000000],
            'upper_bound_ip_address': [999999999, 4294967295],
            'country': ['US', 'UK']
        })
        
        # Use only first few rows for testing
        fraud_sample = sample_fraud_data.head(10).copy()
        
        result = merge_with_geolocation(fraud_sample, ip_country_df)
        assert 'country' in result.columns
        assert not result.empty
