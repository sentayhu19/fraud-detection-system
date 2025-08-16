"""
Unit tests for model training utilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.model_training import ModelTrainer, cross_validate_models


@pytest.mark.unit
@pytest.mark.model
class TestModelTrainer:
    """Test ModelTrainer class."""
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(random_state=42)
        assert trainer.random_state == 42
        assert trainer.models == {}
        assert trainer.best_params == {}
    
    def test_train_logistic_regression_no_tuning(self, model_trainer, sample_features_targets):
        """Test logistic regression training without hyperparameter tuning."""
        X, y = sample_features_targets
        model = model_trainer.train_logistic_regression(X, y, hyperparameter_tuning=False)
        
        assert isinstance(model, LogisticRegression)
        assert 'logistic_regression' in model_trainer.models
        assert model.random_state == 42
    
    @patch('utils.model_training.GridSearchCV')
    def test_train_logistic_regression_with_tuning(self, mock_grid_search, model_trainer, sample_features_targets):
        """Test logistic regression training with hyperparameter tuning."""
        X, y = sample_features_targets
        
        # Mock GridSearchCV
        mock_grid_instance = Mock()
        mock_grid_instance.best_params_ = {'C': 1.0, 'penalty': 'l2'}
        mock_grid_instance.best_score_ = 0.85
        mock_grid_instance.best_estimator_ = LogisticRegression()
        mock_grid_search.return_value = mock_grid_instance
        
        model = model_trainer.train_logistic_regression(X, y, hyperparameter_tuning=True)
        
        assert mock_grid_search.called
        assert 'logistic_regression' in model_trainer.best_params
    
    def test_train_random_forest_no_tuning(self, model_trainer, sample_features_targets):
        """Test random forest training without hyperparameter tuning."""
        X, y = sample_features_targets
        model = model_trainer.train_random_forest(X, y, hyperparameter_tuning=False)
        
        assert isinstance(model, RandomForestClassifier)
        assert 'random_forest' in model_trainer.models
        assert model.random_state == 42
    
    def test_train_xgboost_no_tuning(self, model_trainer, sample_features_targets):
        """Test XGBoost training without hyperparameter tuning."""
        X, y = sample_features_targets
        model = model_trainer.train_xgboost(X, y, hyperparameter_tuning=False)
        
        assert isinstance(model, XGBClassifier)
        assert 'xgboost' in model_trainer.models
        assert model.random_state == 42
    
    def test_train_lightgbm_not_available(self, model_trainer, sample_features_targets):
        """Test LightGBM training when not available."""
        X, y = sample_features_targets
        with patch('utils.model_training.LIGHTGBM_AVAILABLE', False):
            model = model_trainer.train_lightgbm(X.values, y.values)
            assert model is None
    
    def test_get_model(self, model_trainer, sample_features_targets):
        """Test getting trained model."""
        X, y = sample_features_targets
        model_trainer.train_logistic_regression(X, y, hyperparameter_tuning=False)
        
        retrieved_model = model_trainer.get_model('logistic_regression')
        assert retrieved_model is not None
        assert isinstance(retrieved_model, LogisticRegression)
    
    def test_get_best_params(self, model_trainer):
        """Test getting best parameters."""
        model_trainer.best_params['test_model'] = {'param1': 'value1'}
        params = model_trainer.get_best_params('test_model')
        assert params == {'param1': 'value1'}
    
    def test_train_all_models(self, model_trainer, sample_features_targets):
        """Test training all models."""
        X, y = sample_features_targets
        models_to_train = ['logistic_regression', 'random_forest']
        
        trained_models = model_trainer.train_all_models(
            X, y, 
            models_to_train=models_to_train, 
            hyperparameter_tuning=False
        )
        
        assert len(trained_models) == 2
        assert 'logistic_regression' in trained_models
        assert 'random_forest' in trained_models
    
    def test_train_unknown_model(self, model_trainer, sample_features_targets):
        """Test training unknown model type."""
        X, y = sample_features_targets
        
        trained_models = model_trainer.train_all_models(
            X, y, 
            models_to_train=['unknown_model'], 
            hyperparameter_tuning=False
        )
        
        assert len(trained_models) == 0


@pytest.mark.unit
@pytest.mark.model
class TestCrossValidation:
    """Test cross-validation functions."""
    
    @patch('utils.model_training.cross_val_score')
    def test_cross_validate_models(self, mock_cv_score, sample_features_targets):
        """Test cross-validation of models."""
        X, y = sample_features_targets
        
        # Mock cross-validation scores
        mock_cv_score.return_value = np.array([0.8, 0.85, 0.82, 0.88, 0.79])
        
        models = {
            'model1': Mock(),
            'model2': Mock()
        }
        
        results = cross_validate_models(models, X, y)
        
        assert len(results) == 2
        assert 'model1' in results
        assert 'model2' in results
        assert 'mean_score' in results['model1']
        assert 'std_score' in results['model1']
        assert 'scores' in results['model1']
