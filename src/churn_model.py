"""
Customer Churn Prediction Model using XGBoost
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
from sklearn.preprocessing import StandardScaler
import joblib
import json
from typing import Tuple, Dict, Any
from feature_engineering import FeatureEngineer

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.metrics = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'churn') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Engineer features
        X = self.feature_engineer.engineer_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              tune_hyperparameters: bool = True) -> None:
        """Train XGBoost model"""
        if tune_hyperparameters:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='f1',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train with default parameters
            self.model = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Feature importance
        feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'feature_{i}' for i in range(X_test.shape[1])]
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_.tolist()))
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1', n_jobs=-1)
        cv_results = {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores.tolist()
        }
        print(f"Cross-validation F1 scores: {scores}")
        print(f"Mean F1: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return cv_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str) -> None:
        """Save model and feature engineer to disk"""
        joblib.dump({
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk"""
        artifacts = joblib.load(filepath)
        self.model = artifacts['model']
        self.feature_engineer = artifacts['feature_engineer']
        self.metrics = artifacts['metrics']
        self.feature_importance = artifacts['feature_importance']
        print(f"Model loaded from {filepath}")
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top N most important features"""
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = pd.DataFrame(sorted_features[:n], columns=['feature', 'importance'])
        return top_features
    
    def plot_feature_importance(self, n: int = 15) -> None:
        """Plot feature importance"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        top_features = self.get_top_features(n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top Feature Importance for Churn Prediction')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()