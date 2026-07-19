"""
Feature Engineering Module for Customer Churn Prediction
"""
import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.scaler = None
        self.encoder = None
        
    def create_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create usage-based features"""
        df['avg_monthly_spend'] = df['total_charges'] / (df['tenure'] + 1)
        df['charges_per_service'] = df['total_charges'] / (df['services_count'] + 1)
        df['monthly_charge_ratio'] = df['monthly_charges'] / (df['avg_monthly_spend'] + 1)
        return df
    
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tenure-based features"""
        df['tenure_group'] = pd.cut(df['tenure'], 
                                     bins=[0, 12, 24, 48, 72], 
                                     labels=['0-12', '13-24', '25-48', '49+'])
        df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
        df['is_long_term'] = (df['tenure'] >= 48).astype(int)
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        df['has_support_issues'] = (df['tech_support_calls'] > 5).astype(int)
        df['multiple_services_risk'] = ((df['services_count'] <= 2) & 
                                         (df['contract_type'] == 'Month-to-month')).astype(int)
        df['high_value_risk'] = ((df['total_charges'] > df['total_charges'].quantile(0.75)) & 
                                  (df['contract_type'] == 'Month-to-month')).astype(int)
        return df
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """One-hot encode categorical features"""
        self.categorical_features = categorical_cols
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return df
    
    def scale_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Scale numeric features"""
        from sklearn.preprocessing import StandardScaler
        self.numeric_features = numeric_cols
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        # Create derived features
        df = self.create_usage_features(df)
        df = self.create_tenure_features(df)
        df = self.create_interaction_features(df)
        
        # Define feature groups
        numeric_cols = ['tenure', 'monthly_charges', 'total_charges', 
                       'services_count', 'avg_monthly_spend', 'charges_per_service',
                       'monthly_charge_ratio', 'tech_support_calls', 'billing_issues']
        
        categorical_cols = ['gender', 'senior_citizen', 'partner', 'dependents',
                           'phone_service', 'multiple_lines', 'internet_service',
                           'online_security', 'online_backup', 'device_protection',
                           'tech_support', 'streaming_tv', 'streaming_movies',
                           'contract_type', 'paperless_billing', 'payment_method',
                           'tenure_group']
        
        # Encode categoricals
        df = self.encode_categorical(df, [col for col in categorical_cols if col in df.columns])
        
        # Scale numerics
        df = self.scale_features(df, [col for col in numeric_cols if col in df.columns])
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names"""
        return self.numeric_features + self.categorical_features