"""
Training Script for Customer Churn Prediction Model
"""
import pandas as pd
import numpy as np
import json
from churn_model import ChurnPredictor
import os

def load_sample_data() -> pd.DataFrame:
    """
    Load or create sample customer churn dataset
    In production, replace with actual data loading from database or CSV
    """
    # Sample data structure - replace with actual data source
    np.random.seed(42)
    n_samples = 7043
    
    data = {
        'customer_id': range(n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.2]),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(0, 8000, n_samples),
        'services_count': np.random.randint(1, 6, n_samples),
        'tech_support_calls': np.random.randint(0, 10, n_samples),
        'billing_issues': np.random.randint(0, 5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (churn) with realistic correlations
    churn_prob = (
        0.3 * (df['contract_type'] == 'Month-to-month').astype(int) +
        0.2 * (df['tenure'] < 12).astype(int) +
        0.15 * (df['tech_support_calls'] > 5).astype(int) +
        0.1 * (df['payment_method'] == 'Electronic check').astype(int) +
        0.05 * np.random.random(n_samples)
    )
    
    df['churn'] = (churn_prob > 0.4).astype(int)
    
    return df

def train_model(data_path: str = None, tune_hyperparameters: bool = True) -> ChurnPredictor:
    """
    Main training pipeline
    Args:
        data_path: Path to dataset CSV file. If None, uses sample data
        tune_hyperparameters: Whether to perform hyperparameter tuning
    Returns:
        Trained ChurnPredictor model
    """
    # Load data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Using sample data for demonstration")
        df = load_sample_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # Initialize model
    predictor = ChurnPredictor()
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(df)
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train model
    print("\nTraining model...")
    predictor.train(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric:15s}: {value:.4f}")
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    # Combine train and test for cross-validation
    X_combined = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0).values
    y_combined = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    predictor.cross_validate(X_combined, y_combined, cv=5)
    
    # Feature importance
    print("\n" + "="*50)
    print("TOP 10 IMPORTANT FEATURES")
    print("="*50)
    top_features = predictor.get_top_features(10)
    print(top_features.to_string(index=False))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/churn_model.joblib'
    predictor.save_model(model_path)
    
    # Save metrics to JSON
    metrics_path = 'models/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({k: v for k, v in metrics.items()}, f, indent=2, default=str)
    
    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    
    return predictor

if __name__ == "__main__":
    # Train model with hyperparameter tuning
    model = train_model(tune_hyperparameters=False)
    
    # Example prediction on new data
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # You would load actual customer data here
    # new_customers = pd.read_csv('new_customers.csv')
    # predictions = model.predict(new_customers)
    # prediction_probas = model.predict_proba(new_customers)