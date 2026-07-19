"""
Demo Script - Customer Churn Prediction Model
This script demonstrates how to use the trained model for predictions and retention strategies
"""
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from retention_strategies import analyze_and_strategize

def main():
    print("="*60)
    print("CUSTOMER CHURN PREDICTION MODEL - DEMO")
    print("="*60)
    
    # Run complete analysis pipeline
    print("\n1. Running retention analysis...")
    action_plan, impact_metrics = analyze_and_strategize(
        model_path='models/churn_model.joblib'
        # Uses sample data by default
    )
    
    # Display results
    print("\n2. Top 10 High-Risk Customers:")
    print("-"*60)
    print(action_plan.head(10).to_string(index=False))
    
    print("\n3. Business Impact Metrics:")
    print("-"*60)
    for key, value in impact_metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.2f}")
        else:
            print(f"{key:30s}: {value}")
    
    print("\n4. Sample Predictions:")
    print("-"*60)
    sample_customers = action_plan.head(5)
    for _, row in sample_customers.iterrows():
        print(f"\nCustomer {row['customer_id']}:")
        print(f"  Churn Probability: {row['churn_probability']:.1%}")
        print(f"  Risk Level: {row['risk_level']}")
        print(f"  Expected Value: ${row['expected_value']:,.2f}")
        print(f"  Recommended Actions: {row['recommended_actions']}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()