import pandas as pd
import numpy as np
import pickle
from feature_engineering import engineer_features

def load_model_and_scaler():
    """Load the trained model and scaler"""
    with open("house/ml_models/saved_models/decision_tree.pkl", "rb") as f:
        model = pickle.load(f)
    with open("house/ml_models/saved_models/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("house/ml_models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

def create_test_data(floors, area=10.0, city_kathmandu=1, road_type_blacktopped=1):
    """Create test data for a specific number of floors"""
    # Create base features
    data = {
        'Floors': floors,
        'Area': area,
        'City_Kathmandu': city_kathmandu,
        'Road_Type_Blacktopped': road_type_blacktopped,
        'Road_Type_Gravelled': 0,
        'Road_Type_Soil Stabilized': 0,
        'City_Bhaktapur': 0,
        'City_Lalitpur': 0,
        'Price': 30000000  # Dummy price for feature engineering
    }
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    df_engineered = engineer_features(df)
    
    return df_engineered

def test_floor_predictions():
    """Test price predictions for different floor numbers"""
    # Load model and scaler
    model, scaler, feature_names = load_model_and_scaler()
    
    # Test different floor numbers
    floor_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    predictions = []
    
    print("\nTesting price predictions for different floor numbers:")
    print("Floor | Predicted Price (in NPR) | Price Difference")
    print("-" * 65)
    
    prev_price = None
    for floors in floor_numbers:
        # Create test data
        test_data = create_test_data(floors)
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in test_data.columns:
                test_data[feature] = 0
        
        # Select only the features used by the model
        test_data = test_data[feature_names]
        
        # Scale the features
        test_data_scaled = scaler.transform(test_data)
        
        # Make prediction
        pred_log = model.predict(test_data_scaled)
        pred_price = np.expm1(pred_log)[0]
        
        predictions.append(pred_price)
        
        # Calculate price difference
        if prev_price is not None:
            price_diff = pred_price - prev_price
            print(f"{floors:5d} | {pred_price:,.2f} | {price_diff:+,.2f}")
        else:
            print(f"{floors:5d} | {pred_price:,.2f} | -")
        
        prev_price = pred_price
    
    # Calculate average price increase per floor
    price_increases = [predictions[i] - predictions[i-1] for i in range(1, len(predictions))]
    avg_increase = sum(price_increases) / len(price_increases)
    print(f"\nAverage price increase per floor: {avg_increase:,.2f} NPR")
    
    # Calculate percentage increase per floor
    pct_increases = [(predictions[i] - predictions[i-1])/predictions[i-1] * 100 for i in range(1, len(predictions))]
    avg_pct_increase = sum(pct_increases) / len(pct_increases)
    print(f"Average percentage increase per floor: {avg_pct_increase:.2f}%")

if __name__ == "__main__":
    test_floor_predictions() 