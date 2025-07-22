# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# import pickle
# from feature_engineering import engineer_features

# data = pd.read_csv("6thSemProject-main/homePricePredictior/house/Final_cleaned_encoded_data.csv")

# print("Initial data shape:", data.shape)
# print("\nInitial price statistics:")
# print(data["Price"].describe())

# # Apply feature engineering
# print("\nApplying feature engineering...")
# data = engineer_features(data)
# print("\nNew features added:")
# print(data.columns.tolist())

# # Replace inf/-inf with NaN, then fill NaN
# data = data.replace([np.inf, -np.inf], np.nan)
# # Fill numeric columns with median again (in case new NaNs were introduced)
# numeric_cols = data.select_dtypes(include=[np.number]).columns
# for col in numeric_cols:
#     data[col] = data[col].fillna(data[col].median())

# # Fill categorical columns with mode
# categorical_cols = data.select_dtypes(include=['object', 'category']).columns
# for col in categorical_cols:
#     data[col] = data[col].fillna(data[col].mode()[0])

# print("\nFinal data shape:", data.shape)

# print("\nFinal price statistics:")
# print(data["Price"].describe())

# X = data.drop("Price", axis=1)
# # Keep only numeric columns for model training
# X = X.select_dtypes(include=[np.number])
# y = data["Price"]

# y_log = np.log1p(y)

# print("\nLog-transformed price statistics:")
# print(y_log.describe())

# print("\nNumber of NaN values in features:", X.isnull().sum().sum())
# print("Number of NaN values in target:", y_log.isnull().sum())

# X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.15, random_state=42)

# train_data = pd.concat([pd.DataFrame(X_train, columns=X.columns), 
#                        pd.Series(y_train, name='Price_Log')], axis=1)
# test_data = pd.concat([pd.DataFrame(X_test, columns=X.columns), 
#                       pd.Series(y_test, name='Price_Log')], axis=1)

# train_data['Price_Original'] = np.expm1(train_data['Price_Log'])
# test_data['Price_Original'] = np.expm1(test_data['Price_Log'])

# train_data.to_csv('media/datasets/train_dataset.csv', index=False)
# test_data.to_csv('media/datasets/test_dataset.csv', index=False)

# print("\nDatasets saved:")
# print(f"Training set shape: {train_data.shape}")
# print(f"Test set shape: {test_data.shape}")

# feature_scaler = StandardScaler()
# X_train_scaled = feature_scaler.fit_transform(X_train)
# X_test_scaled = feature_scaler.transform(X_test)

# # Train models with new features
# tree_regressor = DecisionTreeRegressor(
#     max_depth=15,
#     min_samples_split=10,
#     min_samples_leaf=8,
#     random_state=42,
#     max_features='sqrt',
#     min_weight_fraction_leaf=0.01
# )
# tree_regressor.fit(X_train_scaled, y_train)

# svr_regressor = SVR(
#     kernel='rbf',
#     C=500.0,
#     epsilon=0.01,
#     gamma='scale'
# )
# svr_regressor.fit(X_train_scaled, y_train)

# def evaluate_model(model, X_scaled, y_true, model_name):
#     y_pred_log = model.predict(X_scaled)
#     y_pred = np.expm1(y_pred_log)
#     y_true_original = np.expm1(y_true)
    
#     r2 = r2_score(y_true_original, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true_original, y_pred))
    
#     print(f"\n{model_name} Performance:")
#     print(f"R² Score: {r2:.4f}")
#     print(f"RMSE: {rmse:.4f}")
    
#     return r2, rmse

# dt_r2, dt_rmse = evaluate_model(tree_regressor, X_test_scaled, y_test, "Decision Tree")
# svm_r2, svm_rmse = evaluate_model(svr_regressor, X_test_scaled, y_test, "SVM")

# feature_names = X.columns.tolist()
# print("\nFeatures being saved:", feature_names)

# pickle.dump(tree_regressor, open("house/ml_models/saved_models/decision_tree.pkl", "wb"))
# pickle.dump(svr_regressor, open("house/ml_models/saved_models/svm_model.pkl", "wb"))
# pickle.dump(feature_scaler, open("house/ml_models/saved_models/feature_scaler.pkl", "wb"))
# pickle.dump(feature_names, open("house/ml_models/feature_names.pkl", "wb"))

# print("\nModels trained and saved successfully!")

# # Filtering data based on the new conditions
# data = data[~((data['Price'] > 20000000) & (data['Floors'] < 2) & (data['Area'] < 4))]
# data = data[~((data['Price'] > 50000000) & (data['Floors'] < 3) & (data['Area'] < 5))]
# data = data[~((data['Price'] < 16000000) & (data['Floors'] > 2.5))]

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pickle
import os
from feature_engineering import engineer_features

def safe_load_data(filepath):
    """Safely load data with error handling"""
    try:
        data = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        print("Initial data shape:", data.shape)
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {str(e)}")
        return None

def handle_missing_values(data):
    """Handle missing values with robust filling"""
    if data is None:
        return None
        
    # Replace infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Fill numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isna().any():
            fill_value = data[col].median() if len(data[col].notna()) > 0 else 0
            data[col] = data[col].fillna(fill_value)
    
    # Fill categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if data[col].isna().any() and len(data[col].mode()) > 0:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    return data

def apply_smart_filters(data):
    """Apply data filters only if columns exist"""
    if data is None:
        return None
        
    filters = []
    conditions = [
        ('Price', 'Floors', 'Area', 20000000, 2, 4),
        ('Price', 'Floors', 'Area', 50000000, 3, 5),
        ('Price', 'Floors', None, 16000000, 2.5, None)
    ]
    
    for price_col, floor_col, area_col, price_val, floor_val, area_val in conditions:
        cols_exist = all(col in data.columns for col in [price_col, floor_col] if area_col is None else [price_col, floor_col, area_col])
        
        if cols_exist:
            if area_col:
                filters.append(~((data[price_col] > price_val) & 
                              (data[floor_col] < floor_val) & 
                              (data[area_col] < area_val)))
            else:
                filters.append(~((data[price_col] < price_val) & 
                              (data[floor_col] > floor_val)))
    
    if filters:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter &= f
        data = data[combined_filter]
    
    return data

def prepare_features(data):
    """Prepare features with validation"""
    if data is None or 'Price' not in data.columns:
        return None, None
        
    X = data.drop("Price", axis=1)
    X = X.select_dtypes(include=[np.number])  # Keep only numeric features
    
    # Ensure we have features to work with
    if len(X.columns) == 0:
        return None, None
        
    y = data["Price"]
    y_log = np.log1p(y)
    
    return X, y_log

def save_datasets(X_train, X_test, y_train, y_test, feature_names):
    """Save processed datasets"""
    os.makedirs('media/datasets', exist_ok=True)
    
    train_data = pd.concat([pd.DataFrame(X_train, columns=feature_names), 
                         pd.Series(y_train, name='Price_Log')], axis=1)
    test_data = pd.concat([pd.DataFrame(X_test, columns=feature_names), 
                        pd.Series(y_test, name='Price_Log')], axis=1)
    
    train_data['Price_Original'] = np.expm1(train_data['Price_Log'])
    test_data['Price_Original'] = np.expm1(test_data['Price_Log'])
    
    train_data.to_csv('media/datasets/train_dataset.csv', index=False)
    test_data.to_csv('media/datasets/test_dataset.csv', index=False)
    
    return train_data, test_data

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate models with robust scaling"""
    # Initialize scaler
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Model configurations
    models = {
        'Decision Tree': DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=8,
            random_state=42,
            max_features='sqrt',
            min_weight_fraction_leaf=0.01
        ),
        'SVM': SVR(
            kernel='rbf',
            C=500.0,
            epsilon=0.01,
            gamma='scale'
        )
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred_log = model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        
        results[name] = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'model': model
        }
        
        print(f"\n{name} Performance:")
        print(f"R² Score: {results[name]['r2']:.4f}")
        print(f"RMSE: {results[name]['rmse']:.4f}")
    
    return results, feature_scaler

def save_models(models, feature_scaler, feature_names):
    """Save models and artifacts"""
    os.makedirs('house/ml_models/saved_models', exist_ok=True)
    
    for name, result in models.items():
        with open(f"house/ml_models/saved_models/{name.lower().replace(' ', '_')}.pkl", 'wb') as f:
            pickle.dump(result['model'], f)
    
    with open("house/ml_models/saved_models/feature_scaler.pkl", 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    with open("house/ml_models/feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)

def main():
    # Load and prepare data
    data = safe_load_data("6thSemProject-main/homePricePredictior/house/Final_cleaned_encoded_data.csv")
    if data is None:
        return
        
    # Apply feature engineering
    print("\nApplying feature engineering...")
    data = engineer_features(data)
    
    # Handle missing values
    data = handle_missing_values(data)
    if data is None:
        return
        
    # Apply smart filters
    data = apply_smart_filters(data)
    print("\nFiltered data shape:", data.shape)
    
    # Prepare features
    X, y_log = prepare_features(data)
    if X is None:
        return
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.15, random_state=42
    )
    
    # Save datasets
    train_data, test_data = save_datasets(X_train, X_test, y_train, y_test, X.columns.tolist())
    print(f"\nTraining set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")
    
    # Train and evaluate models
    results, feature_scaler = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, X.columns.tolist()
    )
    
    # Save models
    save_models(results, feature_scaler, X.columns.tolist())
    print("\nModels trained and saved successfully!")

if __name__ == "__main__":
    main()