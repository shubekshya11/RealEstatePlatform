import pandas as pd
import numpy as np

def create_floor_features(df):
    """
    Create new features related to floors to better capture the relationship with price.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with 'Floors' column
    
    Returns:
    pandas.DataFrame: Dataframe with new floor-related features
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Floor categories with more granular divisions
    def create_floor_category(floors):
        if floors <= 1:
            return 'single_floor'
        elif floors <= 2:
            return 'double_floor'
        elif floors <= 3:
            return 'triple_floor'
        elif floors <= 5:
            return 'mid_rise'
        else:
            return 'high_rise'
    
    df['floor_category'] = df['Floors'].apply(create_floor_category)
    floor_categories = pd.get_dummies(df['floor_category'], prefix='floor')
    df = pd.concat([df, floor_categories], axis=1)
    
    # Enhanced Floor-Area interactions
    df['area_per_floor'] = df['Area'] / df['Floors']
    df['total_built_area'] = df['Area'] * df['Floors']
    df['floor_area_ratio'] = df['Floors'] / df['Area']  # New feature
    
    # Floor price premium with exponential scaling
    floor_price_avg = df.groupby('Floors')['Price'].mean()
    df['floor_price_premium'] = df['Floors'].map(floor_price_avg)
    df['floor_price_premium_scaled'] = np.exp(df['Floors'] * 0.1)  # New feature
    
    # Floor-Location interaction with more granularity
    if 'City_Kathmandu' in df.columns:
        df['floor_city_interaction'] = df['Floors'] * df['City_Kathmandu']
        df['floor_city_premium'] = df['floor_price_premium'] * df['City_Kathmandu']  # New feature
    
    # Floor quality indicators
    df['floor_quality'] = df['Floors'].apply(lambda x: 'high' if x > 3 else 'medium' if x > 1 else 'low')
    floor_quality_dummies = pd.get_dummies(df['floor_quality'], prefix='floor_quality')
    df = pd.concat([df, floor_quality_dummies], axis=1)
    
    return df

def create_area_features(df):
    """
    Create new features related to area to better capture the relationship with price.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with 'Area' column
    
    Returns:
    pandas.DataFrame: Dataframe with new area-related features
    """
    df = df.copy()
    
    # Price per area
    df['price_per_area'] = df['Price'] / df['Area']
    
    # Area categories
    def create_area_category(area):
        if area <= 4:
            return 'small'
        elif area <= 8:
            return 'medium'
        else:
            return 'large'
    
    df['area_category'] = df['Area'].apply(create_area_category)
    area_categories = pd.get_dummies(df['area_category'], prefix='area')
    df = pd.concat([df, area_categories], axis=1)
    
    return df

def create_location_features(df):
    """
    Create new features related to location to better capture the relationship with price.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with location columns
    
    Returns:
    pandas.DataFrame: Dataframe with new location-related features
    """
    df = df.copy()
    
    # Create interaction between city and road type
    if 'City_Kathmandu' in df.columns and 'Road_Type_Blacktopped' in df.columns:
        df['city_road_interaction'] = df['City_Kathmandu'] * df['Road_Type_Blacktopped']
    
    return df

def engineer_features(df):
    """
    Apply all feature engineering functions to the dataframe.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    pandas.DataFrame: Dataframe with all engineered features
    """
    df = create_floor_features(df)
    df = create_area_features(df)
    df = create_location_features(df)
    
    return df 