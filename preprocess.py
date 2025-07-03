import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
import csv
import logging

os.makedirs('data', exist_ok=True)
with open('data/volve.csv', 'w') as f:
    csv.dump('data/volve.csv', 'w', newline='')
    
# Load dataset
df = pd.read_csv('data/volve.csv')

required_columns = ['Days', 'GOR', 'WHP', 'WHT', 'OilRate', 'WELL']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The dataset is missing the following required columns: {missing_columns}")

df = df.fillna(df.select_dtypes(include=[np.number]).mean())

logging.basicConfig(level=logging.INFO)
logging.info("Loading dataset...")
logging.info(f"Dataset shape: {df.shape}")
# Logging statement moved inside preprocess_data function
logging.info("Feature statistics saved to 'data/volve.csv'")

def preprocess_data():
    # Load dataset
    df = pd.read_csv('data/volve.csv')
    
    # Basic cleaning
    df = df.dropna(subset=['OilRate'])  # Drop rows without target
    df = df.fillna(df.mean())  # Impute missing values
    
    # Feature selection
    features = ['Days', 'GOR', 'WHP', 'WHT']
    target = 'OilRate'
    
    # Split by well to avoid data leakage
    wells = df['WELL'].unique()
    train_wells, test_wells = train_test_split(wells, test_size=0.2, random_state=42)
    
    train_df = df[df['WELL'].isin(train_wells)]
    test_df = df[df['WELL'].isin(test_wells)]
    
    # Log training and testing wells
    logging.info(f"Training wells: {len(train_wells)}, Testing wells: {len(test_wells)}")
    
    # Save feature stats
    feature_stats = {
        col: {
            'min': train_df[col].min(),
            'max': train_df[col].max()
        } for col in features
    }
    
    with open('models/feature_stats.json', 'w') as f:
        json.dump(feature_stats, f)
    
    return train_df[features], train_df[target], test_df[features], test_df[target]

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocess_data()