# src/model/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# --- 1. Setup Data Paths ---
MODEL_PATH = 'src/model/model.pkl'
DATA_PATH = 'src/model/data.csv' 
# In a real project, this would be downloaded from S3/GCS/DVC

def load_data():
    # Placeholder: Replace with actual data loading if available. 
    # For simplicity, we will simulate the structure using the standard
    # Telco Churn dataset features.
    print("Loading simulated Telco Churn dataset...")
    
    # We will use search to find a suitable URL for the dataset
    # If the search fails, we'll proceed with creating a mock dataset (as shown below)

    # Mock Data Creation (if actual dataset is not immediately available):
    data = {
        'tenure': [1, 25, 50, 10, 60],
        'MonthlyCharges': [29.85, 56.95, 104.80, 80.00, 115.00],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'Two year'],
        'OnlineSecurity': ['No', 'Yes', 'No', 'Yes', 'No'],
        'Churn': [1, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    # df.to_csv(DATA_PATH, index=False) # Uncomment to save mock data
    
    return df

def train_and_save_model(df):
    
    # Define features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # --- 2. Preprocessing Pipeline ---
    numeric_features = ['tenure', 'MonthlyCharges']
    categorical_features = ['Contract', 'OnlineSecurity'] 

    # Create the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )

    # --- 3. Full Pipeline (Preprocessing + Model) ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    # Train the model
    print("Starting model training...")
    pipeline.fit(X, y)
    print("Model training complete.")

    # --- 4. Save the model artifact ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

if __name__ == '__main__':
    # We will use the search tool to find the Telco Churn dataset
    df = load_data() 
    train_and_save_model(df)