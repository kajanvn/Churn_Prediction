import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os



# Load encoded feature names used by the trained model
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "label_encoders.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_names.pkl")
label_encoders = joblib.load(ENCODERS_PATH)
feature_names = joblib.load(FEATURES_PATH)


def preprocess(df):
    df = df.copy()
    
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    # Ensure column order and completeness
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    
    return df