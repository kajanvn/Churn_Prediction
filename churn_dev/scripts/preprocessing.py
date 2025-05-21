import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib




def clean_total_charges(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df


'''
def encode_features(df):
    df = df.copy()
    exclude_cols = ['customerID', 'Churn']
    cat_cols = df.select_dtypes(include='object').columns
    cols_to_encode = [col for col in cat_cols if col not in exclude_cols]
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df
'''
def encode_features(df):
    df = df.copy()
    exclude_cols = ['customerID', 'Churn']
    cat_cols = df.select_dtypes(include='object').columns
    cols_to_encode = [col for col in cat_cols if col not in exclude_cols]

    label_encoders = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoder

    # Save all encoders for app usage
    joblib.dump(label_encoders, "../churn_app/app/label_encoders.pkl")

    return df


def scale_features(df):   
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def balance_data(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)  