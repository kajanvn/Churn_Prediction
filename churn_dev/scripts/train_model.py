import os
import joblib
import pandas as pd
from scripts.load_data import load_data
from scripts.preprocessing import (
    clean_total_charges,
    encode_features,
    scale_features,
    balance_data
)

from scripts.model_evaluation import (
    evaluate_models,
    plot_metric_comparison,
    get_best_model,
    plot_feature_importance
)

from scripts.model_training import split_data, train_and_tune_models
from sklearn.metrics import f1_score
from IPython.display import display




# Load and preprocess data
df = load_data("data/Telco-Customer-Churn.csv")
print("📂 Data loaded successfully")




df = clean_total_charges(df)
df = encode_features(df)

X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']


X_scaled = scale_features(X)

X_bal, y_bal = balance_data(X_scaled, y)



# Split and balance
X_train, X_test, y_train, y_test = split_data(X_bal, y_bal)

print("🔧 Data preprocessing completed")

print("🤖 Model training started...")

# Train all models
models = train_and_tune_models(X_train, y_train)

# Select best model using F1 score
best_model = None
best_model_name = ""
best_score = 0

for name, model in models.items():
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name



folder_path = "../churn_app/app"

# ✅ Check and create if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# 🧹 Clear previous best_model files
for filename in os.listdir(folder_path):
    if filename.startswith("best_model") and filename.endswith(".pkl"):
        os.remove(os.path.join(folder_path, filename))
        

# Save best model
model_path = f"../churn_app/app/best_model_{best_model_name}.pkl"
joblib.dump(best_model, model_path)
print(f"✅ Best model ({best_model_name}) saved to {model_path}")


# Save feature names
feature_names = list(X.columns)
feature_names_path = "../churn_app/app/feature_names.pkl"
joblib.dump(feature_names, feature_names_path)
print(f"✅ Feature names saved to {feature_names_path}")



print("🧪 Running model evaluation...")
metrics_df = evaluate_models(models, X_test, y_test)

plot_metric_comparison(metrics_df)

plot_feature_importance(best_model, feature_names,best_model_name)

print("📊 Model evaluation completed")

