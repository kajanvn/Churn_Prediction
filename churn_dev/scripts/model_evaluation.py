import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

def evaluate_models(models, X_test, y_test):
    if os.path.exists("plots"):
        for filename in os.listdir("plots"):
            if filename.endswith(".png") or filename.endswith(".csv"):
                os.remove(os.path.join("plots", filename))
    else:
        os.makedirs("plots", exist_ok=True)


    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)


        metrics.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })


        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix_{name}.png")
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"plots/roc_curve_{name}.png")
        plt.close()

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("plots/model_comparison_metrics.csv", index=False)
    return df_metrics

def plot_metric_comparison(metrics_df):
    metrics_df.set_index("Model")[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].plot(kind='bar')
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("plots/model_comparison_plot.png")
    plt.close()

def get_best_model(models, X_test, y_test):
    best_score = 0
    best_model = None
    best_name = ""
    for name, model in models.items():
        score = f1_score(y_test, model.predict(X_test))
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
    return best_model, best_name

def plot_feature_importance(model, features, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = features
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi_df.head(15), x="Importance", y="Feature")
        plt.title(f"Top 20 Feature Importances - {model_name}")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()

print("📊 Model evaluation completed")