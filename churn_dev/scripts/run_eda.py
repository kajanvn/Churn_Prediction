import os
import glog
from scripts.load_data import load_data
from scripts.eda import (
    missing_value_matrix, 
    visualize_numerical, visualize_categorical,
    correlation_heatmap, boxplot_outliers
)


eda_dir = "eda"

for filename in os.listdir(eda_dir):
    if filename.endswith(".png"):
        os.remove(os.path.join(eda_dir, filename))

df = load_data("data/Telco-Customer-Churn.csv")


missing_value_matrix(df)
visualize_numerical(df)
visualize_categorical(df)
correlation_heatmap(df)
boxplot_outliers(df)

print("✅ EDA plots saved to churn_dev/eda/")