import os
import seaborn as sns
import matplotlib.pyplot as plt


def missing_value_matrix(df):
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values Heatmap")
    plt.savefig("eda/missing_values.png")
    plt.close()

#def check_missing_values(df):
    #print(df.isnull().sum())

def visualize_numerical(df):
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"eda/distribution_{col}.png")
        plt.close()

def visualize_categorical(df):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        sns.countplot(data=df, x=col)
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"eda/countplot_{col}.png")
        plt.close()

def correlation_heatmap(df):
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("eda/correlation_heatmap.png")
    plt.close()

def boxplot_outliers(df):
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"eda/boxplot_{col}.png")
        plt.close()