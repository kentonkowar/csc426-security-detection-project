from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import sys


def read_dataset(file: str) -> pd.DataFrame:
    """
    Read a dataset from CSV (or other formats if needed).
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        print(f"Error reading dataset '{file}': {e}")
        return pd.DataFrame()

def clean_dataset(df: pd.DataFrame):
    # remove inf
    df = df.replace([float("inf"), "inf", "Infinity", "infinity"], sys.float_info.max)
    # remove nan
    df = df.replace("nan", -2)

    return df

def kmeans_clustering(df: pd.DataFrame, k: int = 3):
    """
    Perform K-means clustering on numeric columns.
    """
    if df.empty:
        print("Dataset is empty — cannot run K-means.")
        return None

    # Use only numeric columns for clustering
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    if numeric_df.empty:
        print("No numeric columns found for K-means.")
        return None

    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(numeric_df)

    df["cluster"] = labels
    return df, model



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py dataset.csv")
        exit(1)

    file = sys.argv[1]
    df = read_dataset(file)
    df = clean_dataset(df)
    # Run K-means
    result = kmeans_clustering(df, k=3)
    if result:
        df, kmeans_model = result
        print("K-means clustering complete. Cluster counts:")
        print(df["cluster"].value_counts())

    # Save results
    df.to_csv("output_with_clusters.csv", index=False)
    print("\nResults written to output_with_clusters.csv")

    exit(0)