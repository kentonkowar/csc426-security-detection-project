import pandas as pd # df for train and test data
import numpy as np 
import time # timing model
import sys 
import joblib # saving model and scalar
import os   
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #scale model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
# from algorithms.sorting import heap_sort


TARGET = " Label" # The column to predict
INFINITY = 1e11   # Value to replace 'inf' with

# Normalize dictionary: Groups specific attacks into broader categories
NORMALIZE = {
    "DoS slowloris": "DoS", 
    "DoS Slowhttptest": "DoS", 
    "DoS Hulk": "DoS", 
    "DoS GoldenEye": "DoS", 
    "FTP-Patator": "BruteForce", 
    "SSH-Patator": "BruteForce"
}


def read_dataset(file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        print(f"Error reading dataset '{file}': {e}")
        return pd.DataFrame()
    
def clean_dataset(df: pd.DataFrame):
    # Replace Infinity
    df = df.replace([np.inf, -np.inf, "inf", "Infinity"], INFINITY)
    
    # Drop rows with missing values
    df = df.dropna()
    for c in df.columns:
        df = df[df[c].notna()]
    df = df.reset_index(drop=True)

    # Normalize Labels (Group similar attacks)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].replace(NORMALIZE)
   
    return df

def run_test_timed(model, x, y):
    print("Running Performance Test...")
    start_detect = time.time()
    
    # Predict directly
    ypred = model.predict(x)
    
    detect_time = time.time() - start_detect
    f1s= f1_score(y, ypred, average='weighted')
    # OUTPUT
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, ypred))
    print("\nClassification Report:")
    print(classification_report(y, ypred))
    print("\n" + "="*40)
    print("FINAL METRICS REPORT")
    print("="*40)
    print(f"Total Detection Time:   {detect_time:.4f}s")
    print(f"Avg Time per Flow:      {detect_time/len(x):.8f}s")
    print("-" * 40)
    print(f"F1 Score (Weighted):    {f1s:.4f}")
    print(f"Accuracy:               {np.mean(y == ypred):.4f}")
    print("-" * 40)

    return f1s, detect_time


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py dataset.csv")
        exit(1)

    file = sys.argv[1]
    
    # Load and clean data
    df = clean_dataset(read_dataset(file))

    # Separate Features and Target
    if TARGET not in df.columns:
        print(f"Error: Target column '{TARGET}' not found.")
        exit(1)

    y = df[TARGET]
    # Select only numeric columns for training (removes IPs/Timestamps if present)
    X = df.drop(TARGET, axis=1)#.select_dtypes(include=[np.number])

    # Scale Data
    print("Scaling features...")
    scaler = StandardScaler()
    xscaled = scaler.fit_transform(X)

    # Split Data
    # 'stratify=y' ensures even tiny attacks are present in both Train and Test sets
    print("Splitting data...")
    xtrain, xtest, ytrain, ytest = train_test_split(
        xscaled, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    # Train Random Forest
    print("Training Random Forest...")
    start_train = time.time()
    
    # class_weight='balanced' fixes data imbalance.
    # It tells the model to pay huge attention to rare attacks, not the common benign.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(xtrain, ytrain) 
    
    print(f"Training Complete. Time: {time.time() - start_train:.2f}s")

    # Run Test

    f1s, _ = run_test_timed(model, xtest, ytest)

    # Save Results
    joblib.dump(model, "ids_model.pkl")
    joblib.dump(scaler, "ids_scaler.pkl")
    print("\nModel saved to ids_model.pkl")

    exit(0)

