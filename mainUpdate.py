import pandas as pd
import numpy as np
import time
import sys
import joblib 
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- CONFIGURATION ---
DATA_FILE = "/Volumes/follower/data/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
MODEL_FILENAME = "ids_model.pkl"
SCALER_FILENAME = "ids_scaler.pkl"

# --- THE FEATURE LIST  ---
# We use this to strictly filter the dataset columns
CHOSEN_COLUMNS = [
    " Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets", 
    " Total Length of Bwd Packets", " Fwd Packet Length Max", " Fwd Packet Length Min", " Fwd Packet Length Mean", 
    " Fwd Packet Length Std", "Bwd Packet Length Max", " Bwd Packet Length Min", " Bwd Packet Length Mean", 
    " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", 
    " Flow IAT Min", "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min", "Bwd IAT Total", 
    " Bwd IAT Mean", " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags", " Fwd URG Flags", 
    " Bwd URG Flags", " Fwd Header Length", " Bwd Header Length", "Fwd Packets/s", " Bwd Packets/s", " Min Packet Length", 
    " Max Packet Length", " Packet Length Mean", " Packet Length Std", " Packet Length Variance", "FIN Flag Count", 
    " SYN Flag Count", " RST Flag Count", " PSH Flag Count", " ACK Flag Count", " URG Flag Count", " CWE Flag Count", 
    " ECE Flag Count", " Down/Up Ratio", " Average Packet Size", " Avg Fwd Segment Size", " Avg Bwd Segment Size", 
    " Fwd Header Length", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk", 
    " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", " Subflow Fwd Bytes", " Subflow Bwd Packets", 
    " Subflow Bwd Bytes", "Init_Win_bytes_forward", " Init_Win_bytes_backward", " act_data_pkt_fwd", " min_seg_size_forward", 
    " Active Mean", " Active Std", " Active Max", " Active Min", "Idle Mean", " Idle Std", " Idle Max", " Idle Min"
]

# --- CORE FUNCTIONS ---

def get_majority_vote_map(kmeans_model, X_train, y_train):
    """
    The 'Bridge' Logic: Assigns the most frequent Label (Mode) to each cluster.
    """
    train_clusters = kmeans_model.predict(X_train)
    reference_df = pd.DataFrame({'cluster': train_clusters, 'label': y_train.values})
    
    mapping = {}
    for i in range(kmeans_model.n_clusters):
        subset = reference_df[reference_df['cluster'] == i]
        if not subset.empty:
            mapping[i] = subset['label'].mode()[0]
        else:
            mapping[i] = "Unknown"
    return mapping

def load_and_clean_data(filepath):
    print(f"Loading file: {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"\n[ERROR] File not found at: {filepath}")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    
    # 1. STANDARDIZE COLUMN NAMES
    # Fix the "Hidden Space" problem so we can match the list safely
    df.columns = df.columns.str.strip()
    
    # Clean the friend's list too (just in case)
    clean_feature_list = [c.strip() for c in CHOSEN_COLUMNS]
    
    # Identify Target
    target_col = "Label"
    if target_col not in df.columns:
        print(f"\n[ERROR] Could not find '{target_col}' column.")
        sys.exit(1)

    # 2. FEATURE SELECTION (Using the Friend's List)
    # We keep ONLY the columns in the list + the Target
    # This automatically drops "Flow ID", "IP", "Timestamp" etc.
    cols_to_keep = clean_feature_list + [target_col]
    
    # Safety check: Ensure all requested columns actually exist
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]

    print(f"Initial Shape: {df.shape}")

    # 3. GARBAGE REMOVAL
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 4. LABEL MAPPING (Normalize "DoS" variants)
    attack_mapping = {
        "DoS slowloris": "DoS", 
        "DoS Slowhttptest": "DoS", 
        "DoS Hulk": "DoS", 
        "DoS GoldenEye": "DoS"
    }
    df[target_col] = df[target_col].replace(attack_mapping)

    # 5. STRICT SCOPE FILTERING
    allowed_labels = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "DoS"]
    df = df[df[target_col].isin(allowed_labels)]
    
    print(f"Cleaned Shape: {df.shape}")
    print(f"Features used: {len(df.columns) - 1}") # -1 for Label
    
    return df, target_col

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # STEP 1: LOAD
    df, target_col = load_and_clean_data(DATA_FILE)

    # STEP 2: SEPARATE FEATURES AND TARGET
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    # STEP 3: SCALE (Crucial for Euclidean Distance)
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # STEP 4: SPLIT
    print("Splitting Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # STEP 5: TRAIN
    # Using 12 clusters to allow for variations within the 6 classes
    k = 12 
    print(f"Training K-Means with k={k}...")
    
    start_train = time.time()
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_train)
    print(f"Training Complete. Time: {time.time() - start_train:.2f}s")

    # STEP 6: MAP CLUSTERS TO LABELS
    print("Mapping clusters...")
    cluster_map = get_majority_vote_map(kmeans, X_train, y_train)
    print("Cluster Definitions:", cluster_map)

    # STEP 7: TEST & REPORT
    print("Running Performance Test...")
    start_detect = time.time()
    
    # A. Predict
    test_clusters = kmeans.predict(X_test)
    
    # B. Translate
    y_pred = [cluster_map[c] for c in test_clusters]
    
    detect_time = time.time() - start_detect
    
    # OUTPUT
    print("\n" + "="*40)
    print("FINAL METRICS REPORT")
    print("="*40)
    print(f"Total Detection Time:   {detect_time:.4f}s")
    print(f"Avg Time per Flow:      {detect_time/len(X_test):.8f}s")
    print("-" * 40)
    print(f"F1 Score (Weighted):    {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Accuracy:               {np.mean(y_test == y_pred):.4f}")
    print("-" * 40)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # STEP 8: SAVE
    joblib.dump(kmeans, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"\nSystem saved to {MODEL_FILENAME}")