import pandas as pd # df for train and test data
import numpy as np 
import time # timing model
import sys 
import joblib # saving model and scalar
import os   
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #scale model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer # handle nan values

OPTIONS={}
COLUMNS=[" Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets", 
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
         "Active Mean", " Active Std", " Active Max", " Active Min", "Idle Mean", " Idle Std", " Idle Max", " Idle Min"]
TARGET=" Label"
# label groupings used
ATTACK_ITEMS_SMALL = ['BENIGN', 'Infiltration']#["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "DoS"]
ATTACK_ITEMS = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "FTP-Patator", 
           "SSH-Patator", "DoS slowloris", "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye", "Heartbleed", "DoS"]
REFINED=False
INFINITY = 1e11
# normalize data
NORMALIZE={"DoS slowloris": "Dos", "DoS Slowhttptest": "DoS", "DoS Hulk": "DoS", "DoS GoldenEye": "DoS"}
x = 0
LABEL_MAP_INT = {}
for i in ATTACK_ITEMS:
    LABEL_MAP_INT[i]= x
    x += 1

# number of clusters to use
global CLUSTER_COUNT
CLUSTER_COUNT = len(ATTACK_ITEMS_SMALL) if not REFINED else len(ATTACK_ITEMS)
CLUSTER_COUNT += 1
TRAIN_TEST_P = 0.2 # proportion testing data
TESTING = True

# manipulate dataset
def read_dataset(file: str) -> pd.DataFrame:
    """
    Read a dataset from CSV (or other formats if needed).
    """
    try:
        df = pd.read_csv(file)
        # if TESTING:
            # print(CLUSTER_COUNT)
            # print(len(df[TARGET].unique()))
        return df
    except Exception as e:
        print(f"Error reading dataset '{file}': {e}")
        return pd.DataFrame()
    
def clean_dataset(df: pd.DataFrame):
    # remove inf
    df = df.replace([float("inf"), "inf", "Infinity", "infinity"], INFINITY)
    # remove nan
    df = df.dropna()
    for c in df.columns:
        df = df[df[c].notna()]
    df = df.reset_index(drop=True)

    # reduce label requirements
    if not REFINED:
        df[TARGET] = df[TARGET].replace(NORMALIZE)
        df = df[df[TARGET].isin(ATTACK_ITEMS_SMALL)]
    # df[TARGET] = df[TARGET].replace(LABEL_MAP_INT)

    # df = df.replace([np.inf, -np.inf], np.nan).dropna()

    #split numeric columns
   
    return df

def kmeans_clustering(x: pd.DataFrame, y: pd.DataFrame, k: int = 3):
    if x.empty or y.empty: # error check
        print("Dataset is empty — cannot run K-means.")
        return None

    # Use only numeric columns for clustering
    numeric_df = x.select_dtypes(include=["float64", "int64"])
    if numeric_df.empty:
        print("No numeric columns found for K-means.")
        return None

    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(numeric_df)

    df["cluster"] = labels
    return df, model

def get_majority_vote_map(kmeans_model, X_train, y_train):
    # The 'Bridge' Logic: Assigns the most frequent Label (Mode) to each cluster.

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

def run_test_timed(model, x, y, cluster_map):
    print("Running Performance Test...")
    start_detect = time.time()
    
    # A. Predict
    test_clusters = model.predict(x)

    # B. Translate
    ypred = [cluster_map[c] for c in test_clusters]
    
    detect_time = time.time() - start_detect

    # OUTPUT
    print("\n" + "="*40)
    print("FINAL METRICS REPORT")
    print("="*40)
    print(f"Total Detection Time:   {detect_time:.4f}s")
    print(f"Avg Time per Flow:      {detect_time/len(x):.8f}s")
    print("-" * 40)
    print(f"F1 Score (Weighted):    {f1_score(y, ypred, average='weighted'):.4f}")
    print(f"Accuracy:               {np.mean(y == ypred):.4f}")
    print("-" * 40)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, ypred))
    print("\nClassification Report:")
    print(classification_report(y, ypred))

    return f1_score(y, ypred, average='weighted'), detect_time


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py dataset.csv")
        exit(1)

    file = sys.argv[1]
    df = clean_dataset(read_dataset(file))

    y=df[TARGET]
    X=df.drop(TARGET, axis=1)

    # scalar for data
    scaler = StandardScaler()
    xscaled=scaler.fit_transform(X)

    # X_scaled, y, test_size=0.2, random_state=42, stratify=y
    xtrain, xtest, ytrain, ytest= train_test_split(xscaled, y, test_size=0.2, random_state=42, shuffle=True)

    # Run K-means
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init="auto", random_state=42)
    start_train = time.time() # time and train model
    kmeans.fit(xtrain) 
    print(f"Training Complete. Time: {time.time() - start_train:.2f}s")

    cluster_map = get_majority_vote_map(kmeans, xtrain, ytrain)
    print("cluster definitions ", cluster_map)

    run_test_timed(kmeans, xtest,ytest, cluster_map)

    # # Save results
    # df.to_csv("output_with_clusters.csv", index=False)
    # print("\nResults written to output_with_clusters.csv")

    exit(0)

"""
for imputing data
 numeric_df = df.select_dtypes(include=[np.number])
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    print(numeric_df.shape)
    print(non_numeric_df.shape)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    numeric_imputed = pd.DataFrame(imp.fit_transform(numeric_df), columns=numeric_df.columns)
    df = pd.concat([numeric_imputed, non_numeric_df], axis=1)
"""