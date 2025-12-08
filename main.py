from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys

DATA_FILE="/Volumes/follower/data/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
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
ATTACK_ITEMS_SMALL = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "DoS"]
ATTACK_ITEMS = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "FTP-Patator", 
           "SSH-Patator", "DoS slowloris", "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye", "Heartbleed"]
REFINED=False
# normalize data
NORMALIZE={"DoS slowloris": "Dos", "DoS Slowhttptest": "DoS", "DoS Hulk": "DoS", "DoS GoldenEye": "DoS"}

# number of clusters to use
CLUSTER_COUNT = len(ATTACK_ITEMS_SMALL) if not REFINED else len(ATTACK_ITEMS)
TRAIN_TEST_P = 0.2 # proportion testing data
TESTING = True

# manipulate dataset
def read_dataset(file: str) -> pd.DataFrame:
    """
    Read a dataset from CSV (or other formats if needed).
    """
    try:
        df = pd.read_csv(file)
        if TESTING:
            print(CLUSTER_COUNT)
            global CLUSTER_COUNT
            CLUSTER_COUNT = len(df[TARGET].unique())
            print(CLUSTER_COUNT)
        return df
    except Exception as e:
        print(f"Error reading dataset '{file}': {e}")
        return pd.DataFrame()
def clean_dataset(df: pd.DataFrame):
    # remove inf
    df = df.replace([float("inf"), "inf", "Infinity", "infinity"], sys.float_info.max)
    # remove nan
    df = df.replace(np.nan, -2)

    # reduce label requirements
    if not REFINED:
        df[TARGET] = df[TARGET].replace(NORMALIZE)
        df = df[df[TARGET].isin(ATTACK_ITEMS_SMALL)]

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


if len(sys.argv) < 2:
    print("Usage: main.py dataset.csv")
    exit(1)

file = sys.argv[1]
df = read_dataset(file)
df = clean_dataset(df)
y=df[TARGET]
X=df.drop(TARGET, axis=1)
xtrain, xtest, ytrain, ytest= train_test_split(df, test_size=0.2, random_state=42, shuffle=True)


# Run K-means
kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init="auto", random_state=42)
kmeans.fit(xtrain) 
labels = kmeans.predict(xtest) # to change
inertia = kmeans.inertia_ # to change


# # Save results
# df.to_csv("output_with_clusters.csv", index=False)
# print("\nResults written to output_with_clusters.csv")

exit(0)