import pandas as pd
TARGET=" Label"
import sys
df = pd.read_csv(sys.argv[1], usecols=[TARGET])

unique_values = df[TARGET].unique()

print(unique_values)

filters = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "FTP-Patator", 
           "SSH-Patator", "DoS slowloris", "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye", "Heartbleed"]