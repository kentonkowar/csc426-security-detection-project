import pandas as pd
TARGET=" Label"
import sys
import numpy as np
df = pd.read_csv(sys.argv[1])

numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.replace([float("inf"), "inf", "Infinity", "infinity"], 0)
uniquev = numeric_df.stack().unique() # type: ignore
l = len(uniquev)
print(l)
print(max(uniquev))
# for x in range(int(len(uniquev)/4)*3, len(uniquev)):
#     print(uniquev[x])

unique_values = df[TARGET].unique()

print(unique_values)

filters = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "FTP-Patator", 
           "SSH-Patator", "DoS slowloris", "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye", "Heartbleed"]