##RAW DATA PLOT   

import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd

from pandas.tools.plotting import scatter_matrix

#data extraction
header = ["gpu_ipc","L1IA","L1IM","L1IMR","L1IRF","L1DA","L1DM","L1DMR","L1DRF"]

 
data=np.genfromtxt("FINAL_DATA.csv", dtype="str",delimiter=",")
rawD= data[1:,4:13].astype(float)
raw_df =pd.DataFrame(rawD,columns=header)
scatter_matrix(raw_df, alpha=1, figsize=(9, 9), diagonal='kde')
plt.savefig('foo.png', bbox_inches='tight')
plt.show()


dataN=np.genfromtxt("DATAML.csv", dtype="str",delimiter=",")
rawDN= dataN[1:,4:13].astype(float)
rawN_df =pd.DataFrame(rawDN,columns=header)
scatter_matrix(rawN_df, alpha=1, figsize=(9, 9), diagonal='kde')
plt.savefig('goo.png', bbox_inches='tight')
plt.show()
