import os , csv
from pathlib import Path

import numpy as np
filePath="/home/himanshujain/Desktop/ML_proj/FINAL_DATA.csv"

updatedfile="/home/himanshujain/Desktop/ML_proj/DATAML.csv"

def csv_write(csvFile,dataList):
	if(Path(csvFile).is_file()==False):
		with open(csvFile,"wb") as f:
			w=csv.writer(f,delimiter=",")
			w.writerow(headers)
	with open(csvFile,"a") as f:
		w=csv.writer(f,delimiter=",")
		w.writerow(dataList)
	f.close()

data = np.genfromtxt(filePath, dtype="str",delimiter=",")
header = np.matrix(data[0,4:].tolist()  + ["label"])
bench  = data[1:,0:1]
config = data[1:,1:4].astype(float)
parameters = data[1:,4:].astype(float)


row,col=np.shape(parameters)
start=[0]
end = []
label=[]
k=0

for i in range(row-1):
    if ((bench[i+1] == bench[i])==False):
       start.append(i+1)
       end.append(i)
    if (i == (row-2)):
        end.append(i+1)

for i in range(len(start)):
    for j in range(col):
        normalizedfactor= max(parameters[start[i]:end[i],j])
        for update in range(start[i],end[i]+1):
            if normalizedfactor == 0 :
                continue
            parameters[update,j] = round(parameters[update,j],4)
    for l in range(start[i],end[i]+1):
        label.append(k)
    k=k+1

X= np.concatenate((parameters,np.matrix(label).T),axis=1)
datalist  = np.concatenate((header,X),axis=0)

datalist = datalist.tolist()

csvFile=updatedfile
for i in range(len(datalist)):
    if(Path(csvFile).is_file()==False):
            with open(csvFile,"wb") as f:
                    f.close()
    with open(csvFile,"a") as f:
	    w=csv.writer(f,delimiter=",")
	    w.writerow(datalist[i])
    f.close()









