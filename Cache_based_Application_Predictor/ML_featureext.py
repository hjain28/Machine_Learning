import os,csv,re
from statistics import mean
from pathlib import Path

dataDir ="/Users/himanshujain/Desktop/ML_PROJ/RAW_DATA"
parameter=["system.cpu.ipc","system.cpu.ipc_total"]
headers=["benchmark","nsets","bsize","assoc"] + parameter #+ ["label"]

commonfile ="/Users/himanshujain/Desktop/ML_PROJ/FINAL_DATA.csv"

def csv_write(csvFile,dataList):
	if(Path(csvFile).is_file()==False):
		with open(csvFile,"wb") as f:
			w=csv.writer(f,delimiter=",")
			w.writerow(headers)
	with open(csvFile,"a") as f:
		w=csv.writer(f,delimiter=",")
		w.writerow(dataList)
	f.close()

def FeatureExtraction_nWrite(Fpath):
	bench	= Fpath.split("/")[-2]
	config  = Fpath.split("/")[-1].split(".txt")[0]
	stdoutfile = Fpath.split("/")[-1]
	csvfile  = "/Users/himanshujain/Desktop/ML_PROJ/RAW_DATA" + bench  + "_parameter.csv"
	average_param =[]
	for i in range(0,len(parameter)):
		count =[]
		f2 = open(Fpath,"r")
		for line in f2:
			if re.search(parameter[i],line):
                              count.append([x for x in line.split(" ") if x][1])
		if count == []:
			count =[0]
		average_param.append(mean(map(float,count)))
		f2.close()
	l1 = config.split("-")
	datalist = [bench]+[int(l1[0])]+ [int(l1[1])] + [int(l1[2])] + average_param
	csv_write(commonfile,datalist)


for root,directories, filenames in os.walk(dataDir):
	for filename in filenames:
		Fpath = os.path.join(root,filename)
		if (len(Fpath.split("/")[-1].split(".txt"))>1):
			FeatureExtraction_nWrite(Fpath)






		













		
