import os  
import math 


BENCH = ["astar","cactusADM"]   ##  a2time01", "bitmnpp01", "cacheb01", "bzip2", "libquantum", "mcf"] # select your benchmark and remove the unwanted one 

#os.system("rm -r /home/himanshujain/gem5/hw2out")				#give path for output directory file
#os.system(" mkdir /home/himanshujain/gem5/hw2out")

os.system("mkdir /home/himanshujain/gem5/MLProjD")
os.chdir("/home/himanshujain/gem5/MLProjD")



def CMND(bench,LnSz,AsSz,BkSz):
	cmnd = ("/home/himanshujain/gem5/gem5/build/ARM/gem5.fast --stats-file=/home/himanshujain/gem5/MLProjD/{}/{}-{}-{}.txt  \
					--dump-config=/home/himanshujain/gem5/MLProjD/{}/config{}-{}-{}.ini /home/himanshujain/gem5/gem5/configs/example/se.py  \
					--caches --l1i_size={}kB --l1i_assoc={} --l1d_size={}kB --l1d_assoc={} --cacheline_size={}  \
					--cpu-clock=1.6GHz --mem-size=8192MB --cpu-type=O3_ARM_v7a_3 -n 1 --maxinsts=10000000 --b {}").format(bench, LnSz, AsSz, BkSz,\
					bench, LnSz, AsSz, BkSz, BkSz, AsSz, BkSz, AsSz, LnSz, bench) 
	os.system(cmnd)




for bench in BENCH:
	os.system("mkdir /home/himanshujain/gem5/MLProjD/"+bench)
	for line in [16,32,64]:
		for Assc in  [1,2,4]:
			if Assc==1 :
				for bank in [8,16,32] :
					CMND(bench,line,Assc,bank)
			elif Assc ==2:
				for bank  in [16,32]:
					CMND(bench,line,Assc,bank)
			elif Assc ==4:
				bank = 32
				CMND(bench,line,Assc,bank)

			
