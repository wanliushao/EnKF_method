#This program test MPI-EnKF algorithm
#Right now it just for sigle grid, using LAI and also not for stochastic prior parameters
#Update 08/03/2017 Shaoqing
from mpi4py import MPI
from numpy.linalg import inv
from datetime import date
import numpy as np
import numpy.matlib
import os
from netCDF4 import Dataset
import glob
import scipy.stats as stats

#print("hello world")
comm=MPI.COMM_WORLD
rank=comm.Get_rank()

recv_data = None
file1='/home/sqliu/software_install/cesm1_2_2/scripts/Mojave_2x2_syntest'
file2='/Mojave_2x2_syntest_trans_'
file3='/rundir'
command1='./Mojave_2x2_syntest_trans_'
command2='.run'

EnKF_size=100;
EnKF_number=6;
npros=11; #mpirun -np np programe
EnKF_time=85; #EnKF time steps
per_jobs=EnKF_size/(npros-1); #the number of jobs for each processor to handle

if __name__=="__main__":
	def main():
		if (rank>0):
			for m in range(1,EnKF_time):
				subpro(rank);
				comm.Barrier();
		elif (rank==0):
			data=np.arange(npros)
			lai_sim=np.zeros((EnKF_time,EnKF_size))
			lai_obs=np.empty(0);
			sm1_obs=np.empty(0);
			sm2_obs=np.empty(0);
			for line in open('lai_sm_syntest_14years.txt'):
				line=line.rstrip('\n');
				lai_obs=np.append(lai_obs,line.split(" ")[0]);
				sm1_obs=np.append(lai_obs,line.split(" ")[1]);
				sm2_obs=np.append(lai_obs,line.split(" ")[2]);
			print("process {} scatter data {} to other processes".format(rank, data))
			#begin EnKF
			for m in range(1,EnKF_time+1):#this will be time step in future
				#change run_days first
				for j in range(1,EnKF_size+1):
					file=file1+file2+format(j, '003')
					os.chdir(file);

				for i in range(1,npros):
					comm.Send(data[i-1], dest=i, tag=11)
				comm.Barrier();
				for j in range(1,EnKF_size+1):
					file=file1+file2+format(j, '003')
					os.chdir(file)
					if (m==1):
						cmdstring='sed -i "79s/FALSE/TRUE/" env_run.xml'
						os.system(cmdstring)
				observation=np.matlib.zeros((3,1));
				observation[0,0]=lai_obs[m-1];
				observation[1,0]=sm1_obs[m-1];
				observation[2,0]=sm2_obs[m-1];
				observation=observation.astype(np.float);		
				#get LAI simulations and SLA, Nm and RootD parameters
				Y_f=Get_simulation(EnKF_number, EnKF_size);
				logfid=open('/home/sqliu/software_install/cesm1_2_2/scripts/Mojave_2x2_syntest/log','a');

				Y_update=EnKF(observation,Y_f,EnKF_size,EnKF_number)
				for k in range(EnKF_size):
					line=str(Y_update[0,k])+'\t'+str(Y_update[1,k])+'\t'+str(Y_update[2,k])+'\t'+str(observation[0,0])+'\t'+str(observation[1,0])+'\t'+str(observation[2,0])+'\n';
					logfid.write(line)
				logfid.close();
				#Update paras and states;assign new paras and ensembles to each folder(ensemble member)
				Update(Y_update, EnKF_size)				
				#EnKF for one step dona
				#comm.Barrier();
		MPI.Finalize()

	def subpro(rank):
		data=np.empty(1);
		comm.Recv(data, source=0, tag=11)
		for i in range(1,per_jobs+1):
			indx=(rank-1)*per_jobs+i;
			file=file1+file2+format(indx, '003')
			print("process {} deal with directory{}".format(rank, file))
			os.chdir(file)
			cmdstring=command1+format(indx, '003')+command2
			os.system(cmdstring)

	def Get_simulation(EnKF_number,EnKF_size):
		Y_f=np.matlib.zeros((EnKF_number,EnKF_size))
		for j in range(1,EnKF_size+1):
			file=file1+file2+format(j, '003')+file3
			os.chdir(file)
			newest = max(glob.iglob('*clm2.r.*nc'), key=os.path.getctime)
			ncfid=Dataset(newest,'r');
			LAI=ncfid.variables['tlai'][9]*0.15; #test one grid first, at the 8th day
			SM1=ncfid.variables['H2OSOI_LIQ'][0,8];#unit kg/m2, upper 5 layers are snow
			SM2=ncfid.variables['H2OSOI_LIQ'][0,10];#unit kg/m2, upper 5 layers are snow
			ncfid.close()
			Y_f[3,j-1]=LAI;
			Y_f[4,j-1]=SM1;#ignore SM error first
			Y_f[5,j-1]=SM2;#ignore SM error first
			pft_file='/home/sqliu//software_install/cesm1_2_2/Mojave_input/lnd/clm2/pftdata/pft-physiology_constant_allocation_'+str(j)+'.nc'
			ncfid=Dataset(pft_file,'r');
			Y_f[0,j-1]=ncfid.variables['slatop'][9];
			Y_f[2,j-1]=ncfid.variables['rootb_par'][9];
			Y_f[1,j-1]=ncfid.variables['leafcn'][9];
		return (Y_f)

	def EnKF(observation,simulation, EnKF_size, EnKF_number):
		observation_number=3;
		H=np.matlib.zeros((observation_number,EnKF_number));
		H[0,3]=1;
		H[1,4]=1;
		H[2,5]=1;
		R=np.matlib.zeros((observation_number,observation_number));
		error=np.array([0.1,3,8])
		R[0,1]=R[1,0]=0.03;
		R[0,2]=R[2,0]=0.04;
		R[1,2]=R[2,1]=20;
		np.fill_diagonal(R,np.square(error));
		f_mean=np.asmatrix(np.mean(simulation,axis=1));
		Ensemble_dev=simulation-np.repeat(f_mean,EnKF_size,axis=1);
		Pb=np.dot(Ensemble_dev,np.transpose(Ensemble_dev))/(EnKF_size-1);
		temp1=np.dot(np.dot(H,Pb),np.transpose(H))+ R;
		temp2=np.dot(Pb,np.transpose(H));
		K=np.dot(temp2,inv(temp1));
		u_mean=f_mean + np.dot(K,(observation-np.dot(H,f_mean)));

		temp1=1/(np.sqrt(temp1)) #size 1x1
		K_p=np.dot(temp2,np.transpose(temp1)) #size (EnKF_number-observation_number)xobservation_number
		K_p=np.dot(K_p,inv(inv(temp1)+np.sqrt(R))); #size (EnKF_number-observation_number)xobservation_number
		obs_perturb=np.zeros((1,EnKF_size));
		temp1=simulation-np.repeat(f_mean,EnKF_size,axis=1);#(EnKF_number-1)x(EnKF_size)
		EnKF_perturb=temp1+np.dot(K_p,(obs_perturb-np.dot(H,temp1)))#size (EnKF_number-1)xEnKF_size
		Y_update=np.repeat(u_mean,EnKF_size,axis=1)+EnKF_perturb;#size (EnKF_number-1)xEnKF_size igma=0.1*1.2;
		return (Y_update)

	def Update(updates,EnKF_size):
		for j in range(1,EnKF_size+1):
			file=file1+file2+format(j, '003')+file3
			os.chdir(file)
			newest = max(glob.iglob('*clm2.r.*nc'), key=os.path.getctime) #this is updated lai, CLM start next ensemble run based on this restart file
			ncfid=Dataset(newest,'r+');
			ncfid.variables['tlai'][9]=updates[3,j-1]/0.15; #BES fraction in one grid cell 15%
			ncfid.variables['H2OSOI_LIQ'][0,8]=updates[4,j-1];
			ncfid.variables['H2OSOI_LIQ'][0,10]=updates[5,j-1]
			ncfid.close();
			pft_file='/home/sqliu//software_install/cesm1_2_2/Mojave_input/lnd/clm2/pftdata/pft-physiology_constant_allocation_'+str(j)+'.nc'
			ncfid=Dataset(pft_file,'r+');
			ncfid.variables['slatop'][9]=updates[0,j-1];
			ncfid.variables['rootb_par'][9]=updates[2,j-1];
			ncfid.variables['leafcn'][9]=updates[1,j-1];
			ncfid.close();
	main();
