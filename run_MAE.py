import numpy
import os, glob

# Data address
workdir="/home/xqi10/PythonProjects/FBA_run_first_part"
datadir_st_non=os.path.join(workdir,'FBA_MAE_data', 'standard','nonreoriented')
datadir_sp_non=os.path.join(workdir,'FBA_MAE_data', 'specific','nonreoriented')
datadir_st_reo=os.path.join(workdir,'FBA_MAE_data', 'standard','reoriented')
datadir_sp_reo=os.path.join(workdir,'FBA_MAE_data', 'specific','reoriented')

outdir_st_non=os.path.join(workdir,'FBA_MAE_results', 'standard','nonreoriented')
outdir_sp_non=os.path.join(workdir,'FBA_MAE_results', 'specific','nonreoriented')
outdir_st_reo=os.path.join(workdir,'FBA_MAE_results', 'standard','reoriented')
outdir_sp_reo=os.path.join(workdir,'FBA_MAE_results', 'specific','reoriented')

data_st_non=glob.glob(os.path.join(datadir_st_non,'*')) # datalist
data_sp_non=glob.glob(os.path.join(datadir_sp_non,'*'))
data_st_reo=glob.glob(os.path.join(datadir_st_reo,'*'))
data_sp_reo=glob.glob(os.path.join(datadir_sp_reo,'*'))

def generate_comparison_pairs(list_data):
    """
    Input: list of folders that contains fixel files
    Output: list of tuples that contains pairs to be compared.
    """
    from itertools import combinations
    comb=combinations(list_data,2)
    
    return list(comb)

def write_cmdline(cmdline, qscript):
	f=open(qscript,'a')
	f.write(cmdline + '\n')
	f.close

def write_job_list(jobname, joblist):
	f=open(joblist,'a')
	f.write(jobname+'\n')
	f.close

def write_qscript(data_list, outdir):
	
	pairs_list=generate_comparison_pairs(data_list)
	

	job_list=str(outdir + '/jobs' + '/job_list.txt')
	if os.path.exists(job_list):
		os.remove(job_list)
	
	for i in range(len(pairs_list)):
		
		dir_path1=pairs_list[i][0]
		dir_path2=pairs_list[i][1]

		dir_id1=os.path.split(dir_path1)[-1]
		dir_id2=os.path.split(dir_path2)[-1]
		
		pair_name=str(str(dir_id1)+str(dir_id2))

		outputpath=str(outdir + '/' + pair_name)

		qscript=str(outdir + '/jobs' + '/job_' + pair_name + '_qsub.sh')
		if os.path.exists(qscript):
			os.remove(qscript)
		cmdline=str('python ' + workdir + '/calc_MAE.py '+ dir_path1 + ' ' + dir_path2 + ' ' + outputpath)
		cmdsource=str('source activate clusterneuroimaging')

		write_cmdline(cmdsource, qscript)
		write_cmdline(cmdline, qscript)
		write_job_list(pair_name, job_list)

write_qscript(data_st_non, outdir_st_non)
write_qscript(data_sp_non, outdir_sp_non)
write_qscript(data_st_reo, outdir_st_reo)
write_qscript(data_sp_reo, outdir_sp_reo)


