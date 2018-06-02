#!/usr/bin/python3

file_prefix = 'Rescal_lock_wn11_'

i = 0;
for lamdaA in [0,0.1,0.01]:
    for lamdaR in [0,0.1,0.01]:
        for margin in [1,2,4,8]:
            for step in [0.01,0.1,1]:
                file_name = file_prefix+i+'.cmd';
                i=i+1;
                command = '''#!/bin/sh
###############################################################################
###  This is a sample PBS job script for Serial C/F90 program                 #
###  1. To use GNU Compilers (Default)					      #
###     gcc hello.c -o hello-gcc					      #
###     gfortran hello.f90 -o hello-gfortran				      #
###  2. To use PGI Compilers						      #
###     module load pgi							      #
###     pgcc hello.c -o hello-pgcc					      #
###     pgf90 hello.f90 -o hello-pgf90					      #
###  3. To use Intel Compilers						      #
###     module load intel					              #
###     icc hello.c -o hello-icc        				      #
###     ifort hello.f90 -o hello-ifort					      #
###############################################################################

### Job name
#PBS -N rescal_naive16
### Declare job non-rerunable
#PBS -r n
#PBS -k oe

###  Queue name (debug, parallel or fourday)   ################################
###    Queue debug   : Walltime can be  00:00:01 to 00:30:00                  # 
###    Queue parallel: Walltime can be  00:00:01 to 24:00:00                  #
###    Queue fourday : Walltime can be  24:00:01 to 96:00:00                  #
###  #PBS -q parallel                                                         #
###############################################################################
#PBS -q parallel

###  Wall time required. This example is 30 min  ##############################
###  #PBS -l walltime=00:30:00                   			      #
###############################################################################
#PBS -l walltime=18:00:00                   

###  Number of node and cpu core  ############################################# 
###  For serial program, 1 core is used					      #
###  #PBS -l nodes=1:ppn=1						      #
###############################################################################
#PBS -l nodes=1:ppn=20

###############################################################################
#The following stuff will be executed in the first allocated node.            #
#Please don't modify it                                                       #
#                                                                             #
echo $PBS_JOBID : `wc -l < $PBS_NODEFILE` CPUs allocated: `cat $PBS_NODEFILE`
PATH=$PBS_O_PATH
JID=`echo ${PBS_JOBID}| sed "s/.hpc2015-mgt.hku.hk//"`
###############################################################################

echo ===========================================================
echo "Job Start Time is `date "+%Y/%m/%d -- %H:%M:%S"`"


### Run the parallel MPI executable "a.out"
cd $PBS_O_WORKDIR
OUTFILE=${PBS_JOBNAME}.${JID}

mkdir -p build-rk
cd build-rk
cmake ../../
make

train_file="../../data/WN18/wordnet-mlj12-train.txt"
test_file="../../data/WN18/wordnet-mlj12-test.txt"
valid_file="../../data/WN18/wordnet-mlj12-valid.txt"

opt_method="AdaGrad"
dimension="200"
lambdaA="0.1"
lambdaR="0.01"
margin="1"
step_size="0.1"

n="-1"
n_e="-1"

epoch=2000
p_epoch=2000
o_epoch=2000

./runRESCAL_NAIVE --n $n --n_e $n_e --opt $opt_method --d $dimension --margin $margin --step_size $step_size --lambdaA $lambdaA --lambdaR $lambdaR --epoch $epoch --p_epoch $p_epoch --o_epoch $o_epoch --t_path $train_file --v_path $valid_file --e_path $test_file > ${OUTFILE}

echo "Job Finish Time is `date "+%Y/%m/%d -- %H:%M:%S"`"
mv $HOME/${PBS_JOBNAME}.e${JID} $HOME/${PBS_JOBNAME}.o${JID} $PBS_O_WORKDIR

exit 0
'''
                with open (file_name,'w') as f:
                    f.write()