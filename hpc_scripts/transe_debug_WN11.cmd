#!/bin/sh
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
#PBS -N transe_debug_wn11
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
#PBS -l walltime=24:00:00

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


mkdir -p build-rk
cd build-rk
cmake ../../
make

train_file="../../data/WN11/train.txt"
test_file="../../data/WN11/test.txt"
valid_file="../../data/WN11/dev.txt"

opt_method="AdaGrad"
dimension="20 100"
lambdaA="0"
lambdaR="0"
margin="1 4"
step_size="0.01"

n="-1 1"
n_e="-1"

epoch="2000"
p_epoch="50"
o_epoch="2000"

counter="0"
for nt in $n; do
    for d in $dimension; do
        for m in $margin; do
            for l in $step_size; do
                for la in $lambdaA; do
                    for lr in $lambdaR; do
    OUTFILE=${PBS_JOBNAME}.${counter}.${JID}
    ./runTRANSE_NAIVE --n $nt --n_e $n_e --opt $opt_method --d $d --margin $m --step_size $l --lambdaA $la --lambdaR $lr --epoch $epoch --p_epoch $p_epoch --o_epoch $o_epoch --t_path $train_file --v_path $valid_file --e_path $test_file > ${OUTFILE}
    counter=`expr "$counter" + "1"`
                    done
                done
            done
        done
    done
done


echo "Job Finish Time is `date "+%Y/%m/%d -- %H:%M:%S"`"
mv $HOME/${PBS_JOBNAME}.e${JID} $HOME/${PBS_JOBNAME}.o${JID} $PBS_O_WORKDIR

exit 0