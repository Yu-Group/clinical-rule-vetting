#!/bin/sh

#SBATCH --job-name=DataCleaningAnalysis
#SBATCH --mail-type=ALL
#SBATCH --mail-user=omer_ronen@berkeley.edu
#SBATCH -o DataCleaningAnalysis.out #File to which standard out will be written
#SBATCH -e DataCleaningAnalysis.err #File to which standard err will be written
#SBATCH -p jsteinhardt
#SBATCH -C manycore
cd /accounts/campus/omer_ronen/projects/rule-vetting
/scratch/users/omer_ronen/rule-vetting/bin/python -m rulevetting.projects.stability_cleaning
