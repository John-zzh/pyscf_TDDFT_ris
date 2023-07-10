#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
echo $basepath
for i in $(find ./ -type d -printf "%n %p\n" | grep '^2 '| cut -d " " -f 2)
do
cd $i
echo $i
sbatch *.slurm
cd $basepath
done
