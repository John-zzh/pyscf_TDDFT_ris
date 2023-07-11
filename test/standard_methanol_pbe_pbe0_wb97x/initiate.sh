#!/bin/bash



for KS in RKS UKS
do
    for func in pbe pbe0 wb97x
    do
        for calc in TDA TDDFT
        do
        dir="$KS"_"$func"_"$calc"
        echo $dir
        mkdir $dir
        cp coord define.inp escf.slurm $dir
        cd $dir
        if [ $KS == 'RKS' ];then
            sed -i 's/CHARGE/0/g'  define.inp
            
            if [ $calc == 'TDA' ];then
                sed -i "s/CALCULATION/ciss/g"  define.inp
            else
                sed -i "s/CALCULATION/rpas/g"  define.inp
            fi
        else
            sed -i 's/CHARGE/1/g'  define.inp
            if [ $calc == 'TDA' ];then
                sed -i "s/CALCULATION/ucis/g"  define.inp
            else
                sed -i "s/CALCULATION/urpa/g"  define.inp
            fi
        fi
        
        sed -i "s/FUNCTIONAL/$func/g"  define.inp
        
        if [ $func == 'pbe' ];then
            sed -i 's/METHOD/as/g'  escf.slurm
        else
            sed -i 's/METHOD/ris/g'  escf.slurm
        fi
        
        cd ..
        done
    done
done