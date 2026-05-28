#!/bin/bash

DATA_DIR=$1 # The first argument: The data directory storing trajectories.

for i in {0..9}
do   
   N=$(ls -1 $DATA_DIR/*_${i}_*.csv | wc -l)
   echo "The number of digit $i : $N" 
done
