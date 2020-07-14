#!/bin/sh

# scp ../MedSeg.sh springfield:/root/experiments/
for i in $( cat MedSeg/stuff_to_copy.txt ); do
    scp -r $i springfield:/root/experiments/MedSeg/
done
