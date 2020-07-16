for i in $( cat MedSeg/stuff_to_copy.txt ); do
    scp -r MedSeg/$i springfield:/root/experiments/MedSeg/
done
scp MedSeg.sh springfield:/root/experiments/
kubectl job run MedSeg.yaml
