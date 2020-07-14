# ssh springfield rm experiments/MedSeg/datasets/preprocessed_2d/train/slices/*
# ssh springfield rm experiments/MedSeg/datasets/preprocessed_2d/train/labels_liver/*
# ssh springfield rm experiments/MedSeg/datasets/preprocessed_2d/train/labels_lesion/*
# ssh springfield rm experiments/MedSeg/datasets/preprocessed_2d/test/slices/*
# ssh springfield rm experiments/MedSeg/datasets/preprocessed_2d/test/labels_liver/*
# ssh springfield rm experiments/MedSeg/datasets/preprocessed_2d/test/labels_lesion/*
# scp MedSeg/datasets/preprocessed_2d/train/slices/* springfield:/root/experiments/MedSeg/datasets/preprocessed_2d/train/slices/
# scp MedSeg/datasets/preprocessed_2d/test/slices/* springfield:/root/experiments/MedSeg/datasets/preprocessed_2d/test/slices/
scp MedSeg/datasets/preprocessed_2d/train/labels_liver/* springfield:/root/experiments/MedSeg/datasets/preprocessed_2d/train/labels_liver/
scp MedSeg/datasets/preprocessed_2d/test/labels_liver/* springfield:/root/experiments/MedSeg/datasets/preprocessed_2d/test/labels_liver/
scp MedSeg/datasets/preprocessed_2d/train/labels_lesion/* springfield:/root/experiments/MedSeg/datasets/preprocessed_2d/train/labels_lesion/
scp MedSeg/datasets/preprocessed_2d/test/labels_lesion/* springfield:/root/experiments/MedSeg/datasets/preprocessed_2d/test/labels_lesion/
