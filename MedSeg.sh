#!/bin/sh
# curl https://bootstrap.pypa.io/get-pip.py | python3

cd MedSeg/
python controller2d.py \
    --mode test \
    --focus liver \
    --seed 0 \
    --runid 3005 \
    --label_type pixelcount
# python controller3d.py --mode train --focus liver --seed 0 --runid 1003
