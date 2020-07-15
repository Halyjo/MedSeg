#!/bin/sh
# curl https://bootstrap.pypa.io/get-pip.py | python3

cd MedSeg/
python controller2d.py --mode train --focus lesion --seed 0 --runid 3002
# python controller3d.py --mode train --focus liver --seed 0 --runid 1003
