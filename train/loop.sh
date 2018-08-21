#!/bin/bash

for i in `seq 1 50`; do
    echo "epoch $i"
    j=`expr $i - 1`
    if [ "$i" -gt 40 ]; then
        lr=0.000001
    elif [ "$i" -gt 30 ]; then
        lr=0.00001
    elif [ "$i" -gt 20 ]; then
        lr=0.00005
    elif [ "$i" -gt 10 ]; then
        lr=0.0001
    else
        lr=0.0002
    fi
    if [ "$i" -eq 1 ]; then
        prev=""
    else
        prev="--snapshot ./snapshots$j/imitation_resnet_01.h5"
    fi
    echo "python train_imitation.py --gpu-fraction 0.68 --lr $lr --epochs 1 $prev --snapshot-path ./snapshots$i --log-dir logs$i > mylogs$i.txt 2>&1"
    python train_imitation.py  --gpu-fraction 0.68 --lr $lr --epochs 1 $prev --snapshot-path ./snapshots$i --log-dir logs$i > mylogs$i.txt 2>&1
done
