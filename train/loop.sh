#!/bin/bash
set -e

for i in `seq 1 30`; do
    echo "epoch $i"
    j=`expr $i - 1` || true
    if [ "$i" -gt 10 ]; then
        lr=0.000001
    elif [ "$i" -gt 5 ]; then
        lr=0.00001
    else
        lr=0.0001
    fi
    echo "python train_imitation.py  --lr $lr --no-freeze-resnet --aug-factor 0.5 --dropout-factor 0.6  --epochs 1 --snapshot-weight-path ft_snapshot$j/imitation_resnet_01_weights.h5  --snapshot-path ft_snapshot$i --log-dir ft_logs$i > mylogs_ft$i.txt 2>&1" || true
    python train_imitation.py  --lr $lr --no-freeze-resnet --aug-factor 0.5 --dropout-factor 0.6  --epochs 1 --snapshot-weight-path ft_snapshot$j/imitation_resnet_01_weights.h5  --snapshot-path ft_snapshot$i --log-dir ft_logs$i > mylogs_ft$i.txt 2>&1
done
