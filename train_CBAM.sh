#!/bin/bash

# 修改权限：chmod +x ./test.sh

# CABAN+inception

if [ -n "$1" ] #存在参数
then
    epoch=$1
    echo ">>epoch is $epoch"
else
    echo "没有epoch参数，比如:./train.sh 300"
    exit 8
fi


 python train.py --name=unresnet_noInit_inception --arch=unresnet_noInit_inception --epochs=${epoch}
 python train.py --name=unresnet_noInit_CS --arch=unresnet_noInit_CS --epochs=${epoch}
 python train.py --name=unresnet_noInit_C1 --arch=unresnet_noInit_C1 --epochs=${epoch}
 python train.py --name=unresnet_noInit_C2 --arch=unresnet_noInit_C2 --epochs=${epoch}
 python train.py --name=unresnet_noInit_C3 --arch=unresnet_noInit_C3 --epochs=${epoch}
 python train.py --name=unresnet_noInit_C4 --arch=unresnet_noInit_C4 --epochs=${epoch}

 python train.py --name=unresnet_noInit_S1 --arch=unresnet_noInit_S1 --epochs=${epoch}
 python train.py --name=unresnet_noInit_S2 --arch=unresnet_noInit_S2 --epochs=${epoch}
 python train.py --name=unresnet_noInit_S3 --arch=unresnet_noInit_S3 --epochs=${epoch}
 python train.py --name=unresnet_noInit_S4 --arch=unresnet_noInit_S4 --epochs=${epoch}

