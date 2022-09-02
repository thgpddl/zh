#!/bin/bash

# 修改权限：chmod +x ./test.sh

if [ -n "$1" ] #存在参数
then
    epoch=$1
    echo ">>epoch is $epoch"
else
    echo "没有epoch参数，比如:./train.sh 300"
    exit 8
fi

python train.py --name=unresnet_noInit --arch=unresnet_noInit --epochs=${epoch}
#python train.py --name=unresnet_noInit_eca_all --arch=unresnet_noInit_eca_all --epochs=${epoch}
#python train.py --name=unresnet_noInit_eca_b1 --arch=unresnet_noInit_eca_b1 --epochs=${epoch}
#python train.py --name=unresnet_noInit_eca_b2 --arch=unresnet_noInit_eca_b2 --epochs=${epoch}
#python train.py --name=unresnet_noInit_eca_b3 --arch=unresnet_noInit_eca_b3 --epochs=${epoch}
#python train.py --name=unresnet_noInit_eca_b4 --arch=unresnet_noInit_eca_b4 --epochs=${epoch}

