# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 20:48   thgpddl      1.0         None
"""
# TODO：新增mdoel时，需要在这里导入并在arch中添加

from .unresnet_noInit import unresnet_noInit
from .unresnet_noInit_eca import unresnet_noInit_eca_all,unresnet_noInit_eca_b1,unresnet_noInit_eca_b2,unresnet_noInit_eca_b3,unresnet_noInit_eca_b4
from .unresnet_noInit_CMAB import unresnet_noInit_CS,unresnet_noInit_C1,unresnet_noInit_C2,unresnet_noInit_C3,unresnet_noInit_C4,unresnet_noInit_S1,unresnet_noInit_S2,unresnet_noInit_S3,unresnet_noInit_S4
from .unresnet_noInit_inception import unresnet_noInit_inception
from .resnet import ResNet18


