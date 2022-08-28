# -*- encoding: utf-8 -*-
"""
@File    :   get_model.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 21:00   thgpddl      1.0         None
"""

from .models import *
import inspect

def get_model(arch='ResNet18'):
    # assert arch in models,print("{}不在model hub中，请检查是否添加至hub或模型名称是否正确".format(arch))
    try:
        model=eval(arch + '()')
        net_path=inspect.getmodule(model).__file__
        return model,net_path
    except:
        assert False, print("模型名称错误，请检查:{}".format(arch + '()'))
