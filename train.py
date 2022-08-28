import argparse
import time
import warnings
import datetime
from torch.cuda.amp import GradScaler

from clearml import Task, Logger

import os
import torch

# 加载数据就和处理
from utils.dataset import get_dataloaders

# 加载其他工具类：保存函数、写日志函数、损失函数等等
from utils.loops import train, evaluate, test
from utils.opti import get_opti, get_loss_fn
from utils.utils import load_yaml, load_checkpoint, random_seed, save_checkpoint
from utils.getmodel import get_model

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='EFR Project')
parser.add_argument('--name', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--epochs', default=300, type=int)

# clearml
cm= False

def main():
    # 获取配置信息
    config = load_yaml("utils/config.yaml", parser.parse_args())
    if cm:
        task = Task.init(project_name="HP-Fer2013-zh", task_name=config["arch"])  # 负责记录输入输出
        logger = Logger.current_logger()  # 显式报告 包括标量图、线图、直方图、混淆矩阵、2D 和 3D 散点图、文本日志记录、表格以及图像上传和报告

    # 定义本次输出的根目录和checkpoint目录
    root_output_path = os.path.join(config['outputs'],
                                    config['name'],
                                    datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S"))
    checkpoint_path = os.path.join(root_output_path, 'checkpoints')
    if not os.path.exists(checkpoint_path):  # 创建最深的路径，上级路径也就有了
        os.makedirs(checkpoint_path)

    # 随机种子
    if random_seed:
        random_seed(config['seed'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(path=config['data_path'],
                                                            bs=config['batch_size'],
                                                            num_workers=config['num_workers'],
                                                            augment=True)

    model,net_path = get_model(arch=config['arch'])
    if cm:
        task.upload_artifact(name="net path",artifact_object=net_path)  # 记录定义net的py文件
    model = model.to(device)

    # amp
    scaler = GradScaler()

    optimizer, scheduler = get_opti(config, model)

    # 获取损失函数
    loss_fn = get_loss_fn(config)

    # resume
    if config['resume_path']:
        model, optimizer = load_checkpoint(config['resume_path'], model, optimizer)
        print("加载checkpoint成功：", config['resume_path'])

    print("Epoch \t Time \t Train Loss \t Train ACC \t Val Loss \t Val ACC")
    best_acc = 0
    for epoch in range(1, config['epochs'] + 1):
        start_t = time.time()
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device, scaler, config)
        val_loss, val_acc = evaluate(model, val_loader, device, config)

        if config['scheduler'] == 'cos':
            scheduler.step()
        elif config['scheduler'] == 'reduce':
            scheduler.step(val_acc)
        if cm:
            logger.report_scalar(title='Loss', series='Train', value=train_loss, iteration=epoch)
            logger.report_scalar(title='Loss', series='Valid', value=val_loss, iteration=epoch)
            logger.report_scalar(title='Accuracy', series='Train', value=train_acc, iteration=epoch)
            logger.report_scalar(title='Accuracy', series='Valid', value=val_acc, iteration=epoch)

        # 当前最好
        note = ""
        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            if cm:
                logger.report_scalar(title='Best Accuracy', value=best_acc, iteration=epoch)
            best_checkpoint_filename = os.path.join(checkpoint_path,
                                                    "e" + str(epoch) + "_" + str(best_acc) + '_best_checkpoint.tar')
            state_dict = {'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'opt_state_dict': optimizer.state_dict()}
            save_checkpoint(state_dict, best_checkpoint_filename)
            note = "saves best"
        print("%d\t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %s" % (
            epoch,
            time.time() - start_t,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            note))
    print("Best val ACC %.4f" % best_acc)

    # 进行test测试
    print("--------------------------------------------------------")
    model, _ = load_checkpoint(best_checkpoint_filename, model, optimizer)
    print("加载best_checkpoint成功：%s" % best_checkpoint_filename)
    test(model, test_loader, config['Ncrop'], device)


if __name__ == '__main__':
    main()
