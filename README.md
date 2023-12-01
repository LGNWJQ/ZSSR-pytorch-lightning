# ZSSR-pytorch-lightning
电子科技大学研究生课程-高级计算机视觉课程作业-复现ZSSR

## 使用方法
1. 安装环境：python3.8.18 + requirements.txt
2. 在config.py中设定训练参数
3. 运行main.py
4. 运行main.py会生成两个文件夹：
   - lightning_logs：存放实验配置以及tensorboard文件
   - SR_Result：输出训练过程中产生的超分结果

## 文件结构
```yaml
dataset:
   - __init__.py
   - dataset.py: 自定义数据集
model:
   - lightning_model.py: pytorch_lightning框架的核心部分，封装了训练和验证的代码
   - model.py: 原论文的CNN
   - resnet_model.py: 采用ResNet改进后的网络
config.py: 训练和测试的参数设定
main.py: 训练脚本，设定完成config.py后直接运行该文件
requirements.py: 环境配置 pip install -r requirements.txt
```
