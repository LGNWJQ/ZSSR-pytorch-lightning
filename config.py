import argparse


def set_config():
    parser = argparse.ArgumentParser()
    # 模型
    parser.add_argument('--in_channels', type=int, default=3, help='输入图像的通道数')
    parser.add_argument('--channels', type=int, default=64, help='卷积层的通道数')
    parser.add_argument('--num-layer', type=int, default=8, help='卷积层数量')
    # 数据
    parser.add_argument('--image_path', type=str, default=r'C:\Users\WJQ\Desktop\计算机视觉\final_proj\data\BSD100\unknown_kernel_sr2\img_001.png', help='输入图像的路径')
    parser.add_argument('--sr_factor', type=int, default=2, help='超分倍率')
    parser.add_argument('--num_workers', type=int, default=4, help='读取数据使用的cpu线程数量')
    parser.add_argument('--patch_size', type=int, default=128, help='训练时裁剪图像的大小')
    parser.add_argument('--batch_size', type=int, default=8, help='训练的批量大小')
    parser.add_argument('--num_scale', type=int, default=6, help='对输入图像的缩放倍率')
    # 训练
    parser.add_argument('--exp_name', type=str, default='img1s2_res', help='图像的名字')
    parser.add_argument('--num_epoch', type=int, default=10000, help='训练的轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=2000, help='验证频率，单位为迭代步数')
    parser.add_argument('--accelerator', type=str, default='gpu', help='训练使用的设备')

    config = parser.parse_args()
    return config
