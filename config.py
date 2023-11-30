import argparse


def set_config():
    parser = argparse.ArgumentParser()
    # 模型
    parser.add_argument('--in_channels', type=int, default=3, help='')
    parser.add_argument('--channels', type=int, default=64, help='')
    parser.add_argument('--num-layer', type=int, default=8, help='')
    # 数据
    parser.add_argument('--image_path', type=str, default='dataset/img_003.png', help='')
    parser.add_argument('--sr_factor', type=int, default=4, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument('--patch_size', type=int, default=256, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--num_scale', type=int, default=6, help='')
    # 训练
    parser.add_argument('--exp_name', type=str, default='SR_img3', help='')
    parser.add_argument('--num_epoch', type=int, default=5000, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=200, help='')
    parser.add_argument('--save_frequence', type=int, default=100)
    parser.add_argument('--accelerator', type=str, default='gpu', help='')

    config = parser.parse_args()
    return config