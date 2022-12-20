import argparse
 
def ArgumentParser():
    parser = argparse.ArgumentParser(description='forensic')
    # mode
    parser.add_argument('--mode', dest='mode', type=str, default='train')

    # basic setting
    parser.add_argument('--use_gpu', dest='use_gpu', type=bool, default=True)
    parser.add_argument('--use_pretrained', dest='use_pretrained', type=bool, default=True)
    parser.add_argument('--pretrained_path', dest='pretrained_path', type=str, default='./weights/resnet18_pretrained.pth')
    parser.add_argument('--num_classes', dest='num_classes', type=int, default=1440)

    # stage1 train setting
    parser.add_argument('--s1_lr', dest='s1_lr', type=float, default=5e-4) # 2e-4
    parser.add_argument('--s1_epochs', dest='s1_epochs', type=int, default=500)
    parser.add_argument('--s1_batch_size', dest='s1_batch_size', type=int, default=32)
    parser.add_argument('--s1_temperature', dest='s1_temperature', type=float, default=0.5)

    # stage2 train setting
    parser.add_argument('--s2_lr', dest='s2_lr', type=float, default=2e-5) # 2e-6
    parser.add_argument('--s2_epochs', dest='s2_epochs', type=int, default=500)
    parser.add_argument('--s2_batch_size', dest='s2_batch_size', type=int, default=128)
    parser.add_argument('--reload_path', dest='reload_path', type=str, default="./weights/model_stage1.pth")

    # evaluate
    parser.add_argument('--eval_model_path', dest='eval_model_path', type=str, default="./weights/model_stage2.pth")
    parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=256)

    # files
    parser.add_argument('--train_dataset', dest='train_dataset', type=str, default="./dataset8/npy/train/")
    parser.add_argument('--test_dataset', dest='test_dataset', type=str, default="./dataset8/npy/test/")
    parser.add_argument('--log_path', dest='log_path', type=str, default="./result/log/")
    parser.add_argument('--save_model_path', dest='save_model_path', type=str, default="./result/save_model/")

    args = parser.parse_args()
    return args
