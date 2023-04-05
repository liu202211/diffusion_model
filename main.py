import argparse
import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import diffusion_utils
from my_dataset import Large_dataset as data_set
from net import UNet_Conv as My_net
# from unet import UNet as My_net
from torch.utils.data import Dataset, DataLoader

import os
import torchvision


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def args_initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch的尺寸')
    parser.add_argument('--time_step', type=int, default=300, help='t的最大取值， t是扩散模型的扩散步数')
    parser.add_argument('-lr', '--learning_rate', type=int, default=0.0001, help='学习率')
    parser.add_argument('-d', '--device', default='auto', choices=['cpu', 'cuda', 'auto'],
                        help='设备可选: cpu, cuda, auto')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='训练的轮数')
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('-t', '--train', action='store_true', help='train mod')
    # group.add_argument('-e', '--eval', action='store_true', help='eval mod')

    parser.add_argument('--GPU', default=0, type=int, help='如果是gpu运算，且有多个gpu，这里指定使用的cuda号')
    parser.add_argument('-c', '--continue_train', action='store_true', help='是否加载预训练模型， 通过load_para_path指定权重文件位置')
    parser.add_argument('--load_para_path', type=str, help='加载预训练的参数权重， 指定权重文件位置')
    parser.add_argument('--save_para_path', default="./out/para", help='模型参数的保存位置')
    parser.add_argument('--dateset_path', default="./dataset/faces", help='数据集的位置')
    parser.add_argument('--temp_path', default="./out/temp", help='临时文件存放位置')
    parser.add_argument('--save_net_para_step', default=400, type=int, help='自动保存网络参数的频率， 多少步进行一次保存')

    args = parser.parse_args()

    return args


'''
net_module=My_net(
                                               img_channels=3,
                                               base_channels=128,
                                               channel_mults=(1, 2, 2, 2),
                                               time_emb_dim=4 * 128,
                                               norm='gn',
                                               dropout=0.1,
                                               attention_resolutions=(1,),
                                               num_classes=None,
                                               initial_pad=0,
                                           ),  # 网络
'''


def train(args):
    img_size = [3, 96, 96]
    data_path = args.dateset_path
    assert os.path.isdir(data_path)

    '''
    加载扩散模型
    '''
    diff = diffusion_utils.Diffusion_model(time_step=args.time_step,
                                           get_bate_func=diffusion_utils.Get_bate_with_liner(start=0.0001,
                                                                                             end=0.03,
                                                                                             time_step=args.time_step),
                                           data_shape=img_size,
                                           run_model='train',
                                           lr=args.learning_rate,
                                           net_module=My_net(
                                               in_channels=3,
                                               out_channels=3,
                                               base_channels=64,
                                               time_step=args.time_step,
                                               emb_liner_hidden=128,
                                               channel_mults=(1, 2, 2, 4, 8)
                                           ),  # 网络
                                           device=args.device,
                                           cuda_num=args.GPU,
                                           net_parameter_load=args.load_para_path if args.continue_train else None,
                                           )
    # 加载数据集
    loader = DataLoader(data_set(data_path), batch_size=args.batch_size, shuffle=True)
    writer = SummaryWriter(os.path.join(args.temp_path, './logs'))
    step = 0
    total = len(loader)
    for epoch in range(args.epoch):
        step_epoch = 0
        loss_sum = 0
        for data in loader:
            p = step_epoch / total * 100
            loss = diff(data, y=None)  # 将数据喂到网络中
            print(f'\r|epoch = {epoch}| loss: {loss}| {int(p)} |' + '*' * int(p) + '-' * (100 - int(p)) + '|',
                  end='')
            writer.add_scalar('loss', loss, step)
            loss_sum += loss
            step_epoch += 1
            step += 1

            if step % args.save_net_para_step == 0:
                now_time = datetime.datetime.now().strftime("%H:%M:%S")
                diff.save_parameter(filename=os.path.join(args.save_para_path, f'{now_time}_loss:{loss}.pth'))
                diff.check_out(run_model='evaluation')
                imgs = diff(img_num=15)
                filename = os.path.join(args.temp_path, 'test', f'check-{now_time}-{epoch}.jpg')
                torchvision.utils.save_image(imgs, filename, nrow=5)
                diff.check_out()
        print(f'\r|epoch = {epoch}| loss = {loss_sum / step_epoch}')
    writer.close()


# def train_label(args):
#     img_size = [3, 32, 32]
#     # data_path = os.path.join(args.dateset_path, 'faces')
#     # assert os.path.isdir(data_path)
#
#     '''
#     加载扩散模型
#     '''
#     diff = diffusion_utils.Diffusion_model(time_step=args.time_step,
#                                            get_bate_func=diffusion_utils.Get_bate_with_liner(start=0.0001,
#                                                                                              end=0.03,
#                                                                                              time_step=args.time_step),
#                                            data_shape=img_size,
#                                            run_model='train',
#                                            lr=args.learning_rate,
#                                            net_module=My_net(
#                                                # img_channels=3,
#                                                base_channels=128,
#                                                channel_mults=(1, 2, 2, 2),
#                                                time_emb_dim=4 * 128,
#                                                norm='gn',
#                                                dropout=0.1,
#                                                attention_resolutions=(1,),
#                                                num_classes=None,
#                                                initial_pad=0,
#                                            ),  # 网络
#                                            device=args.device,
#                                            cuda_num=args.GPU,
#                                            net_parameter_load=args.load_para_path if args.continue_train else None,
#                                            )
#     # 加载数据集
#     data = torchvision.datasets.CIFAR10('./cifar', download=True, train=True,
#                                         transform=torchvision.transforms.ToTensor())
#     data += torchvision.datasets.CIFAR10('./cifar', download=False, train=True,
#                                          transform=torchvision.transforms.ToTensor())
#     loader = DataLoader(data, batch_size=args.batch_size,
#                         shuffle=True)
#     writer = SummaryWriter(os.path.join(args.temp_path, './logs'))
#     step = 0
#     total = len(loader)
#     for epoch in range(args.epoch):
#         step_epoch = 0
#         loss_sum = 0
#         for data, label in loader:
#             p = step_epoch / total * 100
#             loss = diff(data, y=label)  # 将数据喂到网络中
#             print(f'\r|epoch = {epoch}| loss: {loss}| {int(p)} |' + '*' * int(p) + '-' * (100 - int(p)) + '|',
#                   end='')
#             writer.add_scalar('loss', loss, step)
#             loss_sum += loss
#             step_epoch += 1
#             step += 1
#
#             if step % args.save_net_para_step == 0:
#                 now_time = datetime.datetime.now().strftime("%H:%M:%S")
#                 diff.save_parameter(filename=os.path.join(args.save_para_path, f'{now_time}_loss:{loss}.pth'))
#                 diff.check_out(run_model='evaluation')
#                 imgs = diff(img_num=10, y=torch.tensor([item for item in range(10)]))
#                 filename = os.path.join(args.temp_path, 'test', f'check-{now_time}-{epoch}.jpg')
#                 torchvision.utils.save_image(imgs, filename, nrow=5)
#                 diff.check_out()
#         print(f'\r|epoch = {epoch}| loss = {loss_sum / step_epoch}')
#     writer.close()
#

def main():
    args = args_initialize()
    if not os.path.exists(args.temp_path):  # args.temp_path
        os.mkdir(args.temp_path)
    train(args)


if __name__ == '__main__':
    main()
