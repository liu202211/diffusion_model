'''
该程序的功能是输入一个黑白图像，将图像进行上色。
目的是为了测试网络的条件控制。


'''

import argparse
import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import diffusion_utils
from my_dataset import bch_70_img as data_set
from my_dataset import generator_dataset
from unet import UNet as My_net
# from unet import UNet as My_net
from torch.utils.data import Dataset, DataLoader

import os
import torchvision


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def args_initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5, help='batch的尺寸')
    parser.add_argument('--time_step', type=int, default=1500, help='t的最大取值， t是扩散模型的扩散步数')
    parser.add_argument('-lr', '--learning_rate', type=int, default=0.0002, help='学习率')
    parser.add_argument('-d', '--device', default='auto', choices=['cpu', 'cuda', 'auto'],
                        help='设备可选: cpu, cuda, auto')
    parser.add_argument('-e', '--epoch', type=int, default=500, help='训练的轮数')
    #group = parser.add_mutually_exclusive_group(required=True)
    #group.add_argument('-t', '--train', action='store_true', help='train mod')
    #group.add_argument('-e', '--eval', action='store_true', help='eval mod')

    parser.add_argument('--GPU', default=0, type=int, help='如果是gpu运算，且有多个gpu，这里指定使用的cuda号')
    parser.add_argument('-c', '--continue_train', action='store_true',
                        help='是否加载预训练模型， 通过load_para_path指定权重文件位置')
    parser.add_argument('--load_para_path', default='out/18_20_46_loss-0.01740149036049843.pth',
                        type=str, help='加载预训练的参数权重， 指定权重文件位置')
    parser.add_argument('--save_para_path', default="./out/para", help='模型参数的保存位置')
    parser.add_argument('--dateset_path', default='./dataset/256bch70cgansin1/Date', help='数据集的位置')
    parser.add_argument('--temp_path', default="./out/temp", help='临时文件存放位置')
    parser.add_argument('--save_net_para_step', default=1000, type=int,
                        help='自动保存网络参数的频率， 多少步进行一次保存')

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
    img_size = [2, 256, 256]
    data_path = args.dateset_path
    assert os.path.isdir(data_path)
    #args.continue_train = True
    '''
    加载扩散模型
    '''
    diff = diffusion_utils.Diffusion_model(time_step=args.time_step,
                                           get_bate_func=diffusion_utils.Get_bate_with_liner(start=0.0001,
                                                                                             end=0.002,
                                                                                             time_step=args.time_step),
                                           data_shape=img_size,
                                           run_model='train',
                                           lr=args.learning_rate,
                                           # optim_func=torch.optim.SGD,
                                           loss_function= torch.nn.L1Loss,
                                           net_module=My_net(
                                               img_channels=3,
                                               img_out_channels=2,
                                               base_channels=16,
                                               channel_mults=(1, 1, 1, 2, 2, 4, 4),
                                               time_emb_dim=1800,
                                               norm='gn',
                                               dropout=0.1,
                                               attention_resolutions=(2, 4, ),
                                               num_classes=None,
                                               initial_pad=0,
                                               num_groups=4
                                           ),  # 网络
                                           device=args.device,
                                           cuda_num=args.GPU,
                                           net_parameter_load=args.load_para_path if args.continue_train else None,
                                           )
    # 加载数据集
    loader_train = DataLoader(data_set(data_path, mode='TRAIN_MODE'), batch_size=args.batch_size, shuffle=True)
    loader_iter = generator_dataset(DataLoader(data_set(data_path, mode='TEST_MODE'), batch_size=5, shuffle=True))
    loader_iter2 = generator_dataset(DataLoader(data_set(data_path, mode='TRAIN_MODE'), batch_size=5, shuffle=True))
    writer = SummaryWriter(os.path.join(args.temp_path, './logs'))
    step = 0
    total = len(loader_train)

    for epoch in range(args.epoch):
        step_epoch = 0
        loss_sum = 0
        for datas in loader_train:
            p = step_epoch / total * 100
            data, y = datas
            # # 进行傅里叶变换
            # data = torch.fft.fft(torch.fft.fft(data, dim=2), dim=3)
            # data = torch.cat((data.real, data.imag), dim=1)

            # 将变换后的数据输入到网络中进行训练
            loss = diff(data, y=y)  # 将数据喂到网络中
            print(f'\r|epoch = {epoch}| loss: {loss}| {int(p)} |' + '*' * int(p) + '-' * (100 - int(p)) + '|',
                  end='')
            writer.add_scalar('loss', loss, step)
            loss_sum += loss
            step_epoch += 1
            step += 1

        if epoch % 20 == 0  and epoch != 0 and epoch > 100:
            now_time = datetime.datetime.now().strftime("%H_%M_%S")
            diff.save_parameter(filename=os.path.join(args.save_para_path, f'{now_time}_loss-{loss}.pth'))

            diff.check_out(run_model='evaluation')
            
            _, y = next(loader_iter)
            imgs = diff(img_num=5, y=y).cpu()

            filename = os.path.join(args.temp_path, 'test', f'check-{now_time}-{epoch}.jpg')
            img1, img2 = imgs[:, :1, :, :], imgs[:, 1:, :, :]

            temp = torch.cat((y.repeat(1, 3, 1, 1), img1.repeat(1, 3, 1, 1), img2.repeat(1, 3, 1, 1)), 0)
            torchvision.utils.save_image(temp, filename, nrow=5)
            
            _, y = next(loader_iter2)
            imgs = diff(img_num=5, y=y).cpu()

            filename = os.path.join(args.temp_path, 'test', f'tcheck-{now_time}-{epoch}.jpg')
            img1, img2 = imgs[:, :1, :, :], imgs[:, 1:, :, :]

            temp = torch.cat((y.repeat(1, 3, 1, 1), img1.repeat(1, 3, 1, 1), img2.repeat(1, 3, 1, 1)), 0)
            torchvision.utils.save_image(temp, filename, nrow=5)
            diff.check_out()
        print(f'\r|epoch = {epoch}| loss = {loss_sum / step_epoch}')
    writer.close()


def evaluation(args):
    img_size = [2, 256, 256]
    data_path = args.dateset_path
    assert os.path.isdir(data_path)

    diff = diffusion_utils.Diffusion_model(time_step=args.time_step,
                                           get_bate_func=diffusion_utils.Get_bate_with_liner(start=0.0001,
                                                                                             end=0.005,
                                                                                             time_step=args.time_step),
                                           data_shape=img_size,
                                           run_model='evaluation',
                                           net_module=My_net(
                                               img_channels=3,
                                               img_out_channels=2,
                                               base_channels=30,
                                               channel_mults=(1, 1, 1, 1, 2, 2, 2),
                                               time_emb_dim=1200,
                                               norm='gn',
                                               dropout=0.1,
                                               attention_resolutions=(2, 4, 6),
                                               num_classes=None,
                                               initial_pad=0,
                                               num_groups=10
                                           ),  # 网络
                                           device=args.device,
                                           cuda_num=args.GPU,
                                           net_parameter_load=args.load_para_path,
                                           )

    dataset_test = data_set(data_path, mode='TEST_MODE')

    step = 0
    for datas in dataset_test:
        p = int(step / len(dataset_test) * 100)
        print(f'\r| {int(p)} |' + '*' * int(p) + '-' * (100 - int(p)) + '|', end='')
        real_img, condition_img = datas
        condition_img = torch.unsqueeze(condition_img, dim=0)
        fakes = diff(img_num=1, y=condition_img).cpu()
        fakes = torch.squeeze(fakes, dim=0)

        # 创建一个目录， 存放结果
        os.mkdir(os.path.join(args.temp_path, 'evaluation', f'{step}'))
        fake1, fake2 = fakes[0, :, :].repeat(3, 1, 1), fakes[1, :, :].repeat(3, 1, 1)
        real_img1, real_img2 = real_img[0, :, :].repeat(3, 1, 1), real_img[1, :, :].repeat(3, 1, 1)
        torchvision.utils.save_image(real_img1, os.path.join(args.temp_path, 'evaluation', f'{step}', 'real1.bmp'))
        torchvision.utils.save_image(real_img2, os.path.join(args.temp_path, 'evaluation', f'{step}', 'real2.bmp'))
        torchvision.utils.save_image(fake1, os.path.join(args.temp_path, 'evaluation', f'{step}', 'fake1.bmp'))
        torchvision.utils.save_image(fake2, os.path.join(args.temp_path, 'evaluation', f'{step}', 'fake2.bmp'))
        torchvision.utils.save_image(condition_img,
                                     os.path.join(args.temp_path, 'evaluation', f'{step}', 'condition.bmp'))
        step += 1


def main():
    args = args_initialize()
    if not os.path.exists(args.temp_path):  # args.temp_path
        os.mkdir(args.temp_path)
    train(args)
   # evaluation(args)


if __name__ == '__main__':
    main()
