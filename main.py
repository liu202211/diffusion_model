import argparse
import os

import diffusion_utils
import my_dataset
from unet import UNet as My_net

import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def args_initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--time_step', type=int, default=300, help='input batch size')
    parser.add_argument('-lr', '--learning_rate', type=int, default=0.0001, help='input batch size')
    parser.add_argument('-d', '--device', default='auto', choices=['cpu', 'cuda', 'auto'],
                        help='device: cpu, cuda, auto')
    parser.add_argument('-e', '--epoch', type=int, default=300, help='number of epochs to train for')
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('-t', '--train', action='store_true', help='train mod')
    # group.add_argument('-r', '--run', action='store_true', help='test mod')

    parser.add_argument('--GPU', default="0", help='GPU id')
    help_temp = 'continue train, you must given module args root by --continue-train-path'
    parser.add_argument('--continue_train', action='store_true', help=help_temp)
    parser.add_argument('--continue-train-path', help='path')
    parser.add_argument('--save_path', default="./", help='module arg save path')
    parser.add_argument('--dateset_path', default="./", help='module arg save path')
    parser.add_argument('--temp_path', default="./temp", help='module arg save path')

    args = parser.parse_args()

    return args


def train(args):
    img_size = [3, 96, 96]
    noise_size = [3, 96, 96]
    root = os.path.join(args.dateset_path, 'faces')
    assert os.path.isdir(root)
    diff = diffusion_utils.Diffusion_model(time_step=args.time_step,
                                           get_bate_func=diffusion_utils.Get_bate_with_liner(start=0.0001,
                                                                                             end=0.03,
                                                                                             time_step=args.time_step),
                                           batch_size=args.batch_size,
                                           epoch=args.epoch,
                                           dataset=my_dataset.My_dataset(root),
                                           data_shape=img_size,
                                           get_data_func=lambda x: x,
                                           run_model='train',
                                           lr=args.learning_rate,
                                           net_module=My_net(
                                               img_channels=3,
                                               base_channels=128,
                                               channel_mults=(1, 2, 2, 2),
                                               time_emb_dim=4*128,
                                               norm='gn',
                                               dropout=0.1,
                                               attention_resolutions=(1,),
                                               num_classes=None,
                                               initial_pad=0,
                                           )  # 网络
                                           )
    diff()


def main():
    args = args_initialize()
    if not os.path.exists(args.temp_path):  # args.temp_path
        os.mkdir(args.temp_path)
    train(args)


if __name__ == '__main__':
    main()
