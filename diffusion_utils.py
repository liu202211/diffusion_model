import datetime
import os
import time

import torch
import typing
from torch.utils import data as torch_data
import functools
from torch.utils.tensorboard import SummaryWriter


class Get_bate_base:
    '''
        一个生成bate数值的基础类型
        所有的bate数据生成器都需要继承这个类
        参数：
            start： bate数值开始值
            end :  bate数值结束值
            time_Step： bate所能支持的最大数量
    '''

    def __init__(self, start: float, end: float, time_step: int):
        self.start = start
        self.end = end
        self.time_step = time_step

    def _get_item(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, cnt: int):
        assert 0 < cnt <= self.time_step
        assert isinstance(cnt, int)
        return self._get_item(torch.tensor([i for i in range(cnt)]))


class Get_bate_with_liner(Get_bate_base):
    def __init__(self, start, end, time_step):
        super(Get_bate_with_liner, self).__init__(start, end, time_step)
        self.k = (end - start) / (time_step - 1)
        self.b = start

    def _get_item(self, x: torch.Tensor) -> torch.Tensor:
        return self.k * x + self.b


def _extract(a, t, x_shape):
    '''

    :param a: 数据
    :param t: 步
    :param x_shape: 形状
    :return: 抽取后的数据
    '''
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _q_sample(sqrt_alphas_cumprod: torch.Tensor,
              sqrt_one_minus_alphas_cumprod: torch.Tensor,
              t: torch.Tensor,
              x0: torch.Tensor,
              noise: torch.Tensor) -> torch.Tensor:
    '''
    前向采样
    :param sqrt_alphas_cumprod: 根号下α的连乘
    :param sqrt_one_minus_alphas_cumprod: 根号下1-a的连乘
    :param t:  步数
    :param x0: 初始图像
    :param noise: 噪声
    :return:
    '''

    t = t - 1
    '''
    参考论文中公式4.
    变形在第三页公式9上部
    '''
    return _extract(sqrt_alphas_cumprod, t, x0.shape) * x0 + \
           _extract(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise


def _p_sample_func_step(t: torch.Tensor,
                        alhpa: torch.Tensor,
                        xt: torch.Tensor,
                        sqrt_one_minus_alphas_cumprod: torch.Tensor,
                        get_z_t: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                        noise: torch.Tensor,
                        sigma: torch.Tensor,
                        ):
    '''
        该函数执行论文中算法2 Sampling 的第4步，给定一定的参数， 从xt中求解出x_t-1
        :param t: 当前的步数
        :param sqrt_alpha:根号下a的序列
        :param xt:xt的图像
        :param sqrt_one_minus_alphas_cumprod: 根号下1-alpha的联乘
        :param get_z_t:调用时会返回第t时刻的噪声，
        :param noise: 一个采样自标准正态分布中的噪声
        :param get_var_t:获取方差， sigma(t)
        :return: 返回t-1时刻的图像
    '''
    sqrt_alpha = torch.sqrt(alhpa)
    z_t = get_z_t(xt, t - 1)
    t = t - 1
    part1 = 1. / _extract(sqrt_alpha, t, xt.shape)
    part2 = 1. - _extract(alhpa, t, xt.shape)
    part3 = _extract(sqrt_one_minus_alphas_cumprod, t, xt.shape)
    x_t_minus_1 = part1 * (xt - (part2 / part3) * z_t) + _extract(sigma, t, xt.shape) * noise
    return x_t_minus_1


class Diffusion_model:
    def __init__(self, time_step: int,
                 get_bate_func: Get_bate_base,
                 batch_size: int = 64,
                 epoch: int = 300,
                 dataset: torch_data.Dataset = None,
                 data_shape: typing.Iterable = None,
                 get_data_func: typing.Callable[..., torch.Tensor] = None,  # 如果非空，数据集数据会经过这一项
                 run_model: str = 'train',  # train, evaluation
                 net_module: torch.nn.Module = None,
                 net_parameter_load: str = None,
                 net_parameter_save_path: str = './out/parameter',
                 temp_path: str = './out/temp',
                 save_net_para_step: int = 400,
                 device: str = 'auto',  # auto, cpu, cuda
                 cuda_num: int = 0,
                 lr = 0.0001
                 ):
        '''

        :param time_step: 步数t
        :param get_bate_func: 用于获取bate序列的方法， 注意该类仅允许继承与Get_bate_base类
        :param batch_size: 每个batch 的尺寸
        :param epoch: 训练的轮数
        :param dataset: 数据集， 必须是torch.utils.data.Dataset类
        :param data_shape: 处理后数据的形状
        :param get_data_func: 数据会经过该方法做处理，然后送到网络中 请返回处理后的数据
        :param run_model: 运行模式
        :param net_module: 网络模型
        :param net_parameter_load: 网络权重存储的路径
        :param net_parameter_save_path: 训练中权重文件保存数据的路径
        :param temp_path: 存放检查文件等信息的临时路径
        :param save_net_para_step: 保存网络参数的频率
        :param device: 训练使用的设备， 仅允许为auto， cuda 和 cpu
        :param cuda_num: 如果为cuda， 该项为使用的gpu的序号。

        训练模式下：
        '''
        print('init deffusion model')

        '''
        实例化设备
        '''
        assert device == 'cpu' or device == 'cuda' or device == 'auto'

        self.device = {
            'auto': lambda: torch.device(f'cuda:{cuda_num}') if torch.cuda.is_available() else torch.device('cpu'),
            'cpu': lambda: torch.device('cpu'),
            'cuda': lambda: torch.device(f'cuda:{cuda_num}'),
        }[device]()
        if device == 'cup' and torch.cuda.is_available():
            Warning('the cuda is available, but the device is set as cpu')

        self.time_step = time_step
        assert isinstance(get_bate_func, Get_bate_base)

        '''
            初始化一些序列
        '''
        self.bates = get_bate_func(time_step).to(self.device)
        self.alphas = 1. - self.bates
        self.sigma = torch.sqrt(self.bates)
        self.alphas_cumprod = torch.cumprod(self.alphas, -1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.batch_size = batch_size
        self.epoch = epoch
        self.temp_path = temp_path
        self.save_net_para_step = save_net_para_step
        self.dataset = dataset

        assert data_shape is not None
        assert isinstance(data_shape, typing.Iterable)
        data_shape = list(data_shape)
        assert len(data_shape) == 3
        self.data_shape = data_shape
        self.run_model = run_model
        self.lr = lr

        '''
            加载数据集
        '''
        if run_model == 'train':
            print('训练模式， 正在加载数据集')
            assert dataset is not None
            self.data_loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.get_data_func = get_data_func if get_data_func is not None else lambda x: x
        elif run_model == 'evaluation':
            assert net_parameter_load is not None
        else:
            raise NotImplementedError(f'run_mode: {run_model}  is not implemented')

        '''
            加载网络模型
        '''
        self.net = net_module.to(self.device)
        self.net_parameter_save_path = net_parameter_save_path

        if net_parameter_load is not None:
            assert os.path.exists(net_parameter_load)
            self.net.load_state_dict(torch.load(net_parameter_load))

    def q_sample(self, t: torch.Tensor, x0: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn(x0.shape, device=self.device)

        return _q_sample(sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                         sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                         t=t, x0=x0, noise=noise)

    def train(self):
        loss_func = torch.nn.L1Loss().to(self.device)
        optim = torch.optim.Adam(params=self.net.parameters(), lr=self.lr)
        writer = SummaryWriter(os.path.join(self.temp_path, './logs'))
        step = 0
        total = len(self.data_loader)

        for epoch in range(self.epoch):
            step_epoch = 0
            loss_sum = 0

            for data in self.data_loader:
                p = step_epoch / total * 100

                # 采集一组t
                self.net.zero_grad()
                t = torch.randint(low=1, high=self.time_step + 1, size=[data.shape[0]], device=self.device)

                data = self.get_data_func(data).to(self.device)
                noise = torch.randn(data.shape, device=self.device)
                x_t = self.q_sample(t, data, noise)  # 前向计算xt
                z_t = self.net(x_t, time=t, y=None)  # 让模型预测xt的噪声

                '''
                计算loss 进行训练
                '''
                loss = loss_func(noise, z_t)
                loss.backward()
                optim.step()
                print(f'\r|epoch = {epoch}| loss: {loss}| {int(p)} |' + '*' * int(p) + '-' * (100 - int(p)) + '|',
                      end='')
                writer.add_scalar('loss', loss, step)
                loss_sum += loss
                step_epoch += 1
                step += 1

                '''
                处理模型参数保存
                '''
                if step % self.save_net_para_step == 0:
                    self.save_parameter()
                    filename = os.path.join(self.temp_path, f'check-{time.time()}-{"NULL" if epoch == -1 else epoch}.jpg')
                    self._p_sample(filename=filename)
            print(f'\r|epoch = {epoch}| loss = {loss_sum / step_epoch}')

        writer.close()

    def save_parameter(self, loss=-1):
        parameter = self.net.state_dict()
        now_time = datetime.datetime.now().strftime("%H:%M:%S")
        torch.save(parameter, os.path.join(self.temp_path, f't:{now_time}_loss:{loss}.pth'))

    def _p_sample(self, filename, img_num=10):
        BATCH_SIZE_P = img_num

        _p_sample_step = functools.partial(_p_sample_func_step, alhpa=self.alphas,
                                           sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                                           sigma=self.sigma, get_z_t=self.net)
        shape = [BATCH_SIZE_P] + self.data_shape
        xt = torch.randn(shape, device=self.device)
        for t in range(self.time_step, 0, -1):
            z = (torch.randn_like(xt) if t > 1 else torch.zeros_like(xt, dtype=torch.float)).to(self.device)
            with torch.no_grad():
                xt = _p_sample_step(t=torch.tensor([t], device=self.device).repeat(BATCH_SIZE_P), xt=xt, noise=z)

        self._save_image(xt, filename)

    @staticmethod
    def _save_image(imgs, filename):
        import torchvision
        torchvision.utils.save_image(imgs, filename, nrow=5, normalize=True)

    def evaluation(self, filename, img_num):
        self._p_sample(filename=filename, img_num=img_num)

    def __call__(self, *args, **kwargs):
        '''
        根据初始化的运行模式进行调用， 如果是训练模式， 不接收任何参数
        如果是eval模式， 需提供 filename和img_num参数
        filename： 文件保存的位置
        img_num： 生成图像的数量， 默认是10
        '''
        if self.run_model == 'train':
            self.train()
        else:
            self.evaluation(*args, **kwargs)
