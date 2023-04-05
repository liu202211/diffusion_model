import torch


class Stack_base:
    def __init__(self):
        ...

    def pop(self):
        raise NotImplementedError

    def push(self, x):
        raise NotImplementedError

    def isEmpty(self):
        raise NotImplementedError


class Stack_List(Stack_base):
    def __init__(self):
        super(Stack_List, self).__init__()
        self.lst = []

    def pop(self):
        assert not self.isEmpty()
        return self.lst.pop()

    def push(self, x):
        self.lst.append(x)

    def isEmpty(self):
        return len(self.lst) == 0


class Net_liner(torch.nn.Module):
    def __init__(self, in_size, out_size, time_step):
        super(Net_liner, self).__init__()
        '''
        输入是 3 * 28 * 28 + 1（t）
        输出是 1 * 28 * 28
        '''
        self.in_size = in_size
        self.out_size = out_size

        in_feat = 1
        for item in in_size:
            in_feat *= item
        out_feat = 1
        for item in out_size:
            out_feat *= item

        self.emb = torch.nn.Embedding(num_embeddings=time_step, embedding_dim=time_step)
        self.liner = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_feat + time_step, out_features=1024),
            torch.nn.BatchNorm1d(1024, 0.8),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=1024, out_features=1280),
            torch.nn.BatchNorm1d(1280, 0.8),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=1280, out_features=out_feat),
            torch.nn.LeakyReLU(0.2),
        )
        self.flatten = torch.nn.Flatten()

    def forward(self, x, t):
        t = self.emb(t - 1)
        x = self.flatten(x)
        x = torch.cat((x, t), -1)
        out = self.liner(x)
        out = out.view(-1, *self.out_size)
        return out


class Down_sample(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(Down_sample, self).__init__()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        assert x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0
        x = self.max_pool(x)
        x = self.Conv(x)
        return x


class Up_sample(torch.nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 concatenate_channels,
                 out_channels, ):
        super(Up_sample, self).__init__()

        self.conv_trans = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                                   out_channels=hidden_channels,
                                                   kernel_size=(4, 4),
                                                   stride=(2, 2),
                                                   padding=1)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
        )

        self.cat = lambda x, y: torch.cat((x, y), 1)

        self.conv2 = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=hidden_channels + concatenate_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x1, x2):
        x = self.conv_trans(x1)
        x = self.conv1(x)

        x = self.cat(x, x2)
        x = self.conv2(x)

        return x


class liner_emb(torch.nn.Module):
    def __init__(self,
                 in_features,
                 hid_features,
                 out_features):
        super(liner_emb, self).__init__()
        self.liner1 = torch.nn.Linear(in_features=in_features, out_features=hid_features)
        self.liner2 = torch.nn.Linear(in_features=hid_features, out_features=hid_features)
        self.liner3 = torch.nn.Linear(in_features=hid_features, out_features=out_features)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.liner1(x))
        x = self.activation(self.liner2(x))
        x = self.liner3(x)
        return self.activation(x)


class UNet_Conv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels,
                 time_step,
                 emb_liner_hidden=128,
                 channel_mults=(2, 4, 8, 16),
                 ):
        super(UNet_Conv, self).__init__()

        self.init_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=base_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
        )

        self.time_emb = torch.nn.Embedding(time_step, time_step)

        self.down_sample_lst = torch.nn.ModuleList()
        self.down_time_emb = torch.nn.ModuleList()
        self.up_sample_lst = torch.nn.ModuleList()

        now_channels = base_channels
        channels_lst = [now_channels]
        for mul in channel_mults:
            self.down_sample_lst.append(Down_sample(in_channels=now_channels,
                                                    out_channels=base_channels * mul))
            self.down_time_emb.append(liner_emb(in_features=time_step,
                                                hid_features=emb_liner_hidden,
                                                out_features=now_channels))
            now_channels = base_channels * mul
            channels_lst.append(now_channels)

        self.mid = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=now_channels,
                            out_channels=now_channels,
                            kernel_size=(3, 3),
                            padding=1)
        )
        channels_lst.reverse()
        for i in range(len(channels_lst) - 1):
            self.up_sample_lst.append(
                Up_sample(in_channels=channels_lst[i],
                          hidden_channels=channels_lst[i + 1],
                          concatenate_channels=channels_lst[i + 1],
                          out_channels=channels_lst[i + 1])
            )

        self.end_conv = torch.nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=(3, 3),
                                        padding=1)

        self.act = torch.nn.Tanh()

    def forward(self, x, time=None, y=None):
        x = self.init_conv(x)
        time_emb = self.time_emb(time)
        stack = Stack_List()

        stack.push(x)

        # 下采样
        for index in range(len(self.down_sample_lst)):
            layer1, layer2 = self.down_sample_lst[index], self.down_time_emb[index]
            time_emb_liner = layer2(time_emb)

            x = x + time_emb_liner[:, :, None, None]
            x = layer1(x)
            stack.push(x)

        # 平层
        x = self.mid(x)
        stack.pop()
        # 上采样

        for layer in self.up_sample_lst:
            x1 = stack.pop()
            x = layer(x, x1)

        x = self.end_conv(x)
        x = self.act(x)
        return x


class UNet_Conv_label_2d(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels,
                 time_step,
                 emb_liner_hidden=128,
                 channel_mults=(2, 4, 8, 16),
                 use_label=False,
                 input_label_channels=1,
                 ):
        super(UNet_Conv_label_2d, self).__init__()

        self.init_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels if not use_label else (in_channels+input_label_channels),
                            out_channels=base_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
        )

        self.time_emb = torch.nn.Embedding(time_step, time_step)
        self.use_label = use_label

        self.down_sample_lst = torch.nn.ModuleList()
        self.down_time_emb = torch.nn.ModuleList()

        self.up_sample_lst = torch.nn.ModuleList()

        now_channels = base_channels
        channels_lst = [now_channels]
        for mul in channel_mults:
            self.down_sample_lst.append(Down_sample(in_channels=now_channels,
                                                    out_channels=base_channels * mul))
            self.down_time_emb.append(liner_emb(in_features=time_step,
                                                hid_features=emb_liner_hidden,
                                                out_features=now_channels))
            now_channels = base_channels * mul
            channels_lst.append(now_channels)

        self.mid = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=now_channels,
                            out_channels=now_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=now_channels,
                            out_channels=now_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=now_channels,
                            out_channels=now_channels,
                            kernel_size=(3, 3),
                            padding=1),
            torch.nn.ReLU(),

        )
        channels_lst.reverse()
        for i in range(len(channels_lst) - 1):
            self.up_sample_lst.append(
                Up_sample(in_channels=channels_lst[i],
                          hidden_channels=channels_lst[i + 1],
                          concatenate_channels=channels_lst[i + 1],
                          out_channels=channels_lst[i + 1])
            )

        self.end_conv = torch.nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=(3, 3),
                                        padding=1)

        self.act = torch.nn.Tanh()

    def forward(self, x, time=None, y=None):
        if self.use_label:
            assert y is not None
            x = torch.cat((x, y), 1)
        x = self.init_conv(x)
        time_emb = self.time_emb(time)
        stack = Stack_List()
        stack.push(x)
        # 下采样
        for index in range(len(self.down_sample_lst)):
            layer1, layer2 = self.down_sample_lst[index], self.down_time_emb[index]

            time_emb_liner = layer2(time_emb)
            x = x + time_emb_liner[:, :, None, None]
            x = layer1(x)
            stack.push(x)

        # 平层
        x = self.mid(x)
        stack.pop()
        # 上采样

        for layer in self.up_sample_lst:
            x1 = stack.pop()
            x = layer(x, x1)

        x = self.end_conv(x)
        x = self.act(x)
        return x


if __name__ == '__main__':
    unet = UNet_Conv(in_channels=3,
                     out_channels=3,
                     base_channels=16,
                     time_step=200,
                     emb_liner_hidden=256, )
    x = torch.randn(3, 3, 512, 512)
    t = torch.tensor([5, 7, 8])
    unet(x, t=t)
