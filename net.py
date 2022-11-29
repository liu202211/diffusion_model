import torch


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
        t = self.emb(t-1)
        x = self.flatten(x)
        x = torch.cat((x, t), -1)
        out = self.liner(x)
        out = out.view(-1, *self.out_size)
        return out
