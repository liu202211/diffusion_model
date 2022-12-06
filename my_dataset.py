import os

import torch
from torch.utils.data import Dataset
import cv2
import torchvision

SUPPORT_FILE_TYPE = ['.jpg', '.jpeg', '.bmp', '.gif', '.png']


class Small_dataset(Dataset):

    def __init__(self, root):
        print(f'The dataset will being loaded from “{root}” in here, please waiting...')
        print(f'Do be checking the legitimacy of “{root}”...')
        assert os.path.isdir(root)
        print('checked success, to be going to load dataset...')
        super(Small_dataset, self).__init__()

        no_support_file = []  # 不支持的文件列表
        self.images = []
        toT = torchvision.transforms.ToTensor()
        # 从文件夹中读取文件
        print('reading files...')
        file_paths = []
        dir_list = os.listdir(root)
        dir_len = len(dir_list)
        for index, item in enumerate(dir_list):
            p = int(index / dir_len * 100)
            print('\r|' + '>' * p + '=' * (100 - p) + '|', end='')
            file_path = os.path.join(root, item)
            # 如果不是文件，就终止当前的读入
            if not os.path.isfile(file_path):
                continue

            # 判断是不是支持的文件类型
            file_ext: str = os.path.splitext(file_path)[-1].lower()
            if file_ext not in SUPPORT_FILE_TYPE:
                no_support_file.append(file_path)
                continue

            file_paths.append(file_path)
        # file_paths = file_paths
        available_img_len = len(file_paths)
        print('\r' + ' ' * 110 + '\r', end='')
        print(f'read success, the number of available image is {available_img_len}')
        print('loading images to memory')

        # # 读取支持的文件类型
        for index, file_path in enumerate(file_paths):
            p = int(index / available_img_len * 100)
            print('\r|' + '>' * p + '=' * (100 - p) + '|', end='')
            img = cv2.imread(file_path)
            self.images.append(toT(img))
        print('\r' + ' ' * 110 + '\r', end='')
        print('dataset: success')

    def __getitem__(self, item):
        return self.images[item]

    def __len__(self):
        return len(self.images)


class Large_dataset(Dataset):
    def __init__(self, root):
        print(f'The dataset will being loaded from “{root}” in here, please waiting...')
        print(f'Do be checking the legitimacy of “{root}”...')
        assert os.path.isdir(root)
        print('checked success, to be going to load dataset...')
        no_support_file = []  # 不支持的文件列表
        self.images = []

        # 从文件夹中读取文件
        print('reading files...')
        file_paths = []
        dir_list = os.listdir(root)
        dir_len = len(dir_list)
        for index, item in enumerate(dir_list):
            p = int(index / dir_len * 100)
            print('\r|' + '>' * p + '=' * (100 - p) + '|', end='')
            file_path = os.path.join(root, item)
            # 如果不是文件，就终止当前的读入
            if not os.path.isfile(file_path):
                continue

            # 判断是不是支持的文件类型
            file_ext: str = os.path.splitext(file_path)[-1].lower()
            if file_ext not in SUPPORT_FILE_TYPE:
                no_support_file.append(file_path)
                continue

            file_paths.append(file_path)
        # file_paths = file_paths
        available_img_len = len(file_paths)
        print('\r' + ' ' * 110 + '\r', end='')
        print(f'read success, the number of available image is {available_img_len}')
        self.file_paths = file_paths
        # # 读取支持的文件类型
        print('dataset: success')

    def __getitem__(self, item):
        filename = self.file_paths[item]
        img = cv2.imread(filename)
        return torchvision.transforms.ToTensor()(img)

    def __len__(self):
        return len(self.file_paths)


if __name__ == '__main__':
    data = Large_dataset('./dataset/00000')
    # data = My_dataset('./faces')
    print(len(data))
    print(data[0].shape)
