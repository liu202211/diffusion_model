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
    dataset_mode = {'TRAIN_MODE': lambda x: x[:int(len(x) * 0.7)],
                    'TEST_MODE': lambda x: x[int(len(x) * 0.7):],
                    'ALL_DATA_MODE': lambda x: x}

    def __init__(self, root,
                 mode='ALL_DATA_MODE'):
        print(f'The dataset will being loaded from “{root}” in here, please waiting...')
        print(f'Do be checking the legitimacy of “{root}”...')
        assert os.path.isdir(root)
        print('checked success, to be going to load dataset...')
        print(f'run_mode : {mode}')
        if mode not in Large_dataset.dataset_mode:
            raise RuntimeError(f'run_mode : {mode} is not supported, you can choose TRAIN_MODE, TEST_MODE or '
                               f'ALL_DATA_MODE')
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
        self.file_paths = Large_dataset.dataset_mode[mode](file_paths)
        # # 读取支持的文件类型
        print('dataset: success')
        self.gray_trans = torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

    def __getitem__(self, item):
        filename = self.file_paths[item]
        img = cv2.imread(filename)
        from PIL import Image
        return torchvision.transforms.ToTensor()(img), self.gray_trans(Image.fromarray(img))

    def __len__(self):
        return len(self.file_paths)


class bch_70_img(Dataset):

    def __init__(self, root, mode='TRAIN_MODE'):
        super().__init__()
        print(f'The dataset will being loaded from “{root}” in here, please waiting...')
        print(f'Do be checking the legitimacy of “{root}”...')
        assert os.path.isdir(root)
        print('checked success, to be going to load dataset...')
        print(f'run_mode : {mode}')
        if mode not in Large_dataset.dataset_mode:
            raise RuntimeError(f'run_mode : {mode} is not supported, you can choose TRAIN_MODE, TEST_MODE or '
                               f'ALL_DATA_MODE')
        no_support_file = []  # 不支持的文件列表
        self.images = []

        # 从文件夹中读取根目录处的真图
        print('reading files...')
        file_paths = []
        dir_list = os.listdir(root)

        for item in dir_list:
            file_path = os.path.join(root, item)
            # 如果不是文件，就终止当前的读入
            if not os.path.isfile(file_path):
                continue

            # 判断是不是支持的文件类型
            file_ext: str = os.path.splitext(file_path)[-1].lower()
            if file_ext not in SUPPORT_FILE_TYPE:
                no_support_file.append(file_path)
                continue

            file_paths.append(item)
        import fnmatch
        file_paths = fnmatch.filter(file_paths, '*_p*.*')
        if len(file_paths) & 0x01 != 0x00:
            raise RuntimeError(f'读取的文件数据错误，文件列表为:{file_paths}\n\r'
                               '您必须确定，数据必须是成对出现的')
        real_img_mat = [[None, None] for _ in range(len(file_paths) // 2)]

        for item in file_paths:
            temp_split_index, temp_split_pn = item.split('_')
            temp_split_index = int(temp_split_index)
            temp_split_pn = int(temp_split_pn.split('.')[0][1:])

            real_img_mat[temp_split_index - 1][temp_split_pn - 1] = os.path.join(root, item)

        self.real_img_mat = real_img_mat

        # 读取混叠图像
        if mode == 'TRAIN_MODE':
            condition_path = os.path.join(root, 'PData_Train')
        elif mode == 'TEST_MODE':
            condition_path = os.path.join(root, 'PData_Test')
        else:
            raise NotImplementedError(mode)
        print(condition_path, root)
        assert os.path.isdir(condition_path)

        file_paths = []
        for item in os.listdir(condition_path):
            file_path = os.path.join(condition_path, item)
            # 如果不是文件，就终止当前的读入
            if not os.path.isfile(file_path):
                continue

            # 判断是不是支持的文件类型
            file_ext: str = os.path.splitext(file_path)[-1].lower()
            if file_ext not in SUPPORT_FILE_TYPE:
                no_support_file.append(file_path)
                continue

            file_paths.append(item)
        condition_files = []
        file_paths = fnmatch.filter(file_paths, '*_*.*')

        # 处理文件， 位置0为组号， 位置1是文件路径
        for item in file_paths:
            # 提取组号
            group_index = int(item.split('_')[0]) - 1
            condition_files.append((group_index, os.path.join(condition_path, item)))

        self.condition_files = condition_files

        available_img_len = len(condition_files)
        print('\r' + ' ' * 110 + '\r', end='')
        print(f'read success, the number of available image is {available_img_len}')

        # # 读取支持的文件类型
        print('dataset: success')
        self.gray_trans = torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.condition_files)

    def __getitem__(self, item):
        group_index, condition_file = self.condition_files[item]

        # 读取混叠图
        condition_img = cv2.imread(condition_file, cv2.IMREAD_GRAYSCALE)

        # 读取真图
        real_path_1, real_path_2 = self.real_img_mat[group_index]
        real_img_1 = cv2.imread(real_path_1, cv2.IMREAD_GRAYSCALE)
        real_img_2 = cv2.imread(real_path_2, cv2.IMREAD_GRAYSCALE)

        real_img_1 = torchvision.transforms.ToTensor()(real_img_1)
        real_img_2 = torchvision.transforms.ToTensor()(real_img_2)
        real_img = torch.cat((real_img_1, real_img_2), 0)

        return real_img, torchvision.transforms.ToTensor()(condition_img)



class pic_128_dataset(Dataset):
    dataset_mode = {'TRAIN_MODE': ['train', 'train_r'],
                    'TEST_MODE': ['test', 'test_r']}
    def __init__(self, root, mode):

        super().__init__()
        print(f'The dataset will being loaded from “{root}” in here, please waiting...')
        print(f'Do be checking the legitimacy of “{root}”...')
        assert os.path.isdir(root)
        print('checked success, to be going to load dataset...')
        print(f'run_mode : {mode}')
        if mode not in pic_128_dataset.dataset_mode:
            raise RuntimeError(f'run_mode : {mode} is not supported, you can choose TRAIN_MODE, TEST_MODE or '
                               f'ALL_DATA_MODE')
        no_support_file = []  # 不支持的文件列表
        self.images = []

        # 获取文件名字
        print('reading fileNames...')
        file_paths = []
        condition_path = os.path.join(root, pic_128_dataset.dataset_mode[mode][0])
        self.con_path = condition_path
        self.real_path = os.path.join(root, pic_128_dataset.dataset_mode[mode][1])
        dir_list = os.listdir(condition_path)

        for item in dir_list:
            file_path = os.path.join(condition_path, item)
            # 如果不是文件，就终止当前的读入
            if not os.path.isfile(file_path):
                continue

            # 判断是不是支持的文件类型
            file_ext: str = os.path.splitext(file_path)[-1].lower()
            if file_ext not in SUPPORT_FILE_TYPE:
                no_support_file.append(file_path)
                continue

            file_paths.append(item)
        self.file_paths = file_paths

    def __getitem__(self, item):
        #获取文件名
        fileName = self.file_paths[item]

        #构造混叠图路径和真图路径
        con_filePath = os.path.join(self.con_path, fileName)
        real_filePath = os.path.join(self.real_path,fileName)

        #打开混叠图和真图
        con_img = cv2.imread(con_filePath, cv2.IMREAD_GRAYSCALE)
        real_img = cv2.imread(real_filePath, cv2.IMREAD_GRAYSCALE)

        con_img = torchvision.transforms.ToTensor()(con_img)
        real_img = torchvision.transforms.ToTensor()(real_img)

        return con_img, real_img

    def __len__(self):
        return len(self.file_paths)



def generator_dataset(x: torch.utils.data.DataLoader):
    while True:
        for item in x:
            yield item


if __name__ == '__main__':
    data = pic_128_dataset('./dataset/pictures-128', 'TRAIN_MODE')
    print(len(data))
    img = data[100][0]
    print(img)
    # data = My_dataset('./faces')
    # img = data[350]
    # print(len(data))
    # print(data[0].shape)


