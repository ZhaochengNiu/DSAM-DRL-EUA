import os
import numpy as np
import torch
from torch.utils.data import Dataset

from data.data_generator import init_server, init_users_list_by_server
from util.utils import save_dataset


# 定义了一个名为 EuaTrainDataset 的新类，它是 PyTorch Dataset 类的子类。这个类表示用于训练模型的数据集。
class EuaTrainDataset(Dataset):
    #servers: 包含关于服务器的信息的列表。
    # users_list: 包含关于用户的信息的列表。
    # users_within_servers_list: 指示每个用户由哪些服务器覆盖的列表。
    # users_masks_list: 包含表示用户覆盖情况的掩码的列表。
    # device: 数据将存储在的设备（CPU 或 GPU）。
    def __init__(self, servers, users_list, users_within_servers_list, users_masks_list, device):
        # 这一行将 servers 列表转换为 PyTorch 张量，并将其赋值给数据集对象的 servers 属性。
        # 张量使用指定的数据类型 (torch.float32) 和设备 (device) 存储。
        self.servers = torch.tensor(servers, dtype=torch.float32, device=device)
        # 这些行将 users_list、users_within_servers_list 和 users_masks_list 参数分配给数据集对象的属性。
        # 这些列表包含关于用户、每个用户由哪些服务器覆盖以及表示用户覆盖情况的掩码的信息。
        self.users_list, self.users_within_servers_list, self.users_masks_list = \
            users_list, users_within_servers_list, users_masks_list
        # 将 device 参数分配给数据集对象的 device 属性，指示数据将存储在的设备。
        self.device = device

    def __len__(self):
        # 这个方法返回数据集的长度，由数据集中的用户数量确定。
        return len(self.users_list)

    # 这个方法允许对数据集进行索引，以获取给定索引处的特定样本。
    def __getitem__(self, index):
        # 这些行将指定索引处的用户信息和掩码信息转换为 PyTorch 张量。用户信息存储在 user_seq 中，
        # 数据类型为 torch.float32，掩码信息存储在 mask_seq 中，数据类型为 torch.bool。
        # 这两个张量都存储在指定的 device 上。
        user_seq = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
        # 返回一个元组，其中包含表示给定索引处数据样本的服务器张量 (servers)、用户张量 (user_seq) 和掩码张量 (mask_seq)。
        return self.servers, user_seq, mask_seq


class EuaDataset(Dataset):
    def __init__(self, servers, users_list, users_masks_list, device):
        # servers：服务器列表，包含了服务器的信息。
        # users_list：用户列表，包含了每个用户的信息。
        # users_masks_list：用户掩码列表，用于表示用户的覆盖情况。
        # device：设备信息，指定了数据集存储在哪个设备上（如 CPU 或 GPU）。
        # 这行代码将构造方法中传入的参数存储在对象的属性中，分别为 servers、users_list 和 users_masks_list。
        self.servers, self.users_list, self.users_masks_list = servers, users_list, users_masks_list
        # 这行代码将服务器列表 servers 转换为 PyTorch 张量 servers_tensor，并将其存储在对象的属性中。
        # 张量的数据类型为 torch.float32，设备为构造方法中指定的设备。
        self.servers_tensor = torch.tensor(servers, dtype=torch.float32, device=device)
        # 这行代码将构造方法中传入的设备信息存储在对象的 device 属性中。
        self.device = device

    # 这个方法定义了数据集对象的长度。它返回用户列表 users_list 的长度，即数据集中样本的数量。
    def __len__(self):
        return len(self.users_list)

    # 这是一个特殊方法，用于根据给定的索引 index 获取数据集中指定索引处的数据样本。
    def __getitem__(self, index):
        # 这两行代码分别将用户列表中索引为 index 的用户信息和掩码信息转换为 PyTorch 张量。
        # 用户信息存储在 user_seq 中，掩码信息存储在 mask_seq 中。
        # 张量的数据类型分别为 torch.float32 和 torch.bool，设备为构造方法中指定的设备。
        user_seq = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
        # 这行代码返回一个元组，其中包含三个张量：
        # 第一个张量是服务器张量 servers_tensor，表示数据集中的服务器信息。
        # 第二个张量是用户信息张量 user_seq，表示数据集中的用户信息。
        # 第三个张量是掩码信息张量 mask_seq，表示数据集中的用户覆盖情况。
        return self.servers_tensor, user_seq, mask_seq


# 表示需要排序的 EUA 数据集。
class EuaDatasetNeedSort(Dataset):
    # 初始化服务器列表、用户列表和用户掩码列表，并将它们存储在对象属性中
    def __init__(self, servers, users_list, users_masks_list, device):
        self.servers, self.users_list, self.users_masks_list = servers, users_list, users_masks_list
        # 将服务器列表转换为张量，并存储在对象属性中
        self.servers_tensor = torch.tensor(servers, dtype=torch.float32, device=device)
        # 存储设备信息
        self.device = device

    def __len__(self):
        # 返回数据集的样本数量（即用户列表的长度）
        return len(self.users_list)

    def __getitem__(self, index):
        # 先排序
        # 获取指定索引处的数据样本

        # 获取原始的用户列表和掩码列表
        original_users = self.users_list[index]
        original_masks = self.users_masks_list[index]
        # 将用户列表和掩码列表打包成元组，然后转换为列表
        x = zip(original_users, original_masks)
        x = list(x)
        # 根据用户的负载排序列表x
        x = sorted(x, key=lambda u: u[0][2])
        # 将排序后的用户列表和掩码列表拆分为新的列表
        new_user, new_mask = zip(*x)
        # 将用户序列和掩码序列转换为张量，并存储在对象属性中
        user_seq = torch.tensor(np.array(new_user, dtype=np.float), dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(np.array(new_mask, dtype=np.bool), dtype=torch.bool, device=self.device)
        # 返回服务器张量、用户张量和掩码张量的元组作为数据样本
        return self.servers_tensor, user_seq, mask_seq


def get_dataset(x_end, y_end, miu, sigma, user_num, data_size: {}, min_cov, max_cov, device, dir_name):
    """
    获取dataset
    :param x_end:
    :param y_end:
    :param miu:
    :param sigma:
    :param user_num:
    :param data_size: 字典，key为dataset类型，value为该类型的数量
    :param min_cov:
    :param max_cov:
    :param device:
    :param dir_name: 数据集存放的文件夹
    :return:
    """
    # 构建数据集目录的完整路径，该目录包含服务器数据文件。
    dataset_dir_name = os.path.join(dir_name,
                                    "dataset/server_" + str(x_end) + "_" + str(y_end)
                                    + "_miu_" + str(miu) + "_sigma_" + str(sigma))
    # 构建服务器数据文件的完整路径。
    server_file_name = "server_" + str(x_end) + "_" + str(y_end) + "_miu_" + str(miu) + "_sigma_" + str(sigma)
    server_path = os.path.join(dataset_dir_name, server_file_name) + '.npy'
    # 检查是否存在服务器数据文件，如果存在则加载数据，否则调用 init_server 函数生成服务器数据并保存。
    if os.path.exists(server_path):
        servers = np.load(server_path)
        print("读取服务器数据成功")
    else:
        print("未读取到服务器数据，重新生成")
        os.makedirs(dataset_dir_name, exist_ok=True)
        servers = init_server(0, x_end, 0, y_end, min_cov, max_cov, miu, sigma)
        np.save(server_path, servers)
    # 获取数据集类型（训练集、验证集和测试集），并初始化一个空字典用于存储数据集。
    set_types = data_size.keys()
    datasets = {}
    # 遍历数据集类型，如果发现未知的数据集类型则抛出 NotImplementedError 异常。
    for set_type in set_types:
        if set_type not in ('train', 'valid', 'test'):
            raise NotImplementedError
        # 构建数据集文件的完整路径。
        filename = set_type + "_user_" + str(user_num) + "_size_" + str(data_size[set_type])
        path = os.path.join(dataset_dir_name, filename) + '.npz'
        # 检查数据集文件是否存在，如果存在则加载数据，否则调用 init_users_list_by_server 函数生成用户数据并保存。
        if os.path.exists(path):
            print("正在加载", set_type, "数据集")
            data = np.load(path)
        else:
            print(set_type, "数据集未找到，重新生成", path)
            data = init_users_list_by_server(servers, data_size[set_type], user_num, True, max_cov)
            save_dataset(path, **data)
        # 创建 EuaDataset 类的实例，并将其存储在 datasets 字典中。
        datasets[set_type] = EuaDataset(servers, **data, device=device)

    return datasets


def shuffle_dataset(test_set):
    # 初始化新的用户列表和掩码列表
    new_users = []
    new_masks = []
    # 遍历测试集中的每个样本
    for i in range(len(test_set)):
        # 将用户列表和掩码列表打包成元组，并转换为列表
        x = zip(test_set.users_list[i], test_set.users_masks_list[i])
        x = list(x)
        # 对列表中的元素进行随机打乱
        np.random.shuffle(x)
        # 将打乱后的结果拆分为新的用户列表和掩码列表
        new_user, new_mask = zip(*x)
        # 将新的用户列表和掩码列表添加到相应的列表中
        new_users.append(new_user)
        new_masks.append(new_mask)
    # 将新的用户列表和掩码列表转换为数组
    new_users_array = np.stack(new_users)
    new_masks_array = np.stack(new_masks)
    # 返回包含新数据的 EuaDataset 对象
    return EuaDataset(test_set.servers, new_users_array, new_masks_array, test_set.device)
