import yaml

from data.eua_dataset import get_dataset


def main_get_dataset():
    # 使用 with 上下文管理器打开名为 config.yaml 的配置文件，并将其读取为字典形式的配置信息，
    # 存储在变量 config 中。yaml.safe_load() 函数用于安全地加载 YAML 文件。
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # 这部分代码定义了数据集生成所需的一些参数，包括：
    # dir_name：数据集保存的目录名称，从配置文件中获取。
    # user_num：用户数量，设置为 800。
    # x_end 和 y_end：数据生成区域的边界。
    # min_cov 和 max_cov：服务器的最小和最大覆盖范围。
    # miu 和 sigma：高斯分布的均值和标准差，用于生成用户的位置。
    # data_size：数据集大小的字典，包括训练集、验证集和测试集的样本数量。
    dir_name = config['train']['dir_name']
    user_num = 800
    x_end = 0.5
    y_end = 1
    min_cov = 1
    max_cov = 1.5
    miu = 35
    sigma = 10
    data_size = {
        'train': 100000,
        'valid': 10000,
        'test': 10000
    }
    # 调用名为 get_dataset 的函数，传入上述参数，以生成数据集。
    get_dataset(x_end, y_end, miu, sigma, user_num, data_size, min_cov, max_cov, 'cpu', dir_name)


if __name__ == '__main__':
    main_get_dataset()
