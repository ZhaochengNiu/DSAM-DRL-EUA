import math
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm


workload_list = [
    np.array([1, 2, 1, 2]),
    np.array([2, 3, 3, 4]),
    np.array([5, 7, 6, 6])
]


def random_user_load():
    return random.choice(workload_list)


# 它首先计算用户和服务器之间的欧几里得距离（仅考虑 x 和 y 坐标）。
# 然后，它检查该距离是否小于或等于服务器的覆盖半径。如果是，则返回 True，
# 表示用户位于服务器的覆盖范围内；否则返回 False，表示用户不在覆盖范围内。
def in_coverage(user, server):
    return np.linalg.norm(user[:2] - server[:2]) <= server[2]


def get_within_servers(user_list, server_list, x_start, x_end, y_start, y_end):
    # user_list：用户坐标列表。
    # server_list：服务器坐标列表。
    # x_start、x_end、y_start、y_end：用于定义用户生成范围的值。
    """
    获取用户被哪些服务器覆盖，如果没有覆盖，则重新生成一个用户
    :return: 重新生成的用户列表；用户覆盖的服务器id
    """
    # 这里创建了一个二维数组 users_masks，用于记录每个用户被哪些服务器覆盖。数组的大小是 (用户数量, 服务器数量)。
    users_masks = np.zeros((len(user_list), len(server_list)), dtype=np.bool)
    # 这里定义了一个内部函数 calc_user_within，它用于检查特定用户是否被任何服务器覆盖。
    # 它接受一个用户的坐标和用户在 user_list 中的索引作为参数，并返回一个布尔值，
    # 指示用户是否被任何服务器覆盖。它遍历 server_list 中的每个服务器，
    # 如果用户在某个服务器的覆盖范围内，则将相应的 users_masks 的值设置为 1，
    # 并将 flag 设置为 True。

    def calc_user_within(calc_user, index):
        flag = False
        for j in range(len(server_list)):
            if in_coverage(calc_user, server_list[j]):
                users_masks[index, j] = 1
                flag = True
        return flag
    # 这是一个循环，遍历所有用户。对于每个用户，它首先调用 calc_user_within 函数来检查用户是否被任何服务器覆盖。
    # 如果用户没有被覆盖，则重新生成用户的位置，直到找到一个在范围内的位置，并且该用户被至少一个服务器覆盖。
    for i in range(len(user_list)):
        user = user_list[i]
        user_within = calc_user_within(user, i)
        while not user_within:
            user[0] = random.random() * (x_end - x_start) + x_start
            user[1] = random.random() * (y_end - y_start) + y_start
            user_within = calc_user_within(user, i)
    return user_list, users_masks


def get_whole_capacity(user_list, rate):
    """
    获取所有用户的总需求再乘比例
    :param user_list: 用户列表
    :param rate: 资源冗余比例
    :return: 需要的总资源
    """
    capacity = np.sum(user_list[:, 2:], axis=0) * rate
    return capacity


def evaluate_whole_capacity_by_user_num(user_num, rate=3):
    """
    根据用户数量预估总容量
    :param user_num:
    :param rate:
    :return:
    """
    # 首先，它将负载列表转换为 NumPy 数组，并计算每个负载类型的平均负载。
    loads = np.array(workload_list)
    average_load = loads.mean(axis=0)
    # 然后，它根据用户数量和容量系数，计算整个系统的总负载。最终返回整个系统的总负载。
    whole_load = average_load * user_num * rate
    return whole_load


# 为每个服务器分配capacity
def allocate_capacity(server_list, capacity):
    # 获取服务器列表的长度，即服务器数量
    server_len = len(server_list)
    # 根据平均资源容量 capacity，计算每种资源的最小值和最大值，这些值将用于随机分配资源容量。
    # 考虑到一定的波动范围，最大容量为平均容量的1.25倍，最小容量为平均容量的0.75倍。
    # 使用循环遍历服务器列表，为每个服务器分配随机值的资源容量。随机值的范围在前面计算得到的最小值和最大值之间。
    # 将分配资源后的服务器列表返回。
    # 对服务器的平均资源添加一些随机因素
    # 最小是0.75倍，最大是1.25倍
    # 计算每种资源的最小值和最大值，以平均资源为基准，并添加一些随机因素
    cpu_max = int(capacity[0] * 1.25 / server_len) # CPU 最大容量
    cpu_min = int(capacity[0] * 0.75 / server_len) # CPU 最小容量
    io_max = int(capacity[1] * 1.25 / server_len) # IO 最大容量
    io_min = int(capacity[1] * 0.75 / server_len) # IO 最小容量
    bandwidth_max = int(capacity[2] * 1.25 / server_len) # 带宽 最大容量
    bandwidth_min = int(capacity[2] / 2 / server_len) # 带宽 最小容量
    memory_max = int(capacity[3] * 1.25 / server_len) # 内存 最大容量
    memory_min = int(capacity[3] / 2 / server_len) # 内存 最小容量
    # 遍历服务器列表，并为每个服务器分配资源容量
    for server in server_list:
        # 为每种资源分配随机值在最小值和最大值范围内
        server[3] = random.randint(cpu_min, cpu_max)
        server[4] = random.randint(io_min, io_max)
        server[5] = random.randint(bandwidth_min, bandwidth_max)
        server[6] = random.randint(memory_min, memory_max)
    return server_list


# 绘制服务器和用户的分布情况。具体解释如下：
def draw_data(server_list, user_list):
    # 创建一个新的图形
    fig = plt.figure()
    # 在图形上添加子图
    ax = fig.add_subplot(111)
    # 绘制用户点
    for user in user_list:
        ax.plot(user[0], user[1], 'ro')
    # 绘制服务器圆和点
    for server in server_list:
        # 创建一个圆形对象，以服务器坐标为中心，半径为服务器的覆盖半径，并设置透明度为0.2
        circle = Circle((server[0], server[1]), server[2], alpha=0.2)
        # 将圆形对象添加到子图上
        ax.add_patch(circle)
        # 绘制服务器点，使用蓝色圆圈表示服务器
        ax.plot(server[0], server[1], 'bo')
    # 设置绘图区域的缩放和比例
    plt.axis('scaled')
    plt.axis('equal')
    # 显示图形
    plt.show()


# 用于计算每个样本的用户分配比例和服务器利用率。
def cal_props_by_seqs(user_seqs, server_seqs, user_allocated_servers):
    # 获取批量数据的大小
    batch_size = user_seqs.shape[0]
    # 存储每个样本的用户分配比例和服务器利用率
    user_allocated_props = []
    server_used_props = []
    # 遍历每个样本
    for i in range(batch_size):
        # 获取当前样本的用户序列、服务器序列和分配的服务器序列
        user_seq = user_seqs[i]
        server_seq = server_seqs[i]
        allocated_seq = user_allocated_servers[i]
        # 计算当前样本的用户分配比例和服务器利用率
        user_allocated_prop, server_used_prop = cal_props(user_seq, server_seq, allocated_seq)
        # 将计算得到的用户分配比例和服务器利用率添加到列表中
        user_allocated_props.append(user_allocated_prop)
        server_used_props.append(server_used_prop)
    # 返回所有样本的用户分配比例和服务器利用率
    return user_allocated_props, server_used_props


def can_allocate(workload, capacity):
    for i in range(4):
        if capacity[i] < workload[i]:
            return False
    return True


# 用于计算用户分配比例和服务器利用率。
def cal_props(user_seqs, server_seqs, allocated_seq):
    # 获取每个服务器的剩余资源容量
    tmp_server_capacity = [server_seq[3:] for server_seq in server_seqs]
    # 获取用户序列的长度（用户数量）
    user_num = len(user_seqs)
    # 获取服务器序列的长度（服务器数量）
    server_num = len(server_seqs)
    # 真实分配情况
    # 初始化真实分配情况：用户分配服务器的情况和每个服务器被分配的用户数量
    user_allocate_list = [-1] * user_num # 用户是否分配到服务器，-1表示未分配
    server_allocate_num = [0] * server_num # 每个服务器被分配的用户数量
    # 遍历每个用户
    for i in range(user_num):
        user_seq = user_seqs[i] # 获取当前用户的序列
        server_id = allocated_seq[i] # 获取当前用户被分配的服务器ID
        if server_id == -1:    # 如果用户未被分配到服务器，则跳过
            continue
        # 如果当前用户在当前分配的服务器的覆盖范围内，并且服务器有足够的资源容量来分配给用户
        if in_coverage(user_seq[:2], server_seqs[server_id][:3]) and can_allocate(user_seq[2:],
                                                                                  tmp_server_capacity[server_id]):
            # 标记当前用户分配到了当前服务器
            user_allocate_list[i] = server_id
            server_allocate_num[server_id] += 1
            # 更新当前服务器的剩余资源容量
            for j in range(4):
                tmp_server_capacity[server_id][j] -= user_seq[2 + j]

    # 已分配用户占所有用户的比例
    allocated_user_num = user_num - user_allocate_list.count(-1)  # 已分配用户的数量
    user_allocated_prop = allocated_user_num / user_num # 已分配用户占所有用户的比例

    # 已使用服务器占所有服务器比例
    used_server_num = server_num - server_allocate_num.count(0) # 已使用的服务器数量
    server_used_prop = used_server_num / server_num # 使用服务器占所有服务器的比例

    return user_allocated_prop, server_used_prop


# 获取所有服务器
def get_all_server_xy():
    # 创建一个空列表来存储服务器的地理坐标
    server_list = []
    # 打开 CSV 文件以读取服务器的地理坐标信息
    file = open("data/site-optus-melbCBD.csv", 'r')

    file.readline().strip()  # 数据集的第一行是字段说明信息，不能作为数据，因此跳过
    # 逐行读取文件中的数据
    lines = file.readlines()
    # 遍历文件的每一行数据
    for i in range(len(lines)):
        # 使用逗号分隔符拆分每一行数据，并将结果存储在 result 中
        result = lines[i].split(',')
        # longitude, latitude
        # 从 result 中提取经度和纬度信息并转换为浮点数格式，存储在 server_mes 中
        server_mes = (float(result[2]), float(result[1]))
        # 将经度和纬度转换为特定坐标系下的坐标，并将结果存储在 x 和 y 中
        x, y = miller_to_xy(*server_mes)
        # 将经度和纬度转换后的坐标存储为列表 [x, y]，并添加到 server_list 中
        server_list.append([x, y])
    #  # 关闭文件
    file.close()
    # 将 server_list 转换为 NumPy 数组，以便进行进一步处理
    server_list = np.array(server_list)
    # 找到 server_list 中的最小坐标值，并将所有坐标转换为相对于最小值的相对坐标
    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy
    # 将服务器坐标进行旋转以适应模型的要求
    angel = 13
    for xy in server_list:
        x = xy[0] * math.cos(math.pi / 180 * angel) - xy[1] * math.sin(math.pi / 180 * angel)
        y = xy[0] * math.sin(math.pi / 180 * angel) + xy[1] * math.cos(math.pi / 180 * angel)
        xy[0] = x
        xy[1] = y

    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy
    # 进一步调整服务器坐标以适应模型的要求
    for xy in server_list:
        xy[0] = xy[0] - xy[1] * math.tan(math.pi / 180 * 15)
    # 再次找到 server_list 中的最小坐标值，并将所有坐标转换为相对于最小值的相对坐标
    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy
    # 单位：米转换为100米
    # 将坐标的单位从米转换为 100 米
    server_list /= 100
    # 返回转换后的服务器坐标列表
    return server_list


def miller_to_xy(lon, lat):
    """
    :param lon: 经度
    :param lat: 维度
    :return:
    """
    L = 6381372 * math.pi * 2  # 地球周长
    W = L  # 平面展开，将周长视为X轴
    H = L / 2  # Y轴约等于周长一半
    mill = 2.3  # 米勒投影中的一个常数，范围大约在正负2.3之间
    x = lon * math.pi / 180  # 将经度从度数转换为弧度
    y = lat * math.pi / 180

    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))  # 这里是米勒投影的转换

    # 这里将弧度转为实际距离
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    return x, y


def init_server(x_start_prop, x_end_prop, y_start_prop, y_end_prop,
                min_cov=1, max_cov=1.5, miu=35, sigma=10):
    # 参数中 min_cov 和 max_cov 是服务器的最小和最大覆盖范围，而 miu 和 sigma 则是服务器容量的均值和标准差。
    """
    根据比例从地图中截取一些服务器的坐标
    """
    # 调用 get_all_server_xy 函数获取所有服务器的坐标列表。
    server_xy_list = get_all_server_xy()
    # 计算服务器坐标列表中的最大 x 和 y 坐标。
    max_x_y = np.max(server_xy_list, axis=0)
    # 计算服务器坐标列表中的最大 x 和 y 坐标。
    max_x = max_x_y[0]
    max_y = max_x_y[1]
    # 根据传入的比例计算要截取的服务器坐标的范围。
    x_start = max_x * x_start_prop
    x_end = max_x * x_end_prop
    y_start = max_y * y_start_prop
    y_end = max_y * y_end_prop
    # 筛选出符合要求的服务器坐标。
    filter_server = [x_start <= server[0] <= x_end
                     and y_start <= server[1] <= y_end
                     for server in server_xy_list]
    # 根据筛选结果更新服务器坐标列表。
    server_xy_list = server_xy_list[filter_server]
    # 将这些服务器最左上角定义为(0,0)+覆盖范围
    # 将服务器坐标列表中的坐标进行偏移，使得最左上角的服务器位置为 (0, 0)，并增加最大覆盖范围。
    min_xy = np.min(server_xy_list, axis=0)
    server_xy_list = server_xy_list - min_xy + max_cov
    # 生成服务器的覆盖范围列表，从 min_cov 到 max_cov 之间均匀分布。
    server_cov_list = np.random.uniform(min_cov, max_cov, (len(server_xy_list), 1))
    # 生成服务器的容量列表，使用正态分布，均值为 miu，标准差为 sigma。
    server_capacity_list = np.random.normal(miu, sigma, size=(len(server_xy_list), 4))
    # 将服务器的坐标、覆盖范围和容量列表合并成一个数组。
    server_list = np.concatenate((server_xy_list, server_cov_list, server_capacity_list), axis=1)
    return server_list


def init_users_list_by_server(server_list, data_num, user_num, load_sorted=True, max_cov=1.5):
    # 用于在固定的服务器坐标上生成一组用户，并补充服务器的资源容量信息。
    # 参数中 server_list 是服务器的坐标列表，data_num 指定要生成多少组用户数据，
    # user_num 指定每组用户的数量，load_sorted 控制是否按负载排序用户，max_cov 是最大覆盖半径。
    """
    固定服务器坐标，生成一组user，同时补充服务器的资源容量
    :param server_list:
    :param data_num: 生成多少组
    :param user_num: 用户数
    :param max_cov: 最大覆盖半径，给左上角坐标加上，以免用户只能在左上角第一个服务器的右下角1/4的范围内生成
    :param load_sorted: 是否直接生成已按load排序的用户
    :return:
    """
    # 在服务器列表中找到最大和最小的 x、y 坐标，并根据 max_cov 计算出生成用户的 x 和 y 范围。
    max_server = np.max(server_list, axis=0)
    max_x = max_server[0] + max_cov
    max_y = max_server[1] + max_cov
    min_server = np.min(server_list, axis=0)
    min_x = min_server[0] - max_cov
    min_y = min_server[1] - max_cov
    # 初始化用于存储生成用户列表和掩码列表的空列表。
    users_list = []
    users_masks_list = []
    # 循环生成指定数量的数据组。
    for _ in tqdm(range(data_num)):
        # 使用均匀分布在生成的范围内随机生成用户的 x 和 y 坐标。
        user_x_list = np.random.uniform(min_x, max_x, (user_num, 1))
        user_y_list = np.random.uniform(min_y, max_y, (user_num, 1))
        # 如果 load_sorted 为真，则按照一定比例生成已排序的用户负载；否则，随机生成用户负载。
        if load_sorted:
            num01 = int(1 / 3 * user_num)
            num2 = user_num - 2 * num01
            w0 = np.tile(workload_list[0], (num01, 1))
            w1 = np.tile(workload_list[1], (num01, 1))
            w2 = np.tile(workload_list[2], (num2, 1))
            user_load_list = np.concatenate((w0, w1, w2), axis=0)
        else:
            user_load_list = np.array([random_user_load() for _ in range(user_num)])
        # 将用户的 x 和 y 坐标以及负载列表连接成用户列表。
        user_list = np.concatenate((user_x_list, user_y_list, user_load_list), axis=1)
        # 调用 get_within_servers 函数，将用户限制在服务器的覆盖范围内，并生成相应的用户掩码。
        user_list, users_masks = get_within_servers(user_list, server_list, min_x, max_x, min_y, max_y)
        # 将生成的用户列表和用户掩码列表添加到对应的列表中。
        users_list.append(user_list)
        users_masks_list.append(users_masks)
    # 返回一个字典，包含生成的用户列表和用户掩码列表。
    return {"users_list": users_list, "users_masks_list": users_masks_list}
