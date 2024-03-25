import os
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import yaml

from nets.attention_net import AttentionNet, CriticNet
from util.torch_utils import seed_torch
from util.utils import get_logger
from data.eua_dataset import get_dataset


def train(config):
    # 这一行调用了一个函数 seed_torch()，它可能被用来设置 PyTorch 的随机种子，以确保实验的可重复性。
    seed_torch()
    # 这一行从一个名为 config 的配置文件中获取了三个部分的配置信息：train、data 和 model。
    # 这些配置信息可能包括训练过程的参数、数据集的参数以及模型的参数。
    train_config, data_config, model_config = config['train'], config['data'], config['model']
    # 这一行根据是否有可用的 CUDA（即 GPU 是否可用）来确定程序运行的设备。
    # 如果 CUDA 可用，就选择 GPU 作为设备，否则选择 CPU。
    device = train_config['device'] if torch.cuda.is_available() else 'cpu'
    # 这一行打印出程序最终选择的设备，以便用户知道程序将在哪个设备上运行
    print('Using device: {}'.format(device))
    # 这一行调用了 get_dataset 函数，并传递了一系列参数来生成数据集。
    # 这个函数可能会根据提供的参数生成数据集，然后对数据进行一些预处理。
    dataset = get_dataset(data_config['x_end'], data_config['y_end'], data_config['miu'], data_config['sigma'],
                          data_config['user_num'], data_config['data_size'],
                          data_config['min_cov'], data_config['max_cov'], device, train_config['dir_name'])
    # 这几行分别创建了训练数据加载器、验证数据加载器和测试数据加载器，用于按批次加载相应的数据集。
    # 这些加载器将在模型训练和评估过程中使用。其中，batch_size 参数指定了每个批次中样本的数量，
    # shuffle 参数表示是否在每个 epoch 开始时对数据进行洗牌以增加随机性。
    train_loader = DataLoader(dataset=dataset['train'], batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(dataset=dataset['valid'], batch_size=train_config['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset=dataset['test'], batch_size=train_config['batch_size'], shuffle=False)

    model = AttentionNet(6, 7, hidden_dim=model_config['hidden_dim'], device=device,
                         exploration_c=model_config['exploration_c'],
                         capacity_reward_rate=model_config['capacity_reward_rate'],
                         user_embedding_type=model_config['user_embedding_type'],
                         server_embedding_type=model_config['server_embedding_type'],
                         transformer_n_heads=model_config['transformer_n_heads'],
                         transformer_n_layers=model_config['transformer_n_layers'],
                         transformer_feed_forward_hidden=model_config['transformer_feed_forward_hidden'],
                         user_scale_alpha=model_config['user_scale_alpha'])
    # 使用 Adam 优化器来优化主模型 model 的参数，学习率为配置文件中指定的学习率 train_config['lr']。
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    # 获取训练类型，这个训练类型通常在配置文件中指定。
    original_train_type = train_config['train_type']
    # 根据原始的训练类型，将其赋值给 now_train_type，
    # 如果原始的训练类型是 'RGRB-BL'，则将 now_train_type 设置为 'REINFORCE'，否则保持原始的训练类型。
    if original_train_type == 'RGRB-BL':
        now_train_type = 'REINFORCE'
    else:
        now_train_type = original_train_type
    # 初始化变量 critic_model、critic_optimizer、critic_lr_scheduler 和 model_bl，并将它们都设置为 None。
    critic_model = None
    critic_optimizer = None
    critic_lr_scheduler = None
    model_bl = None
    # 如果 now_train_type 是 'ac'（Actor-Critic 训练），则：
    if now_train_type == 'ac':
        # 创建一个评论者网络 critic_model，其输入维度为用户和服务器的特征维度，
        # 隐藏层维度为 256，并且将用户和服务器的嵌入类型从模型配置文件中获取。
        critic_model = CriticNet(6, 7, 256, device, model['user_embedding_type'], model['server_embedding_type'])
        # 使用 Adam 优化器来优化评论者网络的参数，学习率与主模型相同。
        critic_optimizer = Adam(critic_model.parameters(), lr=train_config['lr'])
    elif original_train_type == 'RGRB-BL':
        # 根据配置文件中的参数创建一个辅助模型 model_bl，这个模型在特定的训练类型下被使用。
        model_bl = AttentionNet(6, 7, hidden_dim=model_config['hidden_dim'], device=device,
                                exploration_c=model_config['exploration_c'],
                                capacity_reward_rate=model_config['capacity_reward_rate'],
                                user_embedding_type=model_config['user_embedding_type'],
                                server_embedding_type=model_config['server_embedding_type'],
                                transformer_n_heads=model_config['transformer_n_heads'],
                                transformer_n_layers=model_config['transformer_n_layers'],
                                transformer_feed_forward_hidden=model_config['transformer_feed_forward_hidden'],
                                user_scale_alpha=model_config['user_scale_alpha'])

    # 加载需要继续训练或微调的模型
    # 检查模型配置中是否需要继续训练。如果是，则执行下面的代码块，否则跳到 else 语句。
    if model_config['need_continue']:
        # 从指定路径加载之前保存的模型参数、优化器状态等信息，
        checkpoint = torch.load(model_config['continue_model_filename'], map_location='cpu')
        # 加载之前训练模型的参数到当前模型中。
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 检查是否需要调整学习率。
        if model_config['continue_lr'] != 0:
            # 遍历优化器的参数组。
            for param_group in optimizer.param_groups:
                # 将每个参数组的学习率设置为配置文件中指定的继续学习率。
                param_group['lr'] = model_config['continue_lr']
        # 获取之前训练模型的最后一个 epoch，并将起始 epoch 设置为其后一个 epoch，用于继续训练。
        start_epoch = checkpoint['epoch'] + 1
        # 如果当前训练类型是 'ac'，则：
        if now_train_type == 'ac':
            # 加载之前训练的评论者网络参数。
            critic_model.load_state_dict(checkpoint['critic_model'])
            # 加载之前保存的评论者网络优化器状态。
            critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        elif original_train_type == 'RGRB-BL':
            # 将当前训练类型设置为 'RGRB-BL'
            now_train_type = 'RGRB-BL'
            # 加载之前训练的辅助模型参数。
            model_bl.load_state_dict(checkpoint['model_bl'])

        print("成功导入预训练模型")
    else:
        # 如果不需要继续训练，则将起始 epoch 设置为 0。
        start_epoch = 0
    # 创建学习率衰减器，用指数衰减方式调整学习率，train_config['lr_decay']是衰减率，
    # last_epoch=start_epoch - 1表示上一个 epoch 的索引。
    # 每轮乘lr_decay
    lr_scheduler = ExponentialLR(optimizer, train_config['lr_decay'], last_epoch=start_epoch - 1)
    print("当前学习率：", lr_scheduler.get_last_lr())
    # 如果当前训练类型是 'ac'，则创建评论者网络优化器的学习率衰减器
    if now_train_type == 'ac':
        critic_lr_scheduler = ExponentialLR(critic_optimizer, train_config['lr_decay'], last_epoch=start_epoch - 1)
    # 初始化评论者网络的指数移动平均值为零向量。
    critic_exp_mvg_avg = torch.zeros(1, device=device)
    # 生成保存训练过程中生成的日志和模型的目录名称，包括当前日期、用户数量、服务器数量等信息。
    dir_name = "" + time.strftime('%m%d%H%M', time.localtime(time.time())) \
               + "_server_" + str(data_config['x_end']) + "_" + str(data_config['y_end']) \
               + "_user_" + str(data_config['user_num']) \
               + "_miu_" + str(data_config['miu']) + "_sigma_" + str(data_config['sigma']) \
               + "_" + model_config['user_embedding_type'] + "_" + model_config['server_embedding_type'] \
               + "_" + original_train_type + "_capa_rate_" + str(model_config['capacity_reward_rate'])
    # 将目录名与根目录组合成完整的保存路径。
    dir_name = os.path.join(train_config['dir_name'], dir_name)
    # 设置日志文件的完整路径和名称。
    log_file_name = dir_name + '/log.log'
    # 创建保存训练过程中生成的日志和模型的目录，如果目录已存在则忽略。
    os.makedirs(dir_name, exist_ok=True)
    # 创建 TensorBoard 的 SummaryWriter，用于记录训练过程中的指标和可视化数据。
    tensorboard_writer = SummaryWriter(dir_name)
    # 获取记录日志的 logger。
    logger = get_logger(log_file_name)
    # 设置标志位表示当前不退出程序。
    now_exit = False

    start_time = time.time()
    all_valid_reward_list = []
    all_valid_user_list = []
    all_valid_server_list = []
    all_valid_capacity_list = []
    best_r = 0
    best_epoch_id = 0
    total_batch_num = 0
    for epoch in range(start_epoch, train_config['epochs']):
        # Train
        # 将模型设置为训练模式，这会启用训练相关的功能，如 dropout 和 batch normalization。
        model.train()
        # 这里设置了模型的策略为采样模式，并将 beam_num 设置为 1。
        model.policy = 'sample'
        model.beam_num = 1
        # 使用 enumerate(train_loader) 遍历训练数据集的每个 batch，并获取 server_seq、user_seq 和 masks。
        for batch_idx, (server_seq, user_seq, masks) in enumerate(train_loader):
            # 将获取的数据转移到指定的设备上，通常是 GPU。
            server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)
            # 使用模型进行推断，获取奖励、动作概率等信息。
            reward, actions_probs, _, user_allocated_props, server_used_props, capacity_used_props, _ \
                = model(user_seq, server_seq, masks)
            # 使用 TensorBoard 记录训练过程中的奖励和其他指标。
            tensorboard_writer.add_scalar('train/train_batch_reward', -torch.mean(reward), total_batch_num)
            total_batch_num += 1
            # 根据不同的训练类型，计算优势（advantage）。
            if now_train_type == 'REINFORCE':
                if batch_idx == 0:
                    critic_exp_mvg_avg = reward.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * train_config['beta']) \
                                         + ((1. - train_config['beta']) * reward.mean())
                advantage = reward - critic_exp_mvg_avg.detach()

            elif now_train_type == 'ac':
                critic_reward = critic_model(user_seq, server_seq)
                advantage = reward - critic_reward
                # 训练critic网络
                critic_loss = F.mse_loss(critic_reward, reward.detach()).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            elif now_train_type == 'SCST':
                model.policy = 'greedy'
                with torch.no_grad():
                    reward2, _, _, _, _, _, _ = model(user_seq, server_seq, masks)
                    advantage = reward - reward2
                model.policy = 'sample'

            elif now_train_type == 'RGRB-BL':
                with torch.no_grad():
                    reward2, _, _, _, _, _, _ = model_bl(user_seq, server_seq, masks)
                    advantage = reward - reward2

            else:
                raise NotImplementedError

            log_probs = torch.zeros(user_seq.size(0), device=device)
            for prob in actions_probs:
                log_prob = torch.log(prob)
                log_probs += log_prob
            log_probs[log_probs < -1000] = -1000.

            reinforce = torch.dot(advantage.detach(), log_probs)
            actor_loss = reinforce.mean()
            # 使用反向传播计算策略梯度，并根据优化器更新模型参数。
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
            # 使用 logger 记录训练过程中的信息，如当前 epoch、训练进度、奖励以及其他指标的均值。
            if batch_idx % int(1024 / train_config['batch_size']) == 0:
                logger.info(
                    'Epoch {}: Train [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                    '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                        epoch,
                        (batch_idx + 1) * len(user_seq),
                        data_config['data_size']['train'],
                        100. * (batch_idx + 1) / len(train_loader),
                        -torch.mean(reward),
                        torch.mean(user_allocated_props),
                        torch.mean(server_used_props),
                        torch.mean(capacity_used_props)))
        # 记录训练过程中的奖励（reward），取负号表示平均负奖励（因为通常优化目标是最大化奖励，
        # 但这里的奖励可能是成本，所以取负号表示最小化成本）。
        # 将均值 -torch.mean(reward) 记录到名为 'train/train_reward' 的标量中。
        # epoch 参数表示当前的训练 epoch，用于在 TensorBoard 中将不同 epoch 的数据进行对比和分析。
        tensorboard_writer.add_scalar('train/train_reward', -torch.mean(reward), epoch)
        # 记录训练过程中用户资源分配比例（user_allocated_props）的平均值。
        # 将平均值 torch.mean(user_allocated_props) 记录到名为 'train/train_user_allocated_props' 的标量中。
        tensorboard_writer.add_scalar('train/train_user_allocated_props', torch.mean(user_allocated_props), epoch)
        # 记录训练过程中服务器资源利用比例（server_used_props）的平均值。
        # 将平均值 torch.mean(server_used_props) 记录到名为 'train/train_server_used_props' 的标量中。
        tensorboard_writer.add_scalar('train/train_server_used_props', torch.mean(server_used_props), epoch)
        # 记录训练过程中容量利用比例（capacity_used_props）的平均值。
        # 将平均值 torch.mean(capacity_used_props) 记录到名为 'train/train_capacity_used_props' 的标量中。
        tensorboard_writer.add_scalar('train/train_capacity_used_props', torch.mean(capacity_used_props), epoch)

        # Valid and Test
        # 将模型设置为评估模式。
        model.eval()
        # 设置模型的策略为“贪婪”。
        model.policy = 'greedy'
        # 记录一条空白日志消息，用于分隔不同阶段的输出。
        logger.info('')
        # 在评估阶段不需要进行梯度计算，因此使用 torch.no_grad() 上下文管理器来禁用梯度计算。
        with torch.no_grad():
            # Validation
            # 初始化用于存储验证集结果的列表。
            valid_R_list = []
            valid_user_allocated_props_list = []
            valid_server_used_props_list = []
            valid_capacity_used_props_list = []
            # 设置模型的策略为“贪婪”，并设置束搜索的数量为 1。
            model.policy = 'greedy'
            model.beam_num = 1
            # 遍历验证集数据加载器，逐个获取批次数据。
            for batch_idx, (server_seq, user_seq, masks) in enumerate(valid_loader):
                # 将服务器序列、用户序列和掩码转移到指定的设备上。
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)
                # 通过模型计算验证集上的奖励和其他指标，如用户分配比例、服务器使用比例和容量使用比例。
                reward, _, _, user_allocated_props, server_used_props, capacity_used_props, _ \
                    = model(user_seq, server_seq, masks)
                # 记录验证集每个批次的性能指标，包括平均奖励、用户分配比例、服务器使用比例和容量使用比例。
                if batch_idx % int(1024 / train_config['batch_size']) == 0:
                    logger.info(
                        'Epoch {}: Valid [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                        '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                            epoch,
                            (batch_idx + 1) * len(user_seq),
                            data_config['data_size']['valid'],
                            100. * (batch_idx + 1) / len(valid_loader),
                            -torch.mean(reward),
                            torch.mean(user_allocated_props),
                            torch.mean(server_used_props),
                            torch.mean(capacity_used_props)
                        ))

                valid_R_list.append(reward)
                valid_user_allocated_props_list.append(user_allocated_props)
                valid_server_used_props_list.append(server_used_props)
                valid_capacity_used_props_list.append(capacity_used_props)
            # 这段代码将每个批次的验证集结果列表拼接成一个整体的张量。
            # valid_R_list 包含每个批次的验证集奖励，
            # valid_user_allocated_props_list 包含每个批次的用户分配比例，
            # valid_server_used_props_list 包含每个批次的服务器使用比例，
            # valid_capacity_used_props_list 包含每个批次的容量使用比例。
            valid_R_list = torch.cat(valid_R_list)
            valid_user_allocated_props_list = torch.cat(valid_user_allocated_props_list)
            valid_server_used_props_list = torch.cat(valid_server_used_props_list)
            valid_capacity_used_props_list = torch.cat(valid_capacity_used_props_list)
            valid_r = torch.mean(valid_R_list)
            valid_user_allo = torch.mean(valid_user_allocated_props_list)
            valid_server_use = torch.mean(valid_server_used_props_list)
            valid_capacity_use = torch.mean(valid_capacity_used_props_list)
            logger.info('Epoch {}: Valid \tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}\tcapacity_props:{:.6f}'
                        .format(epoch, -valid_r, valid_user_allo, valid_server_use, valid_capacity_use))

            tensorboard_writer.add_scalar('valid/valid_reward', -valid_r, epoch)
            tensorboard_writer.add_scalar('valid/valid_user_allocated_props', valid_user_allo, epoch)
            tensorboard_writer.add_scalar('valid/valid_server_used_props', valid_server_use, epoch)
            tensorboard_writer.add_scalar('valid/valid_capacity_used_props', valid_capacity_use, epoch)

            all_valid_reward_list.append(valid_r)
            all_valid_user_list.append(valid_user_allo)
            all_valid_server_list.append(valid_server_use)
            all_valid_capacity_list.append(valid_capacity_use)

            # 每次遇到更好的reward就保存一次模型，并且更新model_bl
            # 这段代码检查当前验证集的平均奖励是否比历史最佳奖励还要好。如果是，则执行以下操作。
            if valid_r < best_r:
                # 更新最佳奖励值 best_r、最佳轮次 best_epoch_id 和最佳时间 best_time。
                best_r = valid_r
                best_epoch_id = epoch
                best_time = 0
                # 记录日志，表示当前验证集奖励是历史最佳奖励。
                logger.info("目前本次reward最好\n")
                # 生成模型文件名，包含了当前时间以及相关的验证集指标信息。
                model_filename = dir_name + "/" + time.strftime(
                    '%m%d%H%M', time.localtime(time.time())
                ) + "_{:.2f}_{:.2f}_{:.2f}".format(all_valid_user_list[best_epoch_id - start_epoch] * 100,
                                                   all_valid_server_list[best_epoch_id - start_epoch] * 100,
                                                   all_valid_capacity_list[best_epoch_id - start_epoch] * 100) + '.mdl'
                # 保存当前模型到文件中，并记录日志。
                torch.save(model.state_dict(), model_filename)
                logger.info("模型已存储到: {}".format(model_filename))
                # 如果使用了基线模型更新策略（RGRB-BL），则加载最新保存的模型文件到 model_bl 中，并记录日志。
                if original_train_type == "RGRB-BL":
                    # 从文件复制回来，保证是深拷贝
                    state_checkpoint = torch.load(model_filename, map_location='cpu')
                    model_bl.load_state_dict(state_checkpoint)
                    logger.info("baseline已更新")
            else:
                # 如果当前验证集的奖励没有比历史最佳奖励更好，则增加记录无进展的轮次 best_time，并记录日志。
                best_time += 1
                logger.info("已经有{}轮效果没变好了\n".format(best_time))

            # Test
            # 测试阶段的代码开始。
            # 初始化存储测试集结果的列表。
            test_R_list = []
            test_user_allocated_props_list = []
            test_server_used_props_list = []
            test_capacity_used_props_list = []
            # 对测试数据加载器进行迭代，获取每个批次的服务器序列、用户序列和掩码。
            for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
                # 将数据移到指定的设备上进行计算。
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)
                # 利用模型计算测试集上的奖励以及相关属性。
                reward, _, _, user_allocated_props, server_used_props, capacity_used_props, _ \
                    = model(user_seq, server_seq, masks)
                # 在每个批次开始时记录测试集的奖励以及相关属性的平均值。
                if batch_idx % int(1024 / train_config['batch_size']) == 0:
                    logger.info(
                        'Epoch {}: Test [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                        '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                            epoch,
                            (batch_idx + 1) * len(user_seq),
                            data_config['data_size']['test'],
                            100. * (batch_idx + 1) / len(valid_loader),
                            -torch.mean(reward),
                            torch.mean(user_allocated_props),
                            torch.mean(server_used_props),
                            torch.mean(capacity_used_props)
                        ))
                # 将当前批次的测试结果添加到相应的列表中。
                test_R_list.append(reward)
                test_user_allocated_props_list.append(user_allocated_props)
                test_server_used_props_list.append(server_used_props)
                test_capacity_used_props_list.append(capacity_used_props)

            test_R_list = torch.cat(test_R_list)
            test_user_allocated_props_list = torch.cat(test_user_allocated_props_list)
            test_server_used_props_list = torch.cat(test_server_used_props_list)
            test_capacity_used_props_list = torch.cat(test_capacity_used_props_list)

            test_r = torch.mean(test_R_list)
            test_user_allo = torch.mean(test_user_allocated_props_list)
            test_server_use = torch.mean(test_server_used_props_list)
            test_capacity_use = torch.mean(test_capacity_used_props_list)

            logger.info('Epoch {}: Test \tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}\tcapacity_props:{:.6f}'
                        .format(epoch, -test_r, test_user_allo, test_server_use, test_capacity_use))
            tensorboard_writer.add_scalar('test/test_reward', -test_r, epoch)
            tensorboard_writer.add_scalar('test/test_user_allocated_props', test_user_allo, epoch)
            tensorboard_writer.add_scalar('test/test_server_used_props', test_server_use, epoch)
            tensorboard_writer.add_scalar('test/test_capacity_used_props', test_capacity_use, epoch)

        logger.info('')

        # 如果超过设定的epoch次数valid奖励都没有再提升，就停止训练；或者如果是RGRB-BL，就切换训练方式
        if best_time >= train_config['wait_best_reward_epoch'] or \
                (original_train_type == 'RGRB-BL' and now_train_type == 'REINFORCE'):
            # 保存一次可继续训练的模型就退出
            now_exit = True

        # 学习率衰减
        # 调用 lr_scheduler 对象的 step 方法，对优化器中的学习率进行衰减。
        lr_scheduler.step()
        # 如果当前的训练类型是 'ac'（Actor-Critic），则对评论家网络的学习率进行衰减，
        # 即调用 critic_lr_scheduler 对象的 step 方法。
        if now_train_type == 'ac':
            critic_lr_scheduler.step()
        # 记录学习率调整后的值。
        logger.info("学习率调整为：{}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        # 每interval个epoch，或者即将退出的时候，保存一次可继续训练的模型：
        if epoch % train_config['save_model_epoch_interval'] == train_config['save_model_epoch_interval'] - 1 \
                or now_exit:
            model_filename = dir_name + "/" + time.strftime(
                '%m%d%H%M', time.localtime(time.time())
            ) + "_{:.2f}_{:.2f}_{:.2f}".format(valid_user_allo * 100,
                                               valid_server_use * 100,
                                               valid_capacity_use * 100) + '.pt'
            if now_train_type == 'ac':
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                         'critic_model': critic_model.state_dict(),
                         'critic_optimizer': critic_optimizer.state_dict()}
            elif now_train_type == 'RGRB-BL':
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                         'model_bl': model_bl.state_dict()}
            else:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_filename)
            logger.info("模型已存储到: {}".format(model_filename))

            if now_exit:
                if original_train_type == 'RGRB-BL':
                    if now_train_type == 'REINFORCE':
                        now_train_type = 'RGRB-BL'
                        logger.info("REINFORCE已训练一轮，切换训练方式为RGRB-BL")
                        now_exit = False
                        best_time = 0
                    else:
                        break
                else:
                    break

    logger.info("效果如下：")
    for i in range(len(all_valid_reward_list)):
        logger.info("Epoch: {}\treward: {:.6f}\tuser_props: {:.6f}"
                    "\tserver_props: {:.6f}\tcapacity_props: {:.6f}"
                    .format(i + start_epoch, -all_valid_reward_list[i], all_valid_user_list[i],
                            all_valid_server_list[i], all_valid_capacity_list[i]))
    logger.info("训练结束，第{}个epoch效果最好，最好的reward: {} ，用户分配率: {:.2f} ，服务器租用率: {:.2f} ，资源利用率: {:.2f}"
                .format(best_epoch_id, -best_r,
                        all_valid_user_list[best_epoch_id - start_epoch] * 100,
                        all_valid_server_list[best_epoch_id - start_epoch] * 100,
                        all_valid_capacity_list[best_epoch_id - start_epoch] * 100))
    end_time = time.time()
    logger.info("训练时间: {:.2f}h".format(((end_time - start_time) / 3600)))
    logger.info("模型已存储到: {}".format(model_filename))


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        # 在这里，打开名为config.yaml的文件，模式为只读模式('r')，并且使用yaml.safe_load()方法加载其中的内容。
        # 加载后的配置将会被存储在loaded_config变量中。
        loaded_config = yaml.safe_load(f)

    train(loaded_config)
