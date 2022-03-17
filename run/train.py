import os
import pickle
import time
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from nets.attention_net import PointerNet
from nets.attention_net import CriticNet
from util.utils import log_and_print
from data.eua_dataset import generate_three_set


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    seed_torch()
    batch_size = 256
    use_cuda = True
    lr = 3e-4
    beta = 0.9
    max_grad_norm = 2.
    epochs = 1000
    dropout = 0
    capacity_reward_rate = 0.2
    user_num = 300
    x_end = 0.5
    y_end = 1
    min_cov = 1
    max_cov = 1.5
    miu = 35
    sigma = 10
    user_embedding_type = 'transformer'
    server_embedding_type = 'linear'
    # train_type = 'REINFORCE'
    # train_type = 'ac'
    train_type = 'RGRB'
    train_size = 100000
    valid_size = 10000
    test_size = 10000
    wait_best_reward_epoch = 10
    save_model_epoch_interval = 10
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    need_continue = True
    continue_model_filename = r"D:\transformer_eua\model" \
                              r"\03142102_server_0.5_1_user_200_miu_35_sigma_10_transformer_linear_RGRB_rate_0.2" \
                              r"\03160100_97.61_59.00_48.95.mdl "

    train_filename = "D:/transformer_eua/dataset/train_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                     + str(user_num) + "_miu_" + str(miu) + "_sigma_" + str(sigma) + "_size_" + str(train_size) + ".pkl"
    valid_filename = "D:/transformer_eua/dataset/valid_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                     + str(user_num) + "_miu_" + str(miu) + "_sigma_" + str(sigma) + "_size_" + str(valid_size) + ".pkl"
    test_filename = "D:/transformer_eua/dataset/test_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                    + str(user_num) + "_miu_" + str(miu) + "_sigma_" + str(sigma) + "_size_" + str(test_size) + ".pkl"
    server_filename = "D:/transformer_eua/dataset/server_" + str(x_end) + "_" + str(y_end) \
                      + "_miu_" + str(miu) + "_sigma_" + str(sigma) + ".pkl"

    try:
        print("正在加载训练数据集")
        with open(train_filename, 'rb') as f:
            train_set = pickle.load(f)
        print("正在加载验证数据集")
        with open(valid_filename, 'rb') as f:
            valid_set = pickle.load(f)
        print("正在加载测试数据集")
        with open(test_filename, 'rb') as f:
            test_set = pickle.load(f)
    except FileNotFoundError as e:
        print("文件{}未找到，重新生成".format(e.filename))
        try:
            with open(server_filename, 'rb') as f:
                servers = pickle.load(f)
            print("成功读取到原有服务器信息")
        except FileNotFoundError as e2:
            servers = None
            print("未读取到原有服务器信息，重新生成")
        train_set, valid_set, test_set, servers = \
            generate_three_set(user_num, (train_size, valid_size, test_size),
                               0, x_end, 0, y_end, device, min_cov, max_cov, miu, sigma, servers)
        with open(server_filename, 'wb') as f:
            pickle.dump(servers, f)
            print("服务器信息保存成功")
        with open(train_filename, 'wb') as f:
            pickle.dump(train_set, f)
            print("保存训练集成功")

        with open(valid_filename, 'wb') as f:
            pickle.dump(valid_set, f)
            print("保存验证集成功")
        with open(test_filename, 'wb') as f:
            pickle.dump(test_set, f)
            print("保存测试集成功")

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    model = PointerNet(6, 7, 256, device=device, dropout=dropout, capacity_reward_rate=capacity_reward_rate,
                       user_embedding_type=user_embedding_type, server_embedding_type=server_embedding_type)
    optimizer = Adam(model.parameters(), lr=lr)

    critic_model = None
    critic_optimizer = None
    if train_type == 'ac':
        critic_model = CriticNet(6, 7, 256, device, dropout, 'transformer', 'linear')
        critic_optimizer = Adam(critic_model.parameters(), lr=lr)

    # 加载需要继续训练的模型
    if need_continue:
        checkpoint = torch.load(continue_model_filename)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        if train_type == 'ac':
            critic_model.load_state_dict(checkpoint['critic_model'])
            critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        print("成功导入预训练模型")
    else:
        start_epoch = 0

    critic_exp_mvg_avg = torch.zeros(1, device=device)

    board_dir_name = "D:/transformer_eua/log/" + time.strftime('%m%d%H%M', time.localtime(time.time())) \
                     + "_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                     + str(user_num) + "_miu_" + str(miu) + "_sigma_" + str(sigma) + "_" + user_embedding_type + "_" \
                     + server_embedding_type + "_" + train_type + "_capa_rate_" + str(capacity_reward_rate)
    log_file_name = board_dir_name + '/log.log'
    model_dir_name = "D:/transformer_eua/model/" + time.strftime('%m%d%H%M', time.localtime(time.time())) \
                     + "_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                     + str(user_num) + "_miu_" + str(miu) + "_sigma_" + str(sigma) + "_" + user_embedding_type + "_" \
                     + server_embedding_type + "_" + train_type + "_capa_rate_" + str(capacity_reward_rate)
    os.makedirs(model_dir_name, exist_ok=True)
    os.makedirs(board_dir_name, exist_ok=True)
    tensorboard_writer = SummaryWriter(board_dir_name)

    all_valid_reward_list = []
    all_valid_user_list = []
    all_valid_server_list = []
    all_valid_capacity_list = []
    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        model.policy = 'sample'
        model.beam_num = 1
        for batch_idx, (server_seq, user_seq, masks) in enumerate(train_loader):
            server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

            reward, actions_probs, action_idx, user_allocated_props, \
                server_used_props, capacity_used_props, user_allocate_list \
                = model(user_seq, server_seq, masks)

            if train_type == 'REINFORCE':
                if batch_idx == 0:
                    critic_exp_mvg_avg = reward.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * reward.mean())
                advantage = reward - critic_exp_mvg_avg

            elif train_type == 'ac':
                critic_reward = critic_model(user_seq, server_seq)
                advantage = reward - critic_reward
                # 训练critic网络
                critic_loss = F.mse_loss(critic_reward, reward.detach()).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            elif train_type == 'RGRB':
                model.policy = 'greedy'
                with torch.no_grad():
                    reward2, actions_probs2, action_idx2, user_allocated_props2, \
                        server_used_props2, capacity_used_props2, user_allocate_list2 \
                        = model(user_seq, server_seq, masks)
                    advantage = reward - reward2
                model.policy = 'sample'

            else:
                raise NotImplementedError

            log_probs = torch.zeros(user_seq.size(0), device=device)
            for prob in actions_probs:
                log_prob = torch.log(prob)
                log_probs += log_prob
            log_probs[log_probs < -1000] = -1000.

            reinforce = torch.dot(advantage.detach(), log_probs)
            actor_loss = reinforce.mean()

            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm), norm_type=2)
            optimizer.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            if batch_idx % int(2048 / batch_size) == 0:
                log_and_print(
                    '{} Epoch {}: Train [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                    '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                        time.strftime('%H:%M:%S', time.localtime(time.time())),
                        epoch,
                        (batch_idx + 1) * len(user_seq),
                        train_size,
                        100. * (batch_idx + 1) / len(train_loader),
                        torch.mean(reward),
                        torch.mean(user_allocated_props),
                        torch.mean(server_used_props),
                        torch.mean(capacity_used_props)),
                    log_file_name)

        tensorboard_writer.add_scalar('train/train_reward', torch.mean(reward), epoch)
        tensorboard_writer.add_scalar('train/train_user_allocated_props', torch.mean(user_allocated_props), epoch)
        tensorboard_writer.add_scalar('train/train_server_used_props', torch.mean(server_used_props), epoch)
        tensorboard_writer.add_scalar('train/train_capacity_used_props', torch.mean(capacity_used_props), epoch)

        # Valid and Test
        model.eval()
        model.policy = 'greedy'
        log_and_print('', log_file_name)
        with torch.no_grad():
            # Validation
            valid_R_list = []
            valid_user_allocated_props_list = []
            valid_server_used_props_list = []
            valid_capacity_used_props_list = []
            model.policy = 'greedy'
            model.beam_num = 1
            for batch_idx, (server_seq, user_seq, masks) in enumerate(valid_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                reward, _, action_idx, user_allocated_props, \
                    server_used_props, capacity_used_props, user_allocate_list \
                    = model(user_seq, server_seq, masks)

                if batch_idx % int(2048 / batch_size) == 0:
                    log_and_print(
                        '{} Epoch {}: Valid [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                        '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                            time.strftime('%H:%M:%S', time.localtime(time.time())),
                            epoch,
                            (batch_idx + 1) * len(user_seq),
                            valid_size,
                            100. * (batch_idx + 1) / len(valid_loader),
                            torch.mean(reward),
                            torch.mean(user_allocated_props),
                            torch.mean(server_used_props),
                            torch.mean(capacity_used_props)
                        ),
                        log_file_name)

                valid_R_list.append(reward)
                valid_user_allocated_props_list.append(user_allocated_props)
                valid_server_used_props_list.append(server_used_props)
                valid_capacity_used_props_list.append(capacity_used_props)

            valid_R_list = torch.cat(valid_R_list)
            valid_user_allocated_props_list = torch.cat(valid_user_allocated_props_list)
            valid_server_used_props_list = torch.cat(valid_server_used_props_list)
            valid_capacity_used_props_list = torch.cat(valid_capacity_used_props_list)
            valid_r = torch.mean(valid_R_list)
            valid_user_allo = torch.mean(valid_user_allocated_props_list)
            valid_server_use = torch.mean(valid_server_used_props_list)
            valid_capacity_use = torch.mean(valid_capacity_used_props_list)
            log_and_print('{} Epoch {}: Valid \tR:{:.6f}\tuser_props: {:.6f}'
                          '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'
                          .format(time.strftime('%H:%M:%S', time.localtime(time.time())), epoch, valid_r,
                                  valid_user_allo, valid_server_use, valid_capacity_use),
                          log_file_name)

            tensorboard_writer.add_scalar('valid/valid_reward', valid_r, epoch)
            tensorboard_writer.add_scalar('valid/valid_user_allocated_props', valid_user_allo, epoch)
            tensorboard_writer.add_scalar('valid/valid_server_used_props', valid_server_use, epoch)
            tensorboard_writer.add_scalar('valid/valid_capacity_used_props', valid_capacity_use, epoch)

            all_valid_reward_list.append(valid_r)
            all_valid_user_list.append(valid_user_allo)
            all_valid_server_list.append(valid_server_use)
            all_valid_capacity_list.append(valid_capacity_use)
            if len(all_valid_reward_list) == 1:
                best_r = valid_r
                best_user = valid_user_allo
                best_server = valid_server_use
                best_capacity = valid_capacity_use
                if train_type == 'ac':
                    best_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                                  'critic_model': critic_model.state_dict(),
                                  'critic_optimizer': critic_optimizer.state_dict()}
                else:
                    best_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                best_time = 0
            else:
                if valid_r < best_r:
                    best_r = valid_r
                    best_user = valid_user_allo
                    best_server = valid_server_use
                    best_capacity = valid_capacity_use
                    best_time = 0
                    log_and_print("目前本次reward最好\n", log_file_name)
                else:
                    best_time += 1
                    log_and_print("已经有{}轮效果没变好了\n".format(best_time), log_file_name)

            # Test
            test_R_list = []
            test_user_allocated_props_list = []
            test_server_used_props_list = []
            test_capacity_used_props_list = []
            for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                reward, _, action_idx, user_allocated_props, \
                    server_used_props, capacity_used_props, user_allocate_list \
                    = model(user_seq, server_seq, masks)

                if batch_idx % int(2048 / batch_size) == 0:
                    log_and_print(
                        '{} Epoch {}: Test [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                        '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                            time.strftime('%H:%M:%S', time.localtime(time.time())),
                            epoch,
                            (batch_idx + 1) * len(user_seq),
                            test_size,
                            100. * (batch_idx + 1) / len(valid_loader),
                            torch.mean(reward),
                            torch.mean(user_allocated_props),
                            torch.mean(server_used_props),
                            torch.mean(capacity_used_props)
                        ),
                        log_file_name)

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
            log_and_print('{} Epoch {}: Test \tR:{:.6f}\tuser_props: {:.6f}'
                          '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'
                          .format(time.strftime('%H:%M:%S', time.localtime(time.time())), epoch, test_r,
                                  test_user_allo, test_server_use, test_capacity_use),
                          log_file_name)
            tensorboard_writer.add_scalar('test/test_reward', test_r, epoch)
            tensorboard_writer.add_scalar('test/test_user_allocated_props', test_user_allo, epoch)
            tensorboard_writer.add_scalar('test/test_server_used_props', test_server_use, epoch)
            tensorboard_writer.add_scalar('test/test_capacity_used_props', test_capacity_use, epoch)

            log_and_print('', log_file_name)
            torch.cuda.empty_cache()

            # 如果超过设定的epoch次数valid奖励都没有再提升，就停止训练
            if best_time >= wait_best_reward_epoch:
                log_and_print("效果如下：", log_file_name)
                for i in range(len(all_valid_reward_list)):
                    log_and_print("Epoch: {}\treward: {:.6f}\tuser_props: {:.6f}"
                                  "\tserver_props: {:.6f}\tcapacity_props: {:.6f}"
                                  .format(i, all_valid_reward_list[i], all_valid_user_list[i],
                                          all_valid_server_list[i], all_valid_capacity_list[i]),
                                  log_file_name)
                model_filename = model_dir_name + "/" + time.strftime(
                    '%m%d%H%M', time.localtime(time.time())
                ) + "_{:.2f}_{:.2f}_{:.2f}".format(best_user * 100, best_server * 100, best_capacity * 100) + '.mdl'
                torch.save(best_state, model_filename)
                log_and_print("模型已存储到: {}".format(model_filename), log_file_name)
                log_and_print("训练结束，最好的reward:{}，用户分配率:{}，服务器租用率:{}，资源利用率:{}"
                              .format(best_r, best_user, best_server, best_capacity), log_file_name)
                exit()

            # 每interval个epoch保存一次模型：
            if epoch % save_model_epoch_interval == save_model_epoch_interval - 1:
                model_filename = model_dir_name + "/" + time.strftime(
                    '%m%d%H%M', time.localtime(time.time())
                ) + "_{:.2f}_{:.2f}_{:.2f}".format(valid_user_allo * 100,
                                                   valid_server_use * 100,
                                                   valid_capacity_use * 100) + '.mdl'
                if train_type == 'ac':
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                             'critic_model': critic_model.state_dict(),
                             'critic_optimizer': critic_optimizer.state_dict()}
                else:
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, model_filename)
                log_and_print("模型已存储到: {}".format(model_filename), log_file_name)
