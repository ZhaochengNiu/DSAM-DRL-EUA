import math

import torch
import torch.nn as nn
from nets.graph_encoder import GraphAttentionEncoder


class UserEncoder(nn.Module):
    # input_dim: 输入特征的维度。
    # hidden_dim: 隐藏层的维度，即神经网络中间层的神经元数量。
    # n_heads: 注意力头的数量，默认为 8。
    # n_layers: 注意力层的数量，默认为 6。
    # normalization: 规范化方式，默认为批次规范化。
    # feed_forward_hidden: 前馈神经网络隐藏层的维度，默认为 512。
    # embedding_type: 嵌入类型，指定了用户特征的编码方式，默认为 "transformer"。
    def __init__(self, input_dim, hidden_dim, n_heads=8, n_layers=6,
                 normalization='batch', feed_forward_hidden=512, embedding_type='transformer'):
        super(UserEncoder, self).__init__()
        # 设置嵌入类型。
        self.embedding_type = embedding_type
        # 如果嵌入类型是 "transformer"，
        # 则创建一个名为 GraphAttentionEncoder 的实例，表示采用图注意力编码器来对用户特征进行编码。
        if embedding_type == 'transformer':
            self.embedding = GraphAttentionEncoder(n_heads, hidden_dim, n_layers,
                                                   input_dim, normalization, feed_forward_hidden)
        # 如果嵌入类型是 "linear"，则创建一个线性层，用于线性变换用户特征的维度。
        elif embedding_type == 'linear':
            self.embedding = nn.Linear(input_dim, hidden_dim)
        # 如果嵌入类型是 "lstm"，则创建一个 LSTM 层，用于对序列特征进行编码。
        elif embedding_type == 'lstm':
            self.embedding = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # 否则，直接使用相应的嵌入层对输入进行编码，并返回编码后的结果。
        else:
            raise NotImplementedError

    def forward(self, inputs):
        if self.embedding_type == 'lstm':
            embedded, _ = self.embedding(inputs)
            return embedded
        else:
            return self.embedding(inputs)


class ServerEncoder(nn.Module):
    # input_dim：输入特征的维度。
    # hidden_dim：隐藏层的维度。
    # n_heads：多头注意力中注意头的数量，默认为 8。
    # n_layers：Transformer 层的数量，默认为 6。
    # normalization：归一化方法，默认为 'batch'。
    # feed_forward_hidden：前馈神经网络中隐藏层的大小，默认为 512。
    # embedding_type：嵌入类型，可以是 'transformer' 或 'linear'。
    def __init__(self, input_dim, hidden_dim, n_heads=8, n_layers=6,
                 normalization='batch', feed_forward_hidden=512, embedding_type='linear'):
        super(ServerEncoder, self).__init__()
        # 将嵌入类型存储在实例变量 embedding_type 中。
        self.embedding_type = embedding_type
        # 根据嵌入类型选择不同的编码方式。
        # 如果嵌入类型为 'transformer'，则使用 GraphAttentionEncoder 进行编码。
        # 如果嵌入类型为 'linear'，则使用线性层进行编码。
        # 根据嵌入类型选择相应的编码器。
        # 如果是 'transformer'，则创建一个 GraphAttentionEncoder 实例；如果是 'linear'，则创建一个线性层。
        if embedding_type == 'transformer':
            self.embedding = GraphAttentionEncoder(n_heads, hidden_dim, n_layers,
                                                   input_dim, normalization, feed_forward_hidden)
        elif embedding_type == 'linear':
            self.embedding = nn.Linear(input_dim, hidden_dim)
        else:
            # 如果嵌入类型不是 'transformer' 或 'linear'，则抛出 NotImplementedError。
            raise NotImplementedError

    def forward(self, inputs):
        # 根据选择的编码器类型，对输入张量进行编码并返回结果。
        return self.embedding(inputs)


class Glimpse(nn.Module):
    # Glimpse 的模块，用于在查询和参考（参考序列）之间进行注意力机制。下面是逐行解释：
    # input :
    # query:    batch_size * 1 * query_input_dim
    # ref:      batch_size * seq_len * ref_hidden_dim
    def __init__(self, hidden_dim):
        super(Glimpse, self).__init__()
        # 使用线性层初始化查询、参考键和参考值的线性转换。这些线性层用于将输入的隐藏维度映射到相同的隐藏维度。
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        # 用于归一化注意力分数的归一化因子。
        self._norm_fact = 1 / math.sqrt(hidden_dim)

    def forward(self, query, ref):
        # 分别将查询、参考键和参考值通过线性层进行转换，以便在相同的隐藏维度上进行计算。
        Q = self.q(query)  # Q: batch_size * 1 * hidden_dim
        K = self.k(ref)  # K: batch_size * seq_len * hidden_dim
        V = self.v(ref)  # V: batch_size * seq_len * hidden_dim
        # 计算查询和参考之间的点积注意力分数。
        # 这里使用了批量矩阵乘法（torch.bmm），并且为了进行矩阵乘法，需要将参考键进行转置以与查询相匹配。
        attn_score = torch.bmm(Q, K.permute(0, 2, 1))
        # 使用归一化因子对注意力分数进行归一化，以防止数值过大。
        attn_score = attn_score * self._norm_fact
        # 将归一化后的注意力分数通过 softmax 函数进行归一化，以得到查询与参考之间的注意力分布。
        attn = torch.softmax(attn_score, dim=-1)    # Q * K.T() # batch_size * 1 * seq_len
        # 使用注意力分布对参考值进行加权平均，得到最终的输出。这里同样使用了批量矩阵乘法，将注意力分布乘以参考值。
        output = torch.bmm(attn, V)  # Q * K.T() * V # batch_size * 1 * hidden_dim
        # 混合了所有服务器的相似度的一个表示服务器的变量
        return output


class Attention(nn.Module):
    def __init__(self, hidden_dim, exploration_c=10, user_scale_alpha=0.05):
        super(Attention, self).__init__()
        self.hidden_size = hidden_dim
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vt = nn.Linear(hidden_dim, 1, bias=False)
        self.exploration_c = exploration_c
        self.user_scale_alpha = user_scale_alpha

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1, hidden_size)
        decoder_transform = self.W2(decoder_state * self.user_scale_alpha)

        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts
        # log_softmax for a better numerical stability
        score = u_i.masked_fill(~mask, value=torch.log(torch.tensor(1e-45))) * self.exploration_c
        prob = torch.softmax(score, dim=-1)
        return prob


class AttentionNet(nn.Module):
    def __init__(self, user_input_dim, server_input_dim, hidden_dim, device, capacity_reward_rate, exploration_c=10,
                 policy='sample', user_embedding_type='transformer', server_embedding_type='linear',
                 transformer_n_heads=8, transformer_n_layers=3, transformer_feed_forward_hidden=512,
                 user_scale_alpha=0.05, beam_num=1, ):
        super(AttentionNet, self).__init__()
        # user_input_dim: 用户输入特征的维度。
        # server_input_dim: 服务器输入特征的维度。
        # hidden_dim: 隐藏层的维度，即神经网络中间层的神经元数量。
        # device: 指定模型所在的设备，可以是 GPU 或 CPU。
        # capacity_reward_rate: 容量奖励率，用于奖励模型在任务中实现更高容量的能力。
        # exploration_c: 探索系数，用于控制模型在训练过程中的探索能力，默认值为 10。
        # policy: 策略，控制模型的行为策略，默认为 "sample"。
        # user_embedding_type: 用户嵌入类型，用于指定如何对用户进行嵌入表示，默认为 "transformer"。
        # server_embedding_type: 服务器嵌入类型，用于指定如何对服务器进行嵌入表示，默认为 "linear"。
        # transformer_n_heads: Transformer 模型中注意力头的数量，默认为 8。
        # transformer_n_layers: Transformer 模型中注意力层的数量，默认为 3。
        # transformer_feed_forward_hidden: Transformer 模型中前馈神经网络隐藏层的维度，默认为 512。
        # user_scale_alpha: 用户尺度参数 α，用于缩放用户特征的参数，默认值为 0.05。
        # beam_num: 光束搜索的数量，默认为 1。
        # decoder hidden size
        # 设置隐藏层维度和设备。
        self.hidden_dim = hidden_dim
        self.device = device
        # 创建用户编码器，根据给定的参数创建一个 UserEncoder 类的实例，并将其放置到指定的设备上。
        self.user_encoder = UserEncoder(user_input_dim, hidden_dim, n_heads=transformer_n_heads,
                                        n_layers=transformer_n_layers,
                                        feed_forward_hidden=transformer_feed_forward_hidden,
                                        embedding_type=user_embedding_type).to(device)
        # 创建服务器编码器，根据给定的参数创建一个 ServerEncoder 类的实例，并将其放置到指定的设备上。
        self.server_encoder = ServerEncoder(server_input_dim + 1, hidden_dim, n_heads=transformer_n_heads,
                                            n_layers=transformer_n_layers,
                                            feed_forward_hidden=transformer_feed_forward_hidden,
                                            embedding_type=server_embedding_type).to(device)
        # 创建关注机制模块，根据给定的参数创建一个 Glimpse 类的实例，并将其放置到指定的设备上。
        # glimpse输入（用户，上次选择的服务器），维度为2*dim， 跟所有的服务器作相似度并输出融合后的服务器
        self.glimpse = Glimpse(hidden_dim).to(device)
        # 创建注意力机制模块，根据给定的参数创建一个 Attention 类的实例，并将其放置到指定的设备上。
        # 设置模型的容量奖励率、策略和光束搜索数量。
        self.pointer = Attention(hidden_dim, exploration_c, user_scale_alpha).to(device)
        self.capacity_reward_rate = capacity_reward_rate
        self.policy = policy
        self.beam_num = beam_num

    def choose_server_id(self, mask, user, static_server_seq, tmp_server_capacity, server_active):
        """
        每一步根据用户和所有服务器，输出要选择的服务器
        """
        server_seq = torch.cat((static_server_seq, tmp_server_capacity, server_active), dim=-1)
        server_encoder_outputs = self.server_encoder(server_seq)
        server_glimpse = self.glimpse(user, server_encoder_outputs)
        server_glimpse = server_glimpse

        # get a pointer distribution over the encoder outputs using attention
        # (batch_size, server_len)
        probs = self.pointer(server_glimpse, server_encoder_outputs, mask)
        # (batch_size, server_len)

        if self.policy == 'sample':
            # (batch_size, 1)
            idx = torch.multinomial(probs, num_samples=self.beam_num)
            prob = torch.gather(probs, dim=1, index=idx)
        elif self.policy == 'greedy':
            prob, idx = torch.topk(probs, k=self.beam_num, dim=-1)
        else:
            raise NotImplementedError

        if self.beam_num == 1:
            prob = prob.squeeze(1)
            idx = idx.squeeze(1)

        return prob, idx

    @staticmethod
    def update_server_capacity(server_id, tmp_server_capacity, user_workload):
        batch_size = server_id.size(0)
        # 取出一个batch里所有第j个用户选择的服务器
        index_tensor = server_id.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, 4)  # 4个资源维度
        j_th_server_capacity = torch.gather(tmp_server_capacity, dim=1, index=index_tensor).squeeze(1)
        # (batch_size)的True，False矩阵
        can_be_allocated = can_allocate(user_workload, j_th_server_capacity)
        # 如果不能分配容量就不减
        mask = can_be_allocated.unsqueeze(-1).expand(batch_size, 4)
        # 切片时指定batch和server_id对应关系
        batch_range = torch.arange(batch_size)
        # 服务器减去相应容量
        tmp_server_capacity[batch_range, server_id] -= user_workload * mask
        # 记录服务器分配情况，即server_id和mask的内积
        server_id = torch.masked_fill(server_id, mask=~can_be_allocated, value=-1)
        return tmp_server_capacity, server_id

    @staticmethod
    def calc_rewards(user_allocate_list, user_len, server_allocate_mat, server_len,
                     original_servers_capacity, batch_size, tmp_server_capacity):
        # 目前user_allocate_list是(batch_size, user_len)
        # 计算每个分配的用户数，即不是-1的个数，(batch_size)
        user_allocate_num = torch.sum(user_allocate_list != -1, dim=1)
        user_allocated_props = user_allocate_num.float() / user_len

        server_used_num = torch.sum(server_allocate_mat[:, :-1], dim=1)
        server_used_props = server_used_num.float() / server_len

        # 已使用的服务器的资源利用率
        server_allocated_flag = server_allocate_mat[:, :-1].unsqueeze(-1).expand(batch_size, server_len, 4)
        # (batch_size, server_len, 4)
        used_original_server = original_servers_capacity.masked_fill(~server_allocated_flag.bool(), value=0)
        # (batch_size, server_len, 4)
        servers_remain_capacity = tmp_server_capacity.masked_fill(~server_allocated_flag.bool(), value=0)
        # 对于每个维度的资源求和，得到的结果应该是(batch_size, 4)，被压缩的维度是1
        sum_all_capacity = torch.sum(used_original_server, dim=1)
        sum_remain_capacity = torch.sum(servers_remain_capacity, dim=1)
        # 对于每个维度的资源求资源利用率
        every_capacity_remain_props = torch.div(sum_remain_capacity, sum_all_capacity)
        mean_capacity_remain_props = torch.mean(every_capacity_remain_props, dim=1)
        capacity_used_props = 1 - mean_capacity_remain_props
        return user_allocated_props, server_used_props, capacity_used_props

    def forward(self, user_input_seq, server_input_seq, masks):
        if self.beam_num != 1:
            return self.beam_forward(user_input_seq, server_input_seq, masks)

        batch_size = user_input_seq.size(0)
        user_len = user_input_seq.size(1)
        server_len = server_input_seq.size(1)

        # 真实分配情况
        user_allocate_list = -torch.ones(batch_size, user_len, dtype=torch.long, device=self.device)
        # 服务器分配矩阵，加一是为了给index为-1的来赋值
        server_allocate_mat = torch.zeros(batch_size, server_len + 1, dtype=torch.long, device=self.device)

        # 服务器信息由三部分组成
        static_server_seq = server_input_seq[:, :, :3]
        tmp_server_capacity = server_input_seq[:, :, 3:].clone()

        user_encoder_outputs = self.user_encoder(user_input_seq)

        action_probs = []
        action_idx = []

        for i in range(user_len):
            mask = masks[:, i]
            user_code = user_encoder_outputs[:, i, :].unsqueeze(1)
            prob, idx = self.choose_server_id(mask, user_code, static_server_seq, tmp_server_capacity,
                                              server_allocate_mat[:, :-1].unsqueeze(-1))

            action_probs.append(prob)
            action_idx.append(idx)

            tmp_server_capacity, idx = self.update_server_capacity(idx, tmp_server_capacity, user_input_seq[:, i, 2:])

            # 真实分配情况
            user_allocate_list[:, i] = idx
            # 给分配了的服务器在服务器分配矩阵中赋值为True
            batch_range = torch.arange(batch_size)
            server_allocate_mat[batch_range, idx] = 1

        action_probs = torch.stack(action_probs)
        action_idx = torch.stack(action_idx, dim=-1)

        user_allocated_props, server_used_props, capacity_used_props = \
            self.calc_rewards(user_allocate_list, user_len, server_allocate_mat, server_len,
                              server_input_seq[:, :, 3:].clone(), batch_size, tmp_server_capacity)

        return -self.get_reward(user_allocated_props, capacity_used_props), action_probs, \
            action_idx, user_allocated_props, server_used_props, capacity_used_props, user_allocate_list

    def beam_forward(self, user_input_seq, server_input_seq, masks):
        batch_size = user_input_seq.size(0)
        user_len = user_input_seq.size(1)
        server_len = server_input_seq.size(1)
        b = self.beam_num
        batch_range = torch.arange(batch_size)

        # 真实分配情况
        user_allocate_lists = -torch.ones(batch_size, b, user_len, dtype=torch.long, device=self.device)
        # 服务器分配矩阵，加一是为了给index为-1的来赋值
        server_allocate_mats = torch.zeros(batch_size, b, server_len + 1, dtype=torch.long, device=self.device)

        # 服务器信息由三部分组成
        static_server_seq = server_input_seq[:, :, :3]
        now_server_capacities = torch.clone(server_input_seq[:, :, 3:])

        user_encoder_outputs = self.user_encoder(user_input_seq)

        # 上一步已经选定的b个可能性的log和
        action_probs_list = torch.zeros(batch_size, b, device=self.device)
        # 上一步已经选定的b个路线 (batch_size, b, user_len)
        action_idxes = None
        # 上一步的服务器剩余容量 (batch_size, b, server_len, server_dim)
        now_server_capacities = now_server_capacities.repeat(b, 1, 1, 1).permute(1, 0, 2, 3)

        for i in range(user_len):
            # 新建3个缓存器
            # 这一步产生的b*b个可能的路线
            tmp_action_idxes = torch.zeros(batch_size, b, b, i + 1, dtype=torch.long, device=self.device)
            # 这一步产生的b*b个可能性的log和
            tmp_probs_list = torch.zeros(batch_size, b, b, device=self.device)

            for m in range(b):
                mask = masks[:, i]
                user_code = user_encoder_outputs[:, i, :].unsqueeze(1)
                # prob: (batch_size, b); idx: (batch_size, b)
                prob, idx = self.choose_server_id(mask, user_code, static_server_seq,
                                                  now_server_capacities[:, m],
                                                  server_allocate_mats[:, m, :-1].unsqueeze(-1))

                # 此时产生了b个结果，把所有结果都先放在tmp中
                prob = torch.log(prob)
                # 此时这b个结果的概率都和上次的概率相乘，即log相加
                prob = prob + action_probs_list[:, m].unsqueeze(-1)

                # 如果是第一个用户分配，那它分配只有b种可能作为分支的开始，也就不需要tmp来缓存b*b个
                if i == 0:
                    action_probs_list = prob
                    # (batch_size, b, 1)
                    action_idxes = idx.unsqueeze(-1)
                    break

                # b*b的概率缓存矩阵中第m行设置为现在的b个概率
                tmp_probs_list[batch_range, m, :] = prob
                # b*b的路径缓存矩阵中第m行设置为现在的b个路径
                # (batch_size, b, i) -> (batch_size, b, i+1)
                # 取原来的路径
                # (batch_size, i) = (batch_size, b, i)取第二维
                old_idxes = action_idxes[:, m, :]
                # 给现在的路径连接上原来的路径，现在路径的维度是(batch_size, b)
                # 要给5个过去的路径分别连上现在的idx，所以过去的路径要*5, 变成(batch_size, b, i)
                old_idxes = old_idxes.unsqueeze(1).repeat(1, b, 1)
                # 现在路径在最后补一维，变成(batch_size, b, 1)
                idx = idx.unsqueeze(-1)
                # 最后变成(batch_size, b, i+1)
                now_b_action_idxes = torch.cat([old_idxes, idx], dim=-1)
                # (batch_size, b, b, i+1)
                tmp_action_idxes[batch_range, m, :] = now_b_action_idxes

            if i != 0:
                # 此时b*b个结果都已经产生，选出其中b个可能性最大的：
                tmp_probs_list = tmp_probs_list.view(batch_size, b * b)
                # 得到使probs最大的index就能用torch.gather取出动作
                tmp_action_idxes = tmp_action_idxes.view(batch_size, b * b, i + 1)
                prob_values, indices = torch.topk(tmp_probs_list, dim=1, k=b)
                # 更新probs存储器
                action_probs_list = torch.gather(tmp_probs_list, dim=1, index=indices)
                # 更新路径存储器
                indices1 = indices.unsqueeze(-1).expand(batch_size, b, i + 1)
                action_idxes = torch.gather(tmp_action_idxes, dim=1, index=indices1)
                # 更新对应的tmp_capacity
                now_server_capacities = now_server_capacities.unsqueeze(2)
                now_server_capacities = now_server_capacities.expand(batch_size, b, b, server_len, 4)
                now_server_capacities = now_server_capacities.reshape(batch_size, b * b, server_len, 4)
                indices2 = indices.view(batch_size, b, 1, 1)
                indices2 = indices2.expand(batch_size, b, server_len, 4)
                now_server_capacities = torch.gather(now_server_capacities, dim=1, index=indices2)
                # 更新真实分配情况
                user_allocate_lists = user_allocate_lists.unsqueeze(2)
                user_allocate_lists = user_allocate_lists.expand(batch_size, b, b, user_len)
                user_allocate_lists = user_allocate_lists.reshape(batch_size, b * b, user_len)
                indices3 = indices.view(batch_size, b, 1)
                indices3 = indices3.expand(batch_size, b, user_len)
                user_allocate_lists = torch.gather(user_allocate_lists, dim=1, index=indices3)
                # 更新服务器使用情况
                server_allocate_mats = server_allocate_mats.unsqueeze(2)
                server_allocate_mats = server_allocate_mats.expand(batch_size, b, b, server_len + 1)
                server_allocate_mats = server_allocate_mats.reshape(batch_size, b * b, server_len + 1)
                indices4 = indices.view(batch_size, b, 1)
                indices4 = indices4.expand(batch_size, b, server_len + 1)
                server_allocate_mats = torch.gather(server_allocate_mats, dim=1, index=indices4)

            for m in range(b):
                # 对现有列表里面的b个结果进行服务器容量更新
                next_server_capacity, idx = self.update_server_capacity(action_idxes[:, m, -1],
                                                                        now_server_capacities[:, m].clone(),
                                                                        user_input_seq[:, i, 2:])
                now_server_capacities[batch_range, m] = next_server_capacity
                # 真实分配情况
                user_allocate_lists[batch_range, m, i] = idx
                # 给分配了的服务器在服务器分配矩阵中赋值为True
                server_allocate_mats[batch_range, m, idx] = 1

        # 最后计算所有的奖励
        user_allo = []
        server_use = []
        capacity_use = []
        rewards = []
        for m in range(b):
            # 所有用户分配完后，把各种指标stack成tensor
            user_allocated_props, server_used_props, capacity_used_props = \
                self.calc_rewards(user_allocate_lists[:, m], user_len, server_allocate_mats[:, m], server_len,
                                  server_input_seq[:, :, 3:].clone(), batch_size, now_server_capacities[:, m])
            reward = (user_allocated_props + capacity_used_props)
            user_allo.append(user_allocated_props)
            server_use.append(server_used_props)
            capacity_use.append(capacity_used_props)
            rewards.append(reward)

        # (batch_size, b)
        rewards = torch.stack(rewards, dim=-1)
        user_allo = torch.stack(user_allo, dim=-1)
        server_use = torch.stack(server_use, dim=-1)
        capacity_use = torch.stack(capacity_use, dim=-1)

        max_reward_value, indices = torch.max(rewards, dim=-1)
        indices = indices.unsqueeze(-1)
        max_user_allo = torch.gather(user_allo, dim=1, index=indices)
        max_server_use = torch.gather(server_use, dim=1, index=indices)
        max_capacity_use = torch.gather(capacity_use, dim=1, index=indices)
        indices = indices.unsqueeze(-1).expand(batch_size, 1, user_len)
        best_idxs = torch.gather(action_idxes, dim=1, index=indices)
        best_user_allocate_lists = torch.gather(user_allocate_lists, dim=1, index=indices)

        return -self.get_reward(max_user_allo, max_capacity_use), \
            action_probs_list, best_idxs, max_user_allo, max_server_use, max_capacity_use, best_user_allocate_lists

    def get_reward(self, user_allocated_props, capacity_used_props):
        return (1 - self.capacity_reward_rate) * user_allocated_props + self.capacity_reward_rate * capacity_used_props


def can_allocate(workload: torch.Tensor, capacity: torch.Tensor):
    """
    计算能不能分配并返回分配情况
    :param workload: (batch, 4)
    :param capacity: (batch, 4)
    :return:
    """
    # (batch, 4)
    bools = capacity >= workload
    # (batch)，bool值
    return bools.all(dim=1)


class CriticNet(nn.Module):
    # 初始化方法，接受用户输入维度 user_input_dim、服务器输入维度 server_input_dim、
    # 隐藏层维度 hidden_dim、设备 device，以及用户和服务器的嵌入类型等参数。
    def __init__(self, user_input_dim, server_input_dim, hidden_dim, device,
                 user_embedding_type='linear', server_embedding_type='linear'):
        super(CriticNet, self).__init__()
        # 将输入的隐藏层维度和设备信息存储在对象属性中。
        self.hidden_dim = hidden_dim
        self.device = device
        # 创建用户编码器和服务器编码器对象。这些编码器用于将用户输入和服务器输入映射到隐藏空间。
        self.user_encoder = UserEncoder(user_input_dim, hidden_dim,
                                        embedding_type=user_embedding_type).to(device)
        self.server_encoder = ServerEncoder(server_input_dim, hidden_dim,
                                            embedding_type=server_embedding_type).to(device)
        # 建一个线性层，用于将用户编码和服务器编码进行融合。
        # 输入维度是隐藏维度的两倍（因为用户编码和服务器编码都是隐藏维度），
        # 输出维度是隐藏维度。这里的 device=device 参数用于指定线性层所在的设备。
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim, device=device)
        # 创建一个线性层，用于将融合后的编码映射到单个值（评分或者价值）。
        # 输入维度是隐藏维度，输出维度是1，因为这是一个回归任务。
        # 同样使用了 device=device 参数指定线性层所在的设备。
        self.out = nn.Linear(hidden_dim, 1, device=device)

    def forward(self, user_input_seq, server_input_seq):
        # 分别将用户输入序列和服务器输入序列通过对应的编码器进行编码，得到用户编码和服务器编码。
        user_code = self.user_encoder(user_input_seq)
        user_all = torch.mean(user_code, dim=1)
        # 对编码后的用户和服务器特征进行平均池化，得到用户和服务器的整体特征表示。
        server_code = self.server_encoder(server_input_seq)
        server_all = torch.mean(server_code, dim=1)
        # 将用户和服务器的整体特征连接起来，通过融合层和输出层进行计算，
        # 得到最终的输出。最后使用 squeeze(-1) 将输出的最后一个维度压缩，以获得一维的输出。
        return self.out(self.fusion(torch.cat([user_all, server_all], dim=-1))).squeeze(-1)
