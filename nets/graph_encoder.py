import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
    # 定义了前向传播方法，用于执行跳跃连接操作。它接受一个输入张量 input，
    # 并将其与子模块处理后的结果相加作为输出。
    def forward(self, input):
        # 将输入张量 input 和经过子模块 module 处理后的结果相加，实现了跳跃连接的功能。
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    # 头数 n_heads、输入维度 input_dim、嵌入维度 embed_dim、值的维度 val_dim 和键的维度 key_dim。
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()
        # 果未提供值和键的维度，则根据给定的嵌入维度和头数计算默认值。
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        # 计算归一化因子，用于注意力分数的归一化。
        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need
        # 定义了查询、键、值和输出的线性变换矩阵，并将其包装为可学习的参数。
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        # 遍历所有参数并将其初始化为服从均匀分布的随机数。
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        # 如果未提供数据 h，则默认计算自注意力。
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        # 获取输入数据的形状信息，并确保输入数据的维度正确。
        # 获取输入张量 h 的维度信息，其中 batch_size 是批量大小，graph_size 是图的大小，input_dim 是输入的特征维度。
        batch_size, graph_size, input_dim = h.size()
        # 获取查询张量 q 的第二维度大小，即查询的数量。
        n_query = q.size(1)
        # 保查询张量的第三维度（特征维度）与输入张量 h 的特征维度相匹配。
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"
        # 将输入张量 h 和查询张量 q 展平为二维张量，以便进行线性变换。
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        # 构建查询、键、值的形状信息，其中 -1 表示自动计算该维度的大小。
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # 分别对查询张量、键张量和值张量执行线性变换，并根据之前定义的形状信息进行形状重塑，得到查询、键、值张量。
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        # 计算注意力分数，首先将查询张量与键张量的转置相乘，然后乘以归一化因子。
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention

        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf
        # 对注意力分数进行 softmax 操作，得到注意力权重。
        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        # （可选）修正可能导致 softmax 函数返回 nan 的部分，将其设为 0。
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        # 将注意力权重乘以值张量，得到多头注意力。
        heads = torch.matmul(attn, V)
        # 执行输出线性变换，将多头注意力转换为最终输出。
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class Normalization(nn.Module):
    # 初始化方法，接受两个参数：embed_dim 表示嵌入维度，normalization 表示归一化类型，默认为批归一化。
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()
        # 根据给定的 normalization 类型选择对应的 PyTorch 归一化层类别。
        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)
        # 据选择的归一化类别创建对应的归一化层，并设置 affine=True 表示需要学习归一化的可学习参数。
        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):
        # 初始化参数方法，用于初始化归一化层的参数。
        # 遍历所有参数。
        for name, param in self.named_parameters():
            # 计算初始化参数的标准差。
            stdv = 1. / math.sqrt(param.size(-1))
            # 将参数初始化为服从均匀分布的随机数。
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # 如果选择的归一化层是批归一化，则执行下面的语句块。
        if isinstance(self.normalizer, nn.BatchNorm1d):
            # 先将输入数据进行形状变换，然后应用批归一化，最后恢复原始形状。
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        # 如果选择的归一化层是实例归一化，则执行下面的语句块。
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            # 先将输入数据的维度换位，然后应用实例归一化，最后再换回原始维度。
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # 如果未知的归一化层类型，则执行下面的语句块。
            assert self.normalizer is None, "Unknown normalizer type"
            # 抛出异常，表示未知的归一化层类型。
            return input


class MultiHeadAttentionLayer(nn.Sequential):
    # n_heads：注意力头的数量，用于并行计算注意力。
    # embed_dim：输入和输出的嵌入维度。
    # feed_forward_hidden：前馈网络隐藏层的大小。
    # normalization：用于注意力计算的归一化方法，通常是 "batch" 或 "layer"。
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            # 使用跳跃连接将多头注意力层包装起来，即将多头注意力层作为子模块传入 SkipConnection 中。
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            # 使用归一化层对多头注意力输出进行归一化，其中 embed_dim 是嵌入维度，normalization 是归一化方法。
            Normalization(embed_dim, normalization),
            # 用跳跃连接将前馈网络包装起来，即将前馈网络作为子模块传入 SkipConnection 中。
            SkipConnection(
                # 定义了一个序列模块，包含了一个线性层、ReLU 激活函数以及一个线性层，用于构建前馈网络。
                # 如果 feed_forward_hidden 大于 0，则创建具有隐藏层的前馈网络；否则创建一个仅包含线性层的前馈网络。
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            # 使用归一化层对前馈网络输出进行归一化，
            # 其中 embed_dim 是嵌入维度，normalization 是归一化方法。
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    # n_heads：注意力头的数量，用于并行计算注意力。
    # embed_dim：嵌入维度，用于表示节点的嵌入向量的大小。
    # n_layers：堆叠的注意力层的数量。
    # node_dim：节点特征的维度。如果为 None，则不执行输入到嵌入空间的映射。
    # normalization：用于注意力计算的归一化方法，通常是 "batch" 或 "layer"。
    # feed_forward_hidden：注意力层中前馈网络隐藏层的大小。
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()
        # 初始化节点特征到嵌入空间的映射。
        # 如果 node_dim 不为 None，则创建一个线性层将输入特征映射到嵌入空间；否则将 init_embed 设置为 None。
        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        # 创建一个堆叠的注意力层。使用 nn.Sequential 将多个 MultiHeadAttentionLayer 实例堆叠在一起，共堆叠 n_layers 次。
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        # mask：（尚未支持）用于指示要应用的注意力掩码。
        assert mask is None, "TODO mask not yet supported!"
        # Batch multiply to get initial embeddings of nodes
        # 如果存在节点特征到嵌入空间的映射，则将输入节点特征 x 映射到嵌入空间；否则保持不变。
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        # 通过堆叠的注意力层处理嵌入的节点特征。
        h = self.layers(h)

        return h    # (batch_size, graph_size, embed_dim)
