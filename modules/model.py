from torch_scatter import scatter_mean
import numpy as np
import torchsparsegradutils
import utils
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import structured_negative_sampling
import scipy.sparse as sp


class Attention(nn.Module):
    def __init__(self,args):
        super(Attention, self).__init__()
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        self.lambda0 = nn.Parameter(torch.zeros(1)) #定义一个可训练的参数lambda0，用于调整模型的学习
        # self.path_emb = nn.Embedding(2**(args.sample_hop+1)-2, 1) #创建一个路径嵌入层，嵌入大小根据args.sample_hop动态计算
        # nn.init.zeros_(self.path_emb.weight) #初始化路径嵌入的权重为零
        self.sqrt_dim = 1./torch.sqrt(torch.tensor(args.hidden_dim))
        self.sqrt_eig = 1./torch.sqrt(torch.tensor(args.eigs_dim))#计算隐藏维度和特征维度的平方根的倒数，用于后续的缩放
        self.my_parameters = [
            {'params': self.lambda0, 'weight_decay': 1e-2},
        ]

    def sum_norm(self, indices, values, n):
        indices = indices.to(self.device)
        values = values.to(self.device)
        s = torch.zeros(n, device=self.device).scatter_add(0, indices[0], values)
        s[s == 0.] = 1.
        return values / s[indices[0]]

    def sparse_softmax(self, indices, values, n):
        return self.sum_norm(indices, torch.clamp(torch.exp(values), min=-5, max=5), n)

    def forward(self, q, k, v,  indices, eigs):
        ni, nx, ny, nz = [], [], [], []
        indices = indices.long()
        # indices = torch.tensor(indices).long()

        # for u, v in zip(indices[0], indices[1]):
        #     x = torch.mul(q[u], k[v]).sum(dim=-1)*self.sqrt_dim #计算两个节点嵌入 q[i[0]] 和 k[i[1]] 的点积，表示它们的相似度，然后乘以缩放系数
        #     nx.append(x)
        #     y = torch.mul(eigs[u], eigs[v]).sum(dim=-1)
        #     ny.append(y)
        #
        #     ni.append([u, v])

        # 直接批量计算点积，不需要循环
        q_selected = q[indices[0]]  # 选取 q 中的 u 索引
        k_selected = k[indices[1]]  # 选取 k 中的 v 索引
        # 计算点积并乘以缩放系数
        x = torch.sum(q_selected * k_selected, dim=-1) * self.sqrt_dim  # 点积并乘以缩放系数
        nx.append(x)
        eigs_selected_u = eigs[indices[0]]  # 选取 eigs 中的 u 索引
        eigs_selected_v = eigs[indices[1]]  # 选取 eigs 中的 v 索引
        y = torch.sum(eigs_selected_u * eigs_selected_v, dim=-1)  # 计算 eigs 的点积
        ny.append(y)

        ni.append(torch.stack([indices[0], indices[1]], dim=0))
        i = torch.concat(ni, dim=-1)

        # 构建 s 列表，首先处理 nx 和 ny
        s = []
        s.append(torch.cat(nx, dim=-1).to(self.device))
        s[0] = s[0] + torch.exp(self.lambda0) * torch.cat(ny, dim=-1).to(self.device)
        # 计算 sparse_softmax
        s = [self.sparse_softmax(i, _, q.shape[0]) for _ in s]
        # 计算平均值
        s = torch.stack(s, dim=1).mean(dim=1)

        sparse_tensor = torch.sparse_coo_tensor(i, s, torch.Size([q.shape[0], k.shape[0]])).to(self.device)
        return torchsparsegradutils.sparse_mm(sparse_tensor, v)



'''这个Encoder类实现了一个基本的自注意力机制'''
class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.self_attention = Attention(args)
        self.my_parameters = self.self_attention.my_parameters
        self.hidden_dim = args.hidden_dim
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    def forward(self, x, indices, eigs):
        y = F.layer_norm(x, normalized_shape=(self.hidden_dim,)).to(self.device)#输入x进行层归一化
        y = self.self_attention(
            y, y, y,
            indices,
            eigs)
        return y


class Aggregator(nn.Module):
    """
    Local Weighted Smoothing aggregation scheme
    """

    def __init__(self, args, n_users, n_virtual, n_iter):
        super(Aggregator, self).__init__()

        self.n_users = n_users
        self.n_virtual = n_virtual
        self.n_iter = n_iter
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        self.args = args
        if n_virtual == 3:
            self.w = torch.nn.Parameter(torch.FloatTensor([0.4, 0.3, 0.3]), requires_grad=True).to(
                self.device)  # 定义了一个可训练的参数向量 self.w，并将其初始化为一个包含特定值的张量
        elif n_virtual == 2:
            self.w = torch.nn.Parameter(torch.FloatTensor([0.5, 0.5]), requires_grad=True).to(self.device)
        elif n_virtual == 1:
            self.w = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True).to(self.device)
        elif n_virtual == 4:
            self.w = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]), requires_grad=True).to(self.device)
        elif n_virtual == 5:
            self.w = torch.nn.Parameter(torch.FloatTensor([0.2, 0.2, 0.2, 0.2, 0.2]), requires_grad=True).to(self.device)
        elif n_virtual == 6:
            self.w = torch.nn.Parameter(torch.FloatTensor([0.16, 0.16, 0.16, 0.16, 0.16, 0.2]), requires_grad=True).to(self.device)
        elif n_virtual == 7:
            self.w = torch.nn.Parameter(torch.FloatTensor([0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16]), requires_grad=True).to(self.device)

    def forward(self, entity_emb, user_emb,
                edge_index, edge_type, adj_mat, ua_adj_mat):
        # device = torch.device("cuda:3")
        n_entities = entity_emb.shape[0]
        n_users = self.n_users

        edge_type_uni = torch.unique(edge_type)
        entity_emb_list = []

        user_index, item_index = adj_mat.nonzero()
        user_index1, a_index = ua_adj_mat.nonzero()
        user_index = torch.tensor(user_index).type(torch.long).to(self.device)
        item_index = torch.tensor(item_index).type(torch.long).to(self.device)
        user_index1 = torch.tensor(user_index1).type(torch.long).to(self.device)
        a_index = torch.tensor(a_index).type(torch.long).to(self.device)


        for i in edge_type_uni:
            index = torch.where(edge_type == i)
            index = index[0]
            head, tail = edge_index
            head = head[index]
            tail = tail[index]
            u = None
            neigh_emb = entity_emb[tail]
            for clus_iter in range(self.n_iter):
                if u is None:
                    u = scatter_mean(src=neigh_emb, index=head, dim_size=n_entities, dim=0)
                else:
                    center_emb = u[head]
                    sim = torch.sum(center_emb * torch.tanh(neigh_emb),
                                    dim=1)
                    n, d = neigh_emb.size()
                    sim = torch.unsqueeze(sim, dim=1)
                    sim.expand(n, d)
                    neigh_emb = sim * neigh_emb
                    u = scatter_mean(src=neigh_emb, index=head, dim_size=n_entities, dim=0)

                if clus_iter < self.n_iter - 1:
                    squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                    u = squash.unsqueeze(1) * F.normalize(u, dim=1)
                u += entity_emb
            entity_emb_list.append(u)
        entity_emb_list = torch.stack(entity_emb_list, dim=0)
        if self.n_virtual == 3:
            item_0 = entity_emb_list[0]
            item_1 = entity_emb_list[1]
            item_2 = entity_emb_list[2]
            w0 = self.w[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
            w1 = self.w[1].unsqueeze(dim=-1).unsqueeze(dim=-1)
            w2 = self.w[2].unsqueeze(dim=-1).unsqueeze(dim=-1)
            w_0 = torch.exp(w0) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
            w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
            w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
            entity_agg = w_0.mul(item_0) + w_1.mul(item_1) + w_2.mul(item_2)
        if self.n_virtual == 2:
            item_0 = entity_emb_list[0]
            item_1 = entity_emb_list[1]

            w0 = self.w[0].unsqueeze(dim=-1)
            w1 = self.w[1].unsqueeze(dim=-1)
            w_0 = torch.exp(w0) / (torch.exp(w0) + torch.exp(w1))
            w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1))
            # w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
            entity_agg = w_0.mul(item_0) + w_1.mul(item_1)
        if self.n_virtual == 1:
            item_0 = entity_emb_list[0]
            w0 = self.w[0]
            w_0 = torch.exp(w0) / (torch.exp(w0) )
            entity_agg = w_0.mul(item_0)
        if self.n_virtual == 4:
            item_0 = entity_emb_list[0]
            item_1 = entity_emb_list[1]
            item_2 = entity_emb_list[2]
            item_3 = entity_emb_list[3]

            w0 = self.w[0].unsqueeze(dim=-1)
            w1 = self.w[1].unsqueeze(dim=-1)
            w2 = self.w[2].unsqueeze(dim=-1)
            w3 = self.w[3].unsqueeze(dim=-1)
            w_0 = torch.exp(w0) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3))
            w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3))
            w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3))
            w_3 = torch.exp(w3) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3))
            entity_agg = w_0.mul(item_0) + w_1.mul(item_1) + w_2.mul(item_2) +w_3.mul(item_3)
        if self.n_virtual == 5:
            item_0 = entity_emb_list[0]
            item_1 = entity_emb_list[1]
            item_2 = entity_emb_list[2]
            item_3 = entity_emb_list[3]
            item_4 = entity_emb_list[4]

            w0 = self.w[0].unsqueeze(dim=-1)
            w1 = self.w[1].unsqueeze(dim=-1)
            w2 = self.w[2].unsqueeze(dim=-1)
            w3 = self.w[3].unsqueeze(dim=-1)
            w4 = self.w[4].unsqueeze(dim=-1)
            w_0 = torch.exp(w0) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4))
            w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4))
            w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4))
            w_3 = torch.exp(w3) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4))
            w_4 = torch.exp(w4) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4))
            entity_agg = w_0.mul(item_0) + w_1.mul(item_1) + w_2.mul(item_2) +w_3.mul(item_3) +w_4.mul(item_4)
        if self.n_virtual == 6:
            item_0 = entity_emb_list[0]
            item_1 = entity_emb_list[1]
            item_2 = entity_emb_list[2]
            item_3 = entity_emb_list[3]
            item_4 = entity_emb_list[4]
            item_5 = entity_emb_list[5]

            w0 = self.w[0].unsqueeze(dim=-1)
            w1 = self.w[1].unsqueeze(dim=-1)
            w2 = self.w[2].unsqueeze(dim=-1)
            w3 = self.w[3].unsqueeze(dim=-1)
            w4 = self.w[4].unsqueeze(dim=-1)
            w5 = self.w[5].unsqueeze(dim=-1)
            w_0 = torch.exp(w0) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5))
            w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5))
            w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5))
            w_3 = torch.exp(w3) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5))
            w_4 = torch.exp(w4) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5))
            w_5 = torch.exp(w5) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5))
            entity_agg = w_0.mul(item_0) + w_1.mul(item_1) + w_2.mul(item_2) +w_3.mul(item_3) +w_4.mul(item_4) + w_5.mul(item_5)
        if self.n_virtual == 7:
            item_0 = entity_emb_list[0]
            item_1 = entity_emb_list[1]
            item_2 = entity_emb_list[2]
            item_3 = entity_emb_list[3]
            item_4 = entity_emb_list[4]
            item_5 = entity_emb_list[5]
            item_6 = entity_emb_list[6]

            w0 = self.w[0].unsqueeze(dim=-1)
            w1 = self.w[1].unsqueeze(dim=-1)
            w2 = self.w[2].unsqueeze(dim=-1)
            w3 = self.w[3].unsqueeze(dim=-1)
            w4 = self.w[4].unsqueeze(dim=-1)
            w5 = self.w[5].unsqueeze(dim=-1)
            w6 = self.w[6].unsqueeze(dim=-1)
            w_0 = torch.exp(w0) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5) + torch.exp(w6))
            w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5) + torch.exp(w6))
            w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5) + torch.exp(w6))
            w_3 = torch.exp(w3) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5) + torch.exp(w6))
            w_4 = torch.exp(w4) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5) + torch.exp(w6))
            w_5 = torch.exp(w5) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5) + torch.exp(w6))
            w_6 = torch.exp(w6) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2) + torch.exp(w3) + torch.exp(w4) + torch.exp(w5) + torch.exp(w6))
            entity_agg = w_0.mul(item_0) + w_1.mul(item_1) + w_2.mul(item_2) +w_3.mul(item_3) +w_4.mul(item_4) + w_5.mul(item_5) + w_6.mul(item_6)
        """LWS for user representation learning"""
        u = None
        ua = None
        for clus_iter in range(self.n_iter):
            neigh_emb = entity_emb[item_index-n_users]
            if u is None:
                u = scatter_mean(src=neigh_emb, index=user_index, dim_size=n_users, dim=0)
            else:
                center_emb = u[user_index]
                sim = torch.sum(center_emb * neigh_emb, dim=1)
                n, d = neigh_emb.size()
                sim = torch.unsqueeze(sim, dim=1)
                sim.expand(n, d)
                neigh_emb = sim * neigh_emb
                u = scatter_mean(src=neigh_emb, index=user_index, dim_size=n_users, dim=0)

            if clus_iter < self.n_iter - 1:
                squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                u = squash.unsqueeze(1) * F.normalize(u, dim=1)
            u += user_emb
        for clus_iter in range(self.n_iter):
            neigh_emb = entity_emb[a_index]
            if ua is None:
                ua = scatter_mean(src=neigh_emb, index=user_index1, dim_size=n_users, dim=0)
            else:
                center_emb = ua[user_index1]
                sim = torch.sum(center_emb * neigh_emb, dim=1)
                n, d = neigh_emb.size()
                sim = torch.unsqueeze(sim, dim=1)
                sim.expand(n, d)
                neigh_emb = sim * neigh_emb
                ua = scatter_mean(src=neigh_emb, index=user_index1, dim_size=n_users, dim=0)

            if clus_iter < self.n_iter - 1:
                squash = torch.norm(ua, dim=1) ** 2 / (torch.norm(ua, dim=1) ** 2 + 1)
                ua = squash.unsqueeze(1) * F.normalize(ua, dim=1)


        user_agg = u + ua

        return entity_agg, user_agg

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self,args, channel, n_hops, n_iter, n_users,
                 n_virtual, n_relations, adj_mat, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.adj_mat = adj_mat

        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_virtual = n_virtual
        self.node_dropout_rate = node_dropout_rate
        self.ua_node_dropout_rate = args.ua_node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.chanel = channel
        self.temperature = 0.2
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")


        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel,device=self.device))
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]


        for i in range(n_hops):
            self.convs.append(Aggregator(args,n_users=n_users, n_virtual=n_virtual, n_iter=n_iter))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):

        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _matrix_sampling(self, adj_mat, rate=0.5):

        coo = adj_mat.tocoo()
        n_edges = coo.nnz

        sample_size = int(n_edges * rate)
        random_indices = np.random.choice(n_edges, size=sample_size, replace=False)

        sampled_row = coo.row[random_indices]
        sampled_col = coo.col[random_indices]
        sampled_data = coo.data[random_indices]
        sampled_adj_mat = sp.coo_matrix((sampled_data, (sampled_row, sampled_col)), shape=adj_mat.shape)

        return sampled_adj_mat



    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                adj_mat, interact_mat,ua_adj_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            ua_adj_mat = self._matrix_sampling(ua_adj_mat, self.ua_node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = 0
        weight = self.weight
        relation_ = torch.mm(weight, latent_emb.t())
        relation_remap = torch.argmax(relation_, dim=1)
        edge_type = relation_remap[edge_type - 1]

        for i in range(len(self.convs)):
            entity_emb,user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type, adj_mat, ua_adj_mat)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)


        return entity_res_emb, user_res_emb, cor
class SRVSKG(nn.Module):
    def __init__(self,data_config, args, indices, L_eigs, graph, adj_mat, ua_adj_mat):
        super(SRVSKG, self).__init__()
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.args = args
        self.decay = args.l2
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.emb_size = args.dim
        self.context_hops = args.context_hops
        self.n_iter = args.n_iter
        self.n_virtual = args.n_virtual
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.ind = args.ind
        self.adj_mat = adj_mat[0]
        self.ua_adj_mat = ua_adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self.hidden_dim = args.hidden_dim
        self.n_layers_sig = args.n_layers_sig
        self.embedding_user = nn.Embedding(self.n_users, self.hidden_dim).to(self.device)
        self.embedding_item = nn.Embedding(self.n_entities, self.hidden_dim).to(self.device)

        self.latent_emb = nn.Embedding(self.n_virtual, self.hidden_dim).to(self.device)
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.hidden_dim)).to(self.device)
        self.all_embed = nn.Parameter(self.all_embed)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.latent_emb.weight,std=0.1)
        self.my_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
        ]

        self.layers = []
        for i in range(args.n_layers_sig):
            layer = Encoder(args).to(self.device)
            self.layers.append(layer)
            self.my_parameters.extend(layer.my_parameters)
        self._users, self._items = None, None
        self.optimizer = torch.optim.Adam(self.my_parameters,lr=args.lr)
        self.indices = indices
        self.indices = self.indices.to(self.device)
        self.L_eigs = L_eigs
        self.L_eigs = self.L_eigs.to(self.device)

        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.gcn = self._init_model()
        self.my_parameters_my = [
            {'params': self.all_embed},
            {'params': self.latent_emb.parameters()},
        ]
        self.my_parameters_my.append({'params': self.gcn.weight})
        for conv in self.gcn.convs:
            self.my_parameters_my.append({'params': conv.parameters()})

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _init_model(self):
        return GraphConv(self.args, channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_iter=self.n_iter,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_virtual=self.n_virtual,
                         adj_mat=self.adj_mat,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def computer_my_model(self, cf_batch):
        user = cf_batch['users']
        pos_item = cf_batch['pos_items']
        neg_item = cf_batch['neg_items']
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb.weight,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.adj_mat,
                                                     self.interact_mat,
                                                     self.ua_adj_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout, )

        user_gcn_emb = torch.add(user_gcn_emb, self.embedding_user.weight)
        entity_gcn_emb = torch.add(entity_gcn_emb, self.embedding_item.weight)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss_my(u_e, pos_e, neg_e)

    def create_bpr_loss_my(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss



    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for i in range(self.n_layers_sig):
            indices = self.indices
            all_emb = self.layers[i](all_emb,
                                     indices,
                                     self.L_eigs)
            embs.append(all_emb)
        embs = [emb.to(self.device) for emb in embs]
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        self._users, self._items = torch.split(light_out, [self.n_users, light_out.size(0) - self.n_users])

    def train_sign(self,train_cf,train_cf_neg,args):
        self.train()



        pos_u = torch.tensor(train_cf[:, 0], device=self.device)
        pos_i = torch.tensor(train_cf[:, 1], device=self.device)
        neg_u = torch.tensor(train_cf_neg[:, 0], device=self.device)
        neg_i = torch.tensor(train_cf_neg[:, 1], device=self.device)


        num_pos_samples = pos_u.shape[0]
        indices = torch.randperm(neg_u.shape[0])[:num_pos_samples]
        neg_u = neg_u[indices]
        neg_i = neg_i[indices]

        total_loss = 0
        s = 0


        while s < num_pos_samples:
            end = min(s + args.batch_size, num_pos_samples)


            batch_pos_u = pos_u[s:end]
            batch_pos_i = pos_i[s:end]


            batch_neg_u = neg_u[s:end]
            batch_neg_i = neg_i[s:end]


            all_j = structured_negative_sampling(
                torch.concat([torch.stack([batch_pos_u, batch_pos_i]),
                              torch.stack([batch_neg_u, batch_neg_i])], dim=1),
                num_nodes=self.n_items)[2]

            pos_j, neg_j = torch.split(all_j, [len(batch_pos_u), len(batch_neg_u)])


            batch_loss = self.loss_one_batch(batch_pos_u, batch_pos_i, pos_j, batch_neg_u, batch_neg_i, neg_j, args)
            total_loss += batch_loss.item()


            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()


            s += args.batch_size


        return total_loss

    def loss_one_batch(self, pos_u, pos_i, pos_j, neg_u, neg_i, neg_j,args):

        my_user_emb = self.all_embed[:self.n_users, :]
        my_item_emb = self.all_embed[self.n_users:self.n_users + self.n_items, :]


        self.computer()
        all_user, all_item = self._users.to(self.device), self._items.to(self.device)
        pos_u, pos_i, pos_j, neg_u, neg_i, neg_j = pos_u.long().to(self.device), pos_i.long().to(self.device), pos_j.long().to(self.device), neg_u.long().to(self.device), neg_i.long().to(self.device), neg_j.long().to(self.device)


        pos_u_emb0, pos_u_emb = self.embedding_user(pos_u), all_user[pos_u]
        pos_i_emb0, pos_i_emb = self.embedding_item(pos_i), all_item[pos_i]
        pos_j_emb0, pos_j_emb = self.embedding_item(pos_j), all_item[pos_j]
        neg_u_emb0, neg_u_emb = self.embedding_user(neg_u), all_user[neg_u]
        neg_i_emb0, neg_i_emb = self.embedding_item(neg_i), all_item[neg_i]
        neg_j_emb0, neg_j_emb = self.embedding_item(neg_j), all_item[neg_j]

        my_pos_u_emb = my_user_emb[pos_u]
        my_pos_i_emb = my_item_emb[pos_i]
        my_pos_j_emb = my_item_emb[pos_j]
        my_neg_u_emb = my_user_emb[neg_u]
        my_neg_i_emb = my_item_emb[neg_i]
        my_neg_j_emb = my_item_emb[neg_j]



        pos_scores_ui = torch.sum(pos_u_emb * pos_i_emb, dim=-1)
        pos_scores_uj = torch.sum(pos_u_emb * pos_j_emb, dim=-1)
        neg_scores_ui = torch.sum(neg_u_emb * neg_i_emb, dim=-1)
        neg_scores_uj = torch.sum(neg_u_emb * neg_j_emb, dim=-1)


        if args.beta_sign == 0:
            reg_loss = (1 / 2) * (pos_u_emb0.norm(2).pow(2) +
                                  pos_i_emb0.norm(2).pow(2) +
                                  pos_j_emb0.norm(2).pow(2)) / float(pos_u.shape[0])
            scores = pos_scores_uj - pos_scores_ui
        else:
            reg_loss = (1 / 2) * (pos_u_emb0.norm(2).pow(2) +
                                  pos_i_emb0.norm(2).pow(2) +
                                  pos_j_emb0.norm(2).pow(2) +
                                  neg_u_emb0.norm(2).pow(2) +
                                  neg_i_emb0.norm(2).pow(2) +
                                  neg_j_emb0.norm(2).pow(2)) / float(pos_u.shape[0] + neg_u.shape[0])
            scores = torch.concat([pos_scores_uj - pos_scores_ui, args.beta_sign * (neg_scores_uj - neg_scores_ui)], dim=0)


        loss = torch.mean(F.softplus(scores))

        return loss + args.lambda_reg * reg_loss



    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def generate(self):
        if self._users is None:
            self.computer()
        user_emb, item_emb = self._users, self._items




        my_user_emb = self.all_embed[:self.n_users, :]
        my_item_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb, cor = self.gcn(my_user_emb,
                                                     my_item_emb,
                                                     self.latent_emb.weight,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.adj_mat,
                                                     self.interact_mat,
                                                     self.ua_adj_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout, )
        if self.args.havesig=="True":
            user_gcn_emb = (user_emb + user_gcn_emb).view((-1, self.hidden_dim))
            entity_gcn_emb = (item_emb + entity_gcn_emb).view((-1, self.hidden_dim))

        return  entity_gcn_emb[:self.n_items, :],user_gcn_emb



    def generate_sig(self):
        if self._users is None:
            self.computer()
        user_emb, item_emb = self._users, self._items


        return  item_emb[:self.n_items, :],user_emb




