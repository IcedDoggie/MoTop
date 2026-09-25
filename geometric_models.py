import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# maybe i need an image encoder
from torch_geometric_temporal.nn.attention.stgcn import TemporalConv


def standardization(values):
    x_mean = values.mean(dim=0, keepdim=True)  # Mean across all nodes
    x_std = values.std(dim=0, keepdim=True)  # Std across all nodes
    x_std = torch.clamp(x_std, min=1e-8)  # Prevent division by zero
    standardized_values = (values - x_mean) / x_std
    return standardized_values


class UnitGCNTCN(torch.nn.Module):
    def __init__(self, input_features, output_features, temporal_kernel, edge_size):
        super(UnitGCNTCN, self).__init__()
        self.edge_weights = torch.nn.Parameter(torch.ones(edge_size))
        self.conv = GCNConv(input_features, output_features)

        self.tcn = TemporalConv(
            output_features, output_features, kernel_size=temporal_kernel
        )
        self.residual = TemporalConv(
            input_features, output_features, kernel_size=temporal_kernel
        )
        # Add a flag to use external edge weights
        self.use_external_weights = False
        self.external_weights = None

    def set_edge_weights(self, edge_weights):
        """Method to set external edge weights"""
        self.external_weights = edge_weights
        self.use_external_weights = True

    def forward(
        self,
        x,
        edge_index,
        texture_context=None,
        cluster_indices=None,
        batch_indices=None,
    ):
        B, T, N, feat = x.shape
        temporal_gcn_feat = []
        # batch_indices = batch_indices.repeat(47, 1).T

        # Choose which edge weights to use
        if self.use_external_weights and self.external_weights is not None:
            current_edge_weights = (
                1.0 * self.external_weights.sigmoid()
                + 0.0 * self.edge_weights.sigmoid()
            )
        else:
            # current_edge_weights = self.edge_weights.sigmoid()
            current_edge_weights = nn.Softmax(dim=0)(self.edge_weights)

        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, F)
            x_t = x_t.reshape(B * N, feat)  # (B*N, F) # CHECK HERE 6/11

            x_t = self.conv(
                x=x_t, edge_index=edge_index, edge_weight=current_edge_weights
            ).relu()  # CHECK HERE 6/11

            temporal_gcn_feat.append(x_t)

        x_t = torch.stack(temporal_gcn_feat, dim=1)  # (B*N, T, F)
        # (batch_size, input_time_steps, num_nodes, in_channels).

        x_t = x_t.reshape((B, T, N, x_t.shape[-1]))
        x_t = self.tcn(x_t)
        x_residual = self.residual(x)
        x_t = x_t + x_residual
        x_t = F.relu(x_t)

        # pass on the edge weight
        edge_weight = self.edge_weights
        return x_t
        # return x_t, edge_weight


class UnitGCN(torch.nn.Module):
    def __init__(self, input_features, output_features, edge_size):
        super(UnitGCN, self).__init__()
        self.edge_weights = torch.nn.Parameter(torch.ones(edge_size))
        self.conv = GCNConv(input_features, output_features)

        # Add a flag to use external edge weights
        self.use_external_weights = False
        self.external_weights = None

    def set_edge_weights(self, edge_weights):
        """Method to set external edge weights"""
        self.external_weights = edge_weights
        self.use_external_weights = True

    def forward(self, x, edge_index):
        B_N, feat = x.shape

        # Choose which edge weights to use
        if self.use_external_weights and self.external_weights is not None:
            current_edge_weights = (
                1.0 * self.external_weights.sigmoid()
                + 0.0 * self.edge_weights.sigmoid()
            )
        else:
            current_edge_weights = nn.Softmax(dim=0)(self.edge_weights)

        x_t = self.conv(
            x=x, edge_index=edge_index, edge_weight=current_edge_weights
        ).relu()  # CHECK HERE 6/11

        return x_t


class Multi_GCN(torch.nn.Module):
    def __init__(self, input_features, output_features, edge_size):
        super(Multi_GCN, self).__init__()
        self.gcn1 = UnitGCN(input_features, output_features, edge_size=edge_size)
        # self.gcn2 = UnitGCN(
        #     output_features, output_features*2, edge_size=edge_size
        # )

    def forward(self, x, edge_index):
        x1 = self.gcn1(x, edge_index)
        # x2 = self.gcn2(x1, edge_index)
        return x1


class MultiGCN_OneTCN(torch.nn.Module):
    def __init__(self, input_features, num_classes, edge_size):
        super(MultiGCN_OneTCN, self).__init__()
        self.gcn1 = Multi_GCN(input_features, output_features=32, edge_size=edge_size)
        self.gcn2 = Multi_GCN(input_features, output_features=32, edge_size=edge_size)
        self.gcn3 = Multi_GCN(input_features, output_features=32, edge_size=edge_size)

        output_features = 32

        self.tcn = TemporalConv(output_features, num_classes, kernel_size=3)
        self.residual = TemporalConv(input_features, num_classes, kernel_size=3)

        self.edge_weight_for_tcn = torch.ones(edge_size).cuda()
        # self.pool = nn.AdaptiveAvgPool1d(12)

    def forward(self, x, edge_index, motion_context=None):
        B, T, N, feat = x.shape
        edge_index = edge_index[0, :, :]  # use this for casme2 datasets

        temporal_gcn_feat = []

        if motion_context is not None:
            if type(motion_context) == list:
                motion_context, texture_context = motion_context
                motion_context = motion_context.unsqueeze(1).repeat(1, T, 1, 1)
                (
                    texture_B,
                    texture_feat,
                    texture_T,
                    texture_x,
                    texture_y,
                ) = texture_context.shape
                texture_context = texture_context.view(
                    ((texture_B * texture_T), texture_feat, (texture_x * texture_y))
                )
                texture_context = self.pool_texture_for_graph(
                    texture_context
                )  # (B*T, feat, target_tokens)
                texture_context = texture_context.view(
                    (texture_B, texture_T, -1, texture_feat)
                )  # (B, T, feat, target_tokens)

                motion_context = torch.concat([motion_context, texture_context], dim=3)

            else:
                motion_context = motion_context.unsqueeze(1).repeat(1, T, 1, 1)
            x = torch.concat([x, motion_context], dim=3)
        B, T, N, feat = x.shape
        x_t1 = x[:, 0, :, :]
        x_t1 = x_t1.reshape(B * N, feat)
        x_t2 = x[:, 1, :, :]
        x_t2 = x_t2.reshape(B * N, feat)
        x_t3 = x[:, 2, :, :]
        x_t3 = x_t3.reshape(B * N, feat)

        x_t1 = self.gcn1(x_t1, edge_index)
        x_t2 = self.gcn2(x_t2, edge_index)
        x_t3 = self.gcn3(x_t3, edge_index)

        x_t = torch.stack([x_t1, x_t2, x_t3], dim=1)  # (B, T, F)
        x_t = x_t.reshape((B, T, N, x_t.shape[-1]))
        x_t = F.relu(x_t)
        x_tcn = self.tcn(x_t)

        x_residual = self.residual(x)
        x_tcn = x_tcn + x_residual
        x_tcn = F.relu(x_tcn)
        x_tcn = x_tcn.mean(1)

        if motion_context is not None:
            motion_temporal_context = self.pool(motion_context[:, 0, :, :])
            x_tcn = x_tcn * motion_temporal_context
        # top_k = 10
        # x_tcn = torch.topk(x_tcn, top_k, dim=1)[0]  # (B, top_k)
        x_tcn = x_tcn.mean(1)
        a = 1
        # for t in range(T):
        #     x_t = x[:, t, :, :]  # (B, N, F)
        #     x_t = x_t.reshape(B * N, feat)  # (B*N, F) # CHECK HERE 6/11
        #     x_t = self.gcn1(x_t, edge_index)

        #     temporal_gcn_feat.append(x_t)
        # x_t = torch.stack(temporal_gcn_feat, dim=1)  # (B*N, T, F)
        # # (batch_size, input_time_steps, num_nodes, in_channels).
        # x_t = x_t.reshape((B, T, N, x_t.shape[-1]))
        # x_t = self.tcn(x_t)
        # x_residual = self.residual(x)
        # x_t = x_t + x_residual
        # x_t = F.relu(x_t)
        return x_tcn, a


class GCN_TCN(torch.nn.Module):
    def __init__(self, node_features=2, num_classes=5):
        super(GCN_TCN, self).__init__()

        self.landmark_count = 47

        # edge_size = 315 # hypergraph
        # edge_size = 14 # 14 edges
        # edge_size = 525
        edge_size = 50  # landmarks 47
        # edge_size = 124 # landmarks 3d 105 mp
        # edge_size = 100
        # edge_size = 150
        # edge_size = 169
        # edge_size = 29584
        # edge_size = 9000
        # edge_size = 9604 # for MP landmarks
        # edge_size = 2116 # for FAN landmarks

        self.gcntcn1 = UnitGCNTCN(
            node_features, 32, temporal_kernel=1, edge_size=edge_size
        )
        self.gcntcn2 = UnitGCNTCN(32, 64, temporal_kernel=1, edge_size=edge_size)
        self.gcntcn3 = UnitGCNTCN(64, 64, temporal_kernel=1, edge_size=edge_size)
        self.gcntcn4 = UnitGCNTCN(64, 128, temporal_kernel=1, edge_size=edge_size)

        # self.linear_more_nodes = torch.nn.Linear(128, num_classes)
        # self.linear_coarser = torch.nn.Linear(768, num_classes)
        # self.linear = torch.nn.Linear(1152, num_classes)
        self.linear = torch.nn.Linear(768, num_classes)
        # self.linear = torch.nn.Linear(128, num_classes)

        self.temporal_conv_finale = TemporalConv(128, num_classes, kernel_size=2)
        self.temporal_conv_finale_skip = TemporalConv(2, num_classes, kernel_size=3)

        # self.sag_pooling = SAGPooling(in_channels=128, ratio=0.5, GNN=GCNConv, min_score=None, multiplier=1, nonlinearity='tanh')

    def forward(
        self,
        x,
        edge_index,
        smirk_context=None,
        motion_context=None,
        texture_context=None,
        cluster_indices=None,
        batch_indices=None,
    ):
        B, T, N, feat = x.shape
        edge_index = edge_index[0, :, :]  # use this for casme2 datasets
        # edge_index = edge_index[0, 0, :, :].T  # use this for casme2 datasets 3D

        # x = x.reshape(B, N * feat, T)
        # x = self.landmarks_feat_bn(x)
        # x = x.reshape(B, T, N, feat)

        # landmark cloning
        if motion_context is not None:
            if type(motion_context) == list:
                motion_context, texture_context = motion_context
                motion_context = motion_context.unsqueeze(1).repeat(1, T, 1, 1)
                (
                    texture_B,
                    texture_feat,
                    texture_T,
                    texture_x,
                    texture_y,
                ) = texture_context.shape
                texture_context = texture_context.view(
                    ((texture_B * texture_T), texture_feat, (texture_x * texture_y))
                )
                texture_context = self.pool_texture_for_graph(
                    texture_context
                )  # (B*T, feat, target_tokens)
                texture_context = texture_context.view(
                    (texture_B, texture_T, -1, texture_feat)
                )  # (B, T, feat, target_tokens)

                motion_context = torch.concat([motion_context, texture_context], dim=3)

            else:
                motion_context = motion_context.unsqueeze(1).repeat(1, T, 1, 1)
            x = torch.concat([x, motion_context], dim=3)
            # x = motion_context # what if we only use motion context?

        if smirk_context is not None:
            smirk_context = smirk_context.unsqueeze(2).repeat(1, 1, N, 1)
            x = torch.concat([x, smirk_context], dim=3)

        x1 = self.gcntcn1(
            x,
            edge_index,
            texture_context=texture_context,
            cluster_indices=cluster_indices,
            batch_indices=batch_indices,
        )
        x2 = self.gcntcn2(
            x1,
            edge_index,
            texture_context=texture_context,
            cluster_indices=cluster_indices,
            batch_indices=batch_indices,
        )
        x3 = self.gcntcn3(
            x2,
            edge_index,
            texture_context=texture_context,
            cluster_indices=cluster_indices,
            batch_indices=batch_indices,
        )
        x4 = self.gcntcn4(
            x3,
            edge_index,
            texture_context=texture_context,
            cluster_indices=cluster_indices,
            batch_indices=batch_indices,
        )

        # way 2 to merge
        x_t_more_nodes = x4.reshape((B, x4.shape[-1], -1))
        x_t_more_nodes = x_t_more_nodes.mean(2)

        # way 7 to merge
        cluster_indices = cluster_indices.reshape(
            (len(cluster_indices) // self.landmark_count, self.landmark_count)
        )[0, :]
        x_t1 = global_mean_pool(x=x4[:, 0, :, :], batch=cluster_indices, size=None)
        x_t2 = global_mean_pool(x=x4[:, 1, :, :], batch=cluster_indices, size=None)
        x_t_pooled = torch.stack([x_t1, x_t2], dim=1)  # (B, T, F)
        x_t_pooled_mean = x_t_pooled.mean(1)
        x_t_pooled = x_t_pooled_mean.reshape((x_t_pooled_mean.shape[0], -1))

        # way 8 to do subspace learning
        x4_mean = x4.mean(1)  # (B, N, F)

        # # final classification layer
        x_t_pooled = self.linear(x_t_pooled)
        return x_t_pooled, x4_mean


