import torch
import torch.nn as nn
from marlin_pytorch import Marlin
from sentence_transformers import SentenceTransformer

from datasets_utility import FocalLoss
from tiny_vit import tiny_vit_21m_224

attn_maps = []


def get_attn_hook(module, input, output):
    # output: (B, num_heads, N_tokens, N_tokens)
    attn_maps.append(output.detach().cpu())


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total_params: {total_params}, Trainable_params: {trainable_params}")
    return total_params, trainable_params





class TinyVIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # for eventual downstream classification
        linear_layer = nn.Linear(in_features=576, out_features=num_classes, bias=True)
        # linear_layer = nn.Linear(in_features=320, out_features=num_classes, bias=True)

        self.flatten = nn.Flatten()
        self.fc = linear_layer

        self.model = tiny_vit_21m_224(pretrained=True)
        # self.model = tiny_vit_5m_224(pretrained=True)
        # self.model = tiny_vit_21m_384(pretrained=True)
        # self.model = tiny_vit_21m_512(pretrained=True)
        self.model.head = linear_layer

        # extra engineering for graph
        # target_tokens = 14
        target_tokens = 47
        # target_tokens = 105 # for 185 3d nodes
        # target_tokens = 105  # for 14 nodes
        # target_tokens = 173  # for 173 nodes
        # target_tokens = 338 # for 51 edges
        # target_tokens = 4955
        # gcntcn4-related (64, 3, 14, 128)

        self.pool = nn.AdaptiveAvgPool1d(target_tokens)
        self.fc_pool_128 = nn.Linear(576, 128)
        self.patch_att_fc_128 = nn.Linear(128, num_classes)
        self.au_existence_fc = nn.Linear(128, 2)

    def forward(self, x, graph_context=None, au_aware_context=None):

        # if graph_context is not None:
        #     output, x_feat = self.model(x, graph_context=graph_context)
        # else:
        #     output, x_feat = self.model(x)

        output, x_feat = self.model(x)
        # extra engineering to extract the patches as soft attention to graph
        B, C, N = x_feat.shape
        # au_existence_logits = None
        # graph_context = None
        x_feat = x_feat.transpose(1, 2)  # (B, C, N)
        x_feat = self.pool(x_feat)  # (B, C, target_tokens)
        x_feat = x_feat.transpose(1, 2)  # (B, target_tokens, C)
        x_feat_128 = self.fc_pool_128(x_feat)  # (B, target_tokens, 128)

        if graph_context is not None:
            graph_context = graph_context[:, 1, :, :] - graph_context[:, 0, :, :]
            # Normalize graph_context to match the range of x_feat_128
            # Get statistics of x_feat_128
            x_feat_min = x_feat_128.min()
            x_feat_max = x_feat_128.max()
            x_feat_mean = x_feat_128.mean()
            x_feat_std = x_feat_128.std()

            # Method 1: Min-Max normalization to match x_feat_128 range
            graph_min = graph_context.min()
            graph_max = graph_context.max()
            graph_context_normalized = (graph_context - graph_min) / (
                graph_max - graph_min + 1e-8
            )
            graph_context_normalized = (
                graph_context_normalized * (x_feat_max - x_feat_min) + x_feat_min
            )
            graph_context = graph_context_normalized
            x_feat_128 = x_feat_128 * graph_context

        p_att_128_logits = x_feat_128.mean(1)
        p_att_128_logits = self.patch_att_fc_128(
            p_att_128_logits
        )  # (B, target_tokens, num_classes)
        if au_aware_context is not None:
            au_existence_logits = self.au_existence_fc(x_feat_128.mean(1))  # (B, 1)
        elif au_aware_context is None:
            au_existence_logits = None
        elif graph_context is None:
            graph_context = None

        return output, x_feat_128, p_att_128_logits, au_existence_logits




