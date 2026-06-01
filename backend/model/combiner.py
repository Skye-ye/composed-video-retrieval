import torch
import torch.nn.functional as F
from clip import tokenize
from torch import nn

from .base import BaseRetrievalModel

from .backbone import clip
from .backbone.model import convert_weights


class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
        if len(input1.shape) == 2:
            input1 = input1.unsqueeze(dim=1)
            input2 = input2.unsqueeze(dim=1)
        attn_output, attn_weight = self.attention(input1, input2, input2)
        output = input2 + self.dropout1(attn_output)
        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)

        if len(attn_output.shape) == 2:
            return output, attn_weight
        else:
            return output.squeeze(1), attn_weight


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        x = token_dict["x"]
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C**0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + (
                dist_matrix.max() + 1
            ) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(
            dist_matrix, k=k, dim=-1, largest=False
        )
        density = (-(dist_nearest**2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = (
            density
            + torch.rand(density.shape, device=density.device, dtype=density.dtype)
            * 1e-6
        )

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(
            B, cluster_num
        )
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


class TokenConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        x = x + self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = token_dict["x"]
    idx_token = token_dict["idx_token"]
    agg_weight = token_dict["agg_weight"]

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(
        dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1)
    )
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight

    x_merged.index_add_(
        dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype)
    )
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new /= agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict["x"] = x_merged
    out_dict["token_num"] = cluster_num
    out_dict["idx_token"] = idx_token_new
    out_dict["agg_weight"] = agg_weight_new
    out_dict["mask"] = None
    return out_dict


class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.dim_out = dim_out
        self.conv = TokenConv(
            in_channels=embed_dim,
            out_channels=dim_out,
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k

    def forward(self, token_dict):
        x = token_dict["x"]
        x = self.conv(x)
        x = self.norm(x)
        token_score = self.score(x)
        token_weight = token_score.squeeze(2)
        if token_dict["mask"] is not None:
            token_weight.masked_fill_(
                (1 - token_dict["mask"]).to(torch.bool), float("-inf")
            )
        token_weight = token_weight.unsqueeze(2).exp()

        token_dict["x"] = x
        token_dict["token_score"] = token_score

        cluster_num = max(2, 1)
        idx_cluster, cluster_num = cluster_dpc_knn(
            token_dict, cluster_num, self.k, token_mask=token_dict["mask"]
        )

        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)
        return down_dict, token_dict


class Combiner(BaseRetrievalModel):
    def __init__(
        self,
        clip_model_name: str,
    ):
        super(Combiner, self).__init__()

        self.clip_model, _ = clip.load(
            clip_model_name,
            device="cpu",
            jit=False,
        )

        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        text_projection = getattr(self.clip_model, "text_projection", None)
        if text_projection is None:
            raise ValueError("Loaded CLIP model does not expose `text_projection`.")

        feature_dim = int(text_projection.shape[1])

        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=feature_dim * 2,
            nhead=8,
            dropout=0.5,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=feature_dim * 3,
            nhead=8,
            dropout=0.5,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.remained_extractor = CrossTransformer(0.5, feature_dim, 8)

        self.trans_1 = nn.TransformerEncoder(
            encoder_layer1, num_layers=6, enable_nested_tensor=False
        )
        self.trans_2 = nn.TransformerEncoder(
            encoder_layer2, num_layers=6, enable_nested_tensor=False
        )

        self.fc = nn.Linear(feature_dim * 3, feature_dim)
        self.clip_bs = 32

        self.ctm = CTM(
            sample_ratio=0.25, embed_dim=feature_dim, dim_out=feature_dim, k=3
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        if mode:
            convert_weights(self.clip_model)
        else:
            self.clip_model.float()
        return self

    def soft_argmax(self, vector, beta=10):
        return F.softmax(vector * beta, dim=-1)

    def encode_query_features(self, batch):
        ref_high_feature = batch["ref_vdo_fea"].float()
        captions = batch["caption"]
        captions_pos = batch["caption_pos"]
        b = ref_high_feature.shape[0]

        text_inputs = tokenize(captions, truncate=True).to(ref_high_feature.device)
        text_inputs_pos = tokenize(captions_pos, truncate=True).to(
            ref_high_feature.device
        )

        with torch.no_grad():
            text_inputs_list = torch.split(text_inputs, self.clip_bs)
            text_inputs_list_pos = torch.split(text_inputs_pos, self.clip_bs)
            text_features = []
            text_features_pos = []
            text_features_token = []
            text_features_token_pos = []
            for mini_batch_i in range(len(text_inputs_list)):
                mini_batch = text_inputs_list[mini_batch_i]
                text_features_tmp, text_features_token_tmp = (
                    self.clip_model.encode_text(mini_batch)
                )
                text_features_tmp = text_features_tmp.float()
                text_features_token_tmp = text_features_token_tmp.float()
                text_features.append(text_features_tmp)
                text_features_token.append(text_features_token_tmp)

                mini_batch_pos = text_inputs_list_pos[mini_batch_i]
                text_features_tmp_pos, text_features_token_tmp_pos = (
                    self.clip_model.encode_text(mini_batch_pos)
                )
                text_features_tmp_pos = text_features_tmp_pos.float()
                text_features_token_tmp_pos = text_features_token_tmp_pos.float()
                text_features_pos.append(text_features_tmp_pos)
                text_features_token_pos.append(text_features_token_tmp_pos)

            text_features = torch.vstack(text_features)
            text_features_token = torch.vstack(text_features_token)
            text_features_pos = torch.vstack(text_features_pos)
            text_features_token_pos = torch.vstack(text_features_token_pos)

        ref_high_feature_mean = (
            ref_high_feature.mean(dim=1)
            if ref_high_feature.ndim == 3
            else ref_high_feature
        )

        # implicit branch
        remained_text_features, _ = self.remained_extractor(
            ref_high_feature_mean, text_features
        )
        residual_text_features = text_features - remained_text_features

        fusion_fea_high = self.trans_1(
            torch.concat(
                [
                    ref_high_feature_mean.unsqueeze(1),
                    remained_text_features.unsqueeze(1),
                ],
                dim=-1,
            )
        )
        fusion_fea_high = self.trans_2(
            torch.concat([fusion_fea_high, residual_text_features.unsqueeze(1)], dim=-1)
        )
        fusion_fea_high = self.fc(fusion_fea_high).squeeze(1)

        # explicit branch
        # negation dpc cluster
        pad_mask_word_pos = (
            (text_inputs_pos == 0).to(ref_high_feature_mean.device).int()
        )
        pad_mask_word_pos = 1 - pad_mask_word_pos
        t_agg_weight_pos = text_inputs_pos.new_ones(
            text_inputs_pos.size(0), text_inputs_pos.size(1), 1
        )
        t_idx_token_pos = torch.arange(text_features_token_pos.size(1))[None, :].repeat(
            text_features_token_pos.size(0), 1
        )
        t_token_dict_pos = {
            "x": text_features_token_pos,  # text_token_fea (b, t, d)
            "token_num": text_features_token_pos.size(1),  # t
            "idx_token": t_idx_token_pos,  # text_range_token (b, t)
            "agg_weight": t_agg_weight_pos,  # text_token_fea (b, t, 1) all one
            "mask": pad_mask_word_pos.detach(),
        }  # text_token_mask (b, t) one is value, zero is empty
        t_token_dict_pos = self.ctm(t_token_dict_pos)

        token_similar_score_pos = ref_high_feature_mean[:, None, :] @ t_token_dict_pos[
            0
        ]["x"].permute(0, 2, 1)
        soft_index_pos = self.soft_argmax(token_similar_score_pos)
        remained_text_features_token_pos = (
            soft_index_pos @ t_token_dict_pos[0]["x"]
        ).squeeze(1)  # for negative sample
        injected_text_features_token_pos = (
            text_features_pos - remained_text_features_token_pos
        )  # for find original injected text

        t_idx_token_pos = torch.arange(text_features_token_pos.size(1))[None, :].repeat(
            text_features_token_pos.size(0), 1
        )
        t_token_dict_rem_pos = {
            "x": text_features_token_pos,  # text_token_fea (b, t, d)
            "token_num": text_features_token_pos.size(1),  # t
            "idx_token": t_idx_token_pos,  # text_range_token (b, t)
            "agg_weight": t_agg_weight_pos,  # retained token to 1, injected token to 0
            "mask": (
                (t_token_dict_pos[0]["idx_token"] == soft_index_pos.argmax(-1))
                * pad_mask_word_pos
            ).detach(),
        }  # retained token to 1, injected token to 0 text_token_mask (b, t) one is value, zero is empty
        t_token_dict_rem_pos = self.ctm(
            t_token_dict_rem_pos
        )  # retained feature into real retained feature and exlcluded feature

        # original coarse dpc cluster
        pad_mask_word = (text_inputs == 0).to(ref_high_feature_mean.device).int()
        pad_mask_word = 1 - pad_mask_word
        t_agg_weight = text_inputs.new_ones(text_inputs.size(0), text_inputs.size(1), 1)
        t_idx_token = torch.arange(text_features_token.size(1))[None, :].repeat(
            text_features_token.size(0), 1
        )
        t_token_dict = {
            "x": text_features_token,  # text_token_fea (b, t, d)
            "token_num": text_features_token.size(1),  # t
            "idx_token": t_idx_token,  # text_range_token (b, t)
            "agg_weight": t_agg_weight,  # text_token_fea (b, t, 1) all one
            "mask": pad_mask_word.detach(),
        }  # text_token_mask (b, t) one is value, zero is empty
        t_token_dict = self.ctm(t_token_dict)

        token_similar_score = injected_text_features_token_pos[
            :, None, :
        ] @ t_token_dict[0]["x"].permute(0, 2, 1)
        soft_index = self.soft_argmax(token_similar_score)
        t_idx_token = torch.arange(text_features_token.size(1))[None, :].repeat(
            text_features_token.size(0), 1
        )
        t_token_dict_rem = {
            "x": text_features_token,  # text_token_fea (b, t, d)
            "token_num": text_features_token.size(1),  # t
            "idx_token": t_idx_token,  # text_range_token (b, t)
            "agg_weight": t_agg_weight,  # text_token_fea (b, t, 1) all one
            "mask": (
                (t_token_dict[0]["idx_token"] == (1 - soft_index).argmax(-1))
                * pad_mask_word
            ).detach(),
        }  # text_token_mask (b, t) one is value, zero is empty
        t_token_dict_rem = self.ctm(t_token_dict_rem)

        # find which two are most similar
        soft_weight = (
            self.soft_argmax(
                (
                    t_token_dict_rem[0]["x"]
                    @ t_token_dict_rem_pos[0]["x"].permute(0, 2, 1)
                ).reshape(b, -1)
            ).reshape(b, 2, 2)
        ).sum(dim=-1, keepdim=True)
        remained_real_part = (soft_weight * t_token_dict_rem[0]["x"]).sum(1)

        fusion_fea_high_token_filteration = self.trans_1(
            torch.concat(
                [ref_high_feature_mean.unsqueeze(1), remained_real_part.unsqueeze(1)],
                dim=-1,
            )
        )
        fusion_fea_high_token = self.trans_2(
            torch.concat(
                [
                    fusion_fea_high_token_filteration,
                    injected_text_features_token_pos.unsqueeze(1),
                ],
                dim=-1,
            )
        )
        fusion_fea_high_token = self.fc(fusion_fea_high_token).squeeze(1)

        fusion_fea_high_token_negation = self.trans_1(
            torch.concat(
                [
                    ref_high_feature_mean.unsqueeze(1),
                    remained_text_features_token_pos.unsqueeze(1),
                ],
                dim=-1,
            )
        )
        fusion_fea_high_token_negation = self.trans_2(
            torch.concat(
                [
                    fusion_fea_high_token_negation,
                    injected_text_features_token_pos.unsqueeze(1),
                ],
                dim=-1,
            )
        )
        fusion_fea_high_token_negation = self.fc(
            fusion_fea_high_token_negation
        ).squeeze(1)

        return (
            fusion_fea_high,
            fusion_fea_high_token,
            remained_text_features,
            ref_high_feature_mean,
            fusion_fea_high_token_negation,
        )

    def encode_target_features(self, batch):
        target = batch["tgt_vdo_fea"].float()
        if target.ndim == 3:
            target = target.mean(dim=1)
        return target


def fdca(model, ckpt, **kwargs):
    if ckpt:
        model.load_from_pretrained(ckpt)
    return model
