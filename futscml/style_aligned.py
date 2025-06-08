import torch

T = torch.Tensor


def expand_first(feat: T, scale=1., ) -> T:
    """assumes that the first sample in the batch is the style sample"""
    b = feat.shape[0]
    if b <= 2:  # no unconditional guidance
        feat_style = feat[:1]
        if scale == 1:
            feat_style = feat_style.expand(1, b, *feat.shape[1:])
        else:
            feat_style = scale * feat_style.repeat(1, b, 1, 1, 1)
            # feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    else:
        feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
        if scale == 1:
            feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
        else:
            raise RuntimeError
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


def modify_qkv(q, k, v, attn_mask=None):
    """assumes that the first sample in the batch is the style sample, the rest are content samples
    param q: query [1, 4096, 256]"""
    q = adain(q)
    k = adain(k)

    k = concat_first(k, -2, scale=1.0)
    v = concat_first(v, -2)

    return q, k, v, attn_mask
