import einops
import torch as th

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock
from ldm.modules.diffusionmodules.util import timestep_embedding


def do_forward(sd_net, ref_net, x_noisy, timesteps, context):
    hs_ref = []
    hs_sd = []

    txt_context = context['c_crossattn'][0]

    t_emb = timestep_embedding(timesteps, sd_net.model_channels, repeat_only=False)
    sd_emb = sd_net.time_embed(t_emb)
    ref_emb = ref_net.time_embed(t_emb)
    h_ref = x_noisy.type(ref_net.dtype)
    h_sd = x_noisy.type(sd_net.dtype)

    oflow_context = None
    for module_idx in range(len(sd_net.input_blocks)):
        sd_module = sd_net.input_blocks[module_idx]
        ref_module = ref_net.input_blocks[module_idx]
        h_ref, h_sd = do_forward_in_module(sd_module, ref_module, sd_emb, ref_emb, h_sd, h_ref,
                                                          txt_context, oflow_context)

        hs_sd.append(h_sd)
        hs_ref.append(h_ref)

    sd_module = sd_net.middle_block
    ref_module = ref_net.middle_block
    h_ref, h_sd = do_forward_in_module(sd_module, ref_module, sd_emb, ref_emb, h_sd, h_ref, txt_context,
                                                      oflow_context)

    for module_idx in range(len(sd_net.output_blocks)):
        sd_module = sd_net.output_blocks[module_idx]
        ref_module = ref_net.output_blocks[module_idx]

        h_sd = th.cat([h_sd, hs_sd.pop()], dim=1)
        h_sd = h_sd.type(sd_net.dtype)

        h_ref = th.cat([h_ref, hs_ref.pop()], dim=1)
        h_ref = h_ref.type(ref_net.dtype)

        h_ref, h_sd = do_forward_in_module(sd_module, ref_module, sd_emb, ref_emb, h_sd, h_ref,
                                                          txt_context, oflow_context)

    h_sd = h_sd.type(x_noisy.dtype)
    return sd_net.out(h_sd)


def do_forward_in_module(sd_module, ref_module, sd_emb, ref_emb, h_sd, h_ref, context, oflow_context):
    for module_idx_idx in range(len(sd_module)):
        # print(f'{ref_module[module_idx_idx]}, {ref_module[module_idx_idx].__class__.__mro__}')
        if isinstance(ref_module[module_idx_idx], SpatialTransformer):
            # spatial-attention, continues ldm.modules.attention line 278
            spatial_h = th.cat((h_sd, h_ref), dim=-1)

            h_sd = sd_module[module_idx_idx](spatial_h, context=context)
            h_ref = ref_module[module_idx_idx](h_ref, context=context)
        else:
            if TimestepBlock in ref_module[module_idx_idx].__class__.__mro__:
                h_sd = sd_module[module_idx_idx](h_sd, sd_emb)
                h_ref = ref_module[module_idx_idx](h_ref, ref_emb)
            else:
                h_sd = sd_module[module_idx_idx](h_sd)
                h_ref = ref_module[module_idx_idx](h_ref)

    return h_ref, h_sd