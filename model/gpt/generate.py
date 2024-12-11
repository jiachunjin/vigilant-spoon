import torch
from tqdm.auto import trange


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)
        # print('prefill, logits', logits.shape, logits[0, 0, :4])

    return torch.bernoulli(logits)
    # return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    # return sample(logits, **sampling_kwargs)
    return torch.bernoulli(logits), logits


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in trange(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            # cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs


@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        raise NotImplementedError
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    print('setup_caches done')
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new, 48), dtype=torch.float32, device=device)
    print(seq.shape)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    print(next_token, seq[:, T:T+1].shape)
    seq[:, T:T+1] = next_token
    # print(seq.shape)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    return seq[:, 1:, :]