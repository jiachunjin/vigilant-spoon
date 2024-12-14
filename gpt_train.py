import os
import torch
import random
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from einops import rearrange

from model.vq_llamagen import VQModel
# from model.gpt.gpt import Transformer, ModelArgs
from model.gpt.gpt_bin import Transformer_bin, ModelArgs

from utils.data_utils import get_dataloader

def get_models():
    # gpt = Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, class_dropout_prob=0.0)) # 111M
    gpt = Transformer_bin(ModelArgs(n_layer=12, n_head=12, dim=768, class_dropout_prob=0.0)) # 111M
    # gpt = Transformer_bin(ModelArgs(n_layer=24, n_head=16, dim=1024, class_dropout_prob=0.0)) # 111M
    # gpt = Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, class_dropout_prob=0.0))
    config = OmegaConf.create({
        'codebook_dim': 64,
        'encoder_ch_mult': [1, 1, 2, 2, 4],
        'decoder_ch_mult': [1, 1, 2, 2, 4],
        'bernoulli': True,
        'use_negative': False,
        'z_channels': 256,
        'dropout_p': 0.0,
        'matryoshka': False,
    })
    bae = VQModel(config)
    ckpt = torch.load('ckpts/a800/vqvae-matryoshka_64-85k', map_location='cpu')
    bae.load_state_dict(ckpt, strict=False)
    bae.eval()
    bae.requires_grad_(False)

    return gpt, bae

def get_accelerator(config):
    output_dir = os.path.join('experiment', config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=None if config.report_to == 'no' else config.report_to,
        mixed_precision=config.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    return accelerator, output_dir

def main():
    config = OmegaConf.load('config/gpt_bin.yaml')
    accelerator, output_dir = get_accelerator(config.train)
    gpt, bae = get_models()
    dataloader = get_dataloader(config.data)
    global_step = config.train.global_step if config.train.global_step is not None else 0

    if config.train.gpt_resume_path is not None:
        ckpt = torch.load(config.train.gpt_resume_path, map_location='cpu')
        m, u = gpt.load_state_dict(ckpt, strict=False)
        print('missing: ', m)
        print('unexpected: ', u)
        if accelerator.is_main_process:
            print(f'GPT ckpt loaded from {config.train.gpt_resume_path}')

    params_to_learn = list(gpt.parameters())
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = 1e-4,
        betas        = (0.9, 0.999),
        weight_decay = 1e-2,
        eps          = 1e-8,
    )

    if accelerator.is_main_process:
        print('Number of learnable parameters: ', sum(p.numel() for p in params_to_learn if p.requires_grad))

    gpt, dataloader, optimizer = accelerator.prepare(gpt, dataloader, optimizer)
    bae = bae.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj)

    training_done = False
    progress_bar = tqdm(
        total=config.train.num_iters,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    while not training_done:
        for x, y in dataloader:
            gpt.train()
            with accelerator.accumulate([gpt]):
                with torch.no_grad():
                    sample, h = bae.encode_for_gpt(x)
                    # sample_1 = rearrange(sample[:, :, :2], 'b (h w) d -> b h w d', h=16)
                    # h_1 = rearrange(h[:, :, :2], 'b (h w) d -> b h w d', h=16)
                    # sample_2 = rearrange(sample[:, :, 2:4], 'b (h w) d -> b h w d', h=16)
                    # h_2 = rearrange(h[:, :, 2:4], 'b (h w) d -> b h w d', h=16)
                    # sample_3 = rearrange(sample[:, :, 4:8], 'b (h w) d -> b h w d', h=16)
                    # h_3 = rearrange(h[:, :, 4:8], 'b (h w) d -> b h w d', h=16)

                    # sample_1 = rearrange(sample_1, 'b (h1 h2) (w1 w2) d -> b (h1 w1) (h2 w2 d)', h1=4, w1=4)
                    # h_1 = rearrange(h_1, 'b (h1 h2) (w1 w2) d -> b (h1 w1) (h2 w2 d)', h1=4, w1=4)
                    # sample_2 = rearrange(sample_2, 'b (h1 h2) (w1 w2) d -> b (h1 w1) (h2 w2 d)', h1=4, w1=4)
                    # h_2 = rearrange(h_2, 'b (h1 h2) (w1 w2) d -> b (h1 w1) (h2 w2 d)', h1=4, w1=4)
                    # sample_3 = rearrange(sample_3, 'b (h1 h2) (w1 w2) d -> b (h1 w1) (h2 w2 d)', h1=4, w1=4)
                    # h_3 = rearrange(h_3, 'b (h1 h2) (w1 w2) d -> b (h1 w1) (h2 w2 d)', h1=4, w1=4)

                    # sample = sample_1
                    # h = h_1
                    # sample = torch.cat([sample_1, sample_2], dim=1)
                    # h = torch.cat([h_1, h_2], dim=1)
                    # print(sample_1.shape, sample_2.shape, sample_3.shape)
                    # print(h_1.shape, h_2.shape, h_3.shape)
                    h1 = 16
                    w1 = 16
                    h2 = 1
                    w2 = 1
                    sample_1 = rearrange(sample[:, :, :2], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    h_1 = rearrange(h[:, :, :2], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    sample_2 = rearrange(sample[:, :, 2:4], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    h_2 = rearrange(h[:, :, 2:4], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    sample_3 = rearrange(sample[:, :, 4:8], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    h_3 = rearrange(h[:, :, 4:8], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    sample_4 = rearrange(sample[:, :, 8:16], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    h_4 = rearrange(h[:, :, 8:16], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    sample_5 = rearrange(sample[:, :, 16:24], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    h_5 = rearrange(h[:, :, 16:24], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    sample_6 = rearrange(sample[:, :, 24:32], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    h_6 = rearrange(h[:, :, 24:32], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    sample_7 = rearrange(sample[:, :, 32:], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)
                    h_7 = rearrange(h[:, :, 32:], 'b (h1 w1 h2 w2) d -> b (h1 w1) (h2 w2 d)', h1=h1, h2=h2, w1=w1, w2=w2)

                    sample = [sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7]
                    h = [h_1, h_2, h_3, h_4, h_5, h_6, h_7]
                    # sample = torch.cat([sample_1, sample_2, sample_3], dim=1)
                    # h = torch.cat([h_1, h_2, h_3], dim=1)

                    seq_len = 256 * 8
                    block_size = 256
                    num_blocks = seq_len // block_size
                    # 构造因果 block mask
                    block_mask = torch.tril(torch.ones(num_blocks, num_blocks))  # 下三角矩阵表示因果关系
                    block_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
                    # 转换为布尔类型（True 表示遮挡）
                    causal_block_mask = block_mask != 0  # (seq_len, seq_len)t_product_attention
                    causal_block_mask = causal_block_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

                cond_idx = y.long()
                # targets = h[:, target_id, :]

                _, loss, losses = gpt(binary_vec=sample, cond_idx=cond_idx, targets=h, mask=causal_block_mask)

                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                
                accelerator.backward(loss)
                optimizer.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                loss = accelerator.gather(loss.detach()).mean().item()
                loss1 = accelerator.gather(losses[0].detach()).mean().item()
                loss2 = accelerator.gather(losses[1].detach()).mean().item()
                loss3 = accelerator.gather(losses[2].detach()).mean().item()
                loss4 = accelerator.gather(losses[3].detach()).mean().item()
                # loss5 = accelerator.gather(losses[4].detach()).mean().item()
                # loss6 = accelerator.gather(losses[5].detach()).mean().item()
                # loss7 = accelerator.gather(losses[6].detach()).mean().item()


                logs = {'loss': loss, 'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss4': loss4}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                gpt.eval()
                state_dict = accelerator.unwrap_model(gpt).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"gpt-{config.train.exp_name}-{global_step // 1000}k"))

            if global_step >= config.train.num_iters:
                training_done = True
                break


if __name__ == '__main__':
    main()