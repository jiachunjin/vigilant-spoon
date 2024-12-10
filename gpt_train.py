import os
import torch
import random
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from model.vq_llamagen import VQModel
from model.gpt.gpt import Transformer, ModelArgs
from utils.data_utils import get_dataloader

def get_models():
    gpt = Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, class_dropout_prob=0.0)) # 111M

    config = OmegaConf.create({
        'codebook_dim': 48,
        'encoder_ch_mult': [1, 1, 2, 2, 4],
        'decoder_ch_mult': [1, 1, 2, 2, 4],
        'bernoulli': True,
        'use_negative': False,
        'z_channels': 256,
        'dropout_p': 0.0,
        'matryoshka': False,
    })
    bae = VQModel(config)
    ckpt = torch.load('experiment/bin_48_bern_01/vqvae-bin_48_bern_01-80k', map_location='cpu')
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
    config = OmegaConf.load('config/gpt_dev.yaml')
    accelerator, output_dir = get_accelerator(config.train)
    gpt, bae = get_models()
    dataloader = get_dataloader(config.data)
    global_step = config.train.global_step if config.train.global_step is not None else 0

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
                sample, h = bae.encode_for_gpt(x)
                # target_id = random.randint(0, 255)
                # print(sample.shape, h.shape, sample.requires_grad, h.requires_grad)
                # binary_vec = sample[:, :target_id, :]
                cond_idx = y.long()
                # targets = h[:, target_id, :]

                logits, loss = gpt(binary_vec=sample, cond_idx=cond_idx, targets=h)

                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                
                accelerator.backward(loss)
                optimizer.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                loss = accelerator.gather(loss.detach()).mean().item()
                logs = {'loss': loss}
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