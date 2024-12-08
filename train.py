import os
import torch

from PIL import Image
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from model.vq_llamagen import VQModel
from loss.loss import VQLoss
from utils.data_utils import get_dataloader
from utils.misc import reconstrut_image

def get_models(config):
    vqvae = VQModel(config.model)
    ckpt = torch.load('/home/jiachun/codebase/rfsq/ckpts/vq_ds16_c2i.pt', map_location='cpu')
    vqvae.load_state_dict(ckpt['model'], strict=False)

    return vqvae

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
    config = OmegaConf.load('config/matry_64.yaml')
    accelerator, output_dir = get_accelerator(config.train)
    vqvae = get_models(config)
    dataloader = get_dataloader(config.data)
    loss = VQLoss(disc_start=config.train.disc_start, disc_weight=config.train.disc_weight)
    global_step = config.train.global_step if config.train.global_step is not None else 0

    if config.train.ae_resume_path is not None:
        ckpt = torch.load(config.train.ae_resume_path, map_location='cpu')
        m, u = vqvae.load_state_dict(ckpt, strict=False)
        print('missing: ', m)
        print('unexpected: ', u)
        if accelerator.is_main_process:
            print(f'autoencoder ckpt loaded from {config.train.ae_resume_path}')
    if config.train.loss_resume_path is not None:
        ckpt = torch.load(config.train.loss_resume_path, map_location='cpu')
        m, u = loss.load_state_dict(ckpt, strict=False)
        print('missing: ', m)
        print('unexpected: ', u)
        if accelerator.is_main_process:
            print(f'loss ckpt loaded from {config.train.loss_resume_path}')


    params_to_learn = list(vqvae.parameters())
    disc_params = list(loss.discriminator.parameters())
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = 1e-4,
        betas        = (0.9, 0.999),
        weight_decay = 1e-2,
        eps          = 1e-8,
    )

    optimizer_disc = torch.optim.AdamW(
        disc_params,
        lr           = 1e-4,
        betas        = (0.9, 0.999),
        weight_decay = 1e-2,
        eps          = 1e-8,
    )

    if accelerator.is_main_process:
        print('Number of learnable parameters: ', sum(p.numel() for p in params_to_learn if p.requires_grad))
    
    vqvae, loss, dataloader, optimizer, optimizer_disc = accelerator.prepare(vqvae, loss, dataloader, optimizer, optimizer_disc)
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj)    

    training_done = False
    progress_bar = tqdm(
        total=config.train.num_iters,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    from loss.loss import Loss_middle
    Loss_intermediate = Loss_middle()

    while not training_done:
        for x, _ in dataloader:
            vqvae.train()
            with accelerator.accumulate([vqvae, loss]):
                quant = vqvae.module.encode_binary(x)
                rec = vqvae.module.decode_fsq(quant)
                rec_intermediate = vqvae.module.decode_intermediate(quant)

                loss_gen = loss(x, rec, optimizer_idx=0, global_step=global_step+1, 
                                   last_layer=vqvae.module.decoder.last_layer)

                loss_intermediate = Loss_intermediate(x, rec_intermediate)

                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                # accelerator.backward(loss_gen)
                accelerator.backward(loss_gen + loss_intermediate)
                optimizer.step()

                loss_disc = loss(x, rec, optimizer_idx=1, global_step=global_step+1)
                optimizer_disc.zero_grad()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(disc_params, 1.0)
                accelerator.backward(loss_disc)
                optimizer_disc.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                loss_gen = accelerator.gather(loss_gen.detach()).mean().item()
                loss_disc = accelerator.gather(loss_disc.detach()).mean().item()
                loss_intermediate = accelerator.gather(loss_intermediate.detach()).mean().item()
                logs = {'loss_gen': loss_gen, 'loss_disc': loss_disc, 'loss_intermediate': loss_intermediate}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.val_every == 0 and accelerator.is_main_process:
                vqvae.eval()
                recon_path = os.path.join('./assets/recons', config.train.exp_name)
                os.makedirs(recon_path, exist_ok=True)
                
                with torch.no_grad():
                    img_dec = reconstrut_image(vqvae)
                img_dec.save(os.path.join(recon_path, f'{global_step:05d}.png'))

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                vqvae.eval()
                state_dict = accelerator.unwrap_model(vqvae).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"vqvae-{config.train.exp_name}-{global_step // 1000}k"))
                state_dict = accelerator.unwrap_model(loss).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"loss-{config.train.exp_name}-{global_step // 1000}k"))

            if global_step >= config.train.num_iters:
                training_done = True
                break




if __name__ == '__main__':
    main()
