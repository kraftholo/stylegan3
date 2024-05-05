import copy
import os
from time import perf_counter
import sys
import click
import glob
from typing import List, Tuple
import imageio
import numpy as np
import PIL.Image

import torch
import torch.nn.functional as F

import dnnlib
from dnnlib.util import format_time
import legacy

from torch_utils import gen_utils
from tqdm import tqdm
from pytorch_ssim import SSIM  # from https://github.com/Po-Hsun-Su/pytorch-ssim

from network_features import VGG16FeaturesNVIDIA, DiscriminatorFeatures

from metrics import metric_utils
# ----------------------------------------------------------------------------
def compute_regularization_loss(noise_buffs, device, batch_size):
    reg_loss = torch.tensor(0.0, device=device)
    for noise in noise_buffs.values():
        noise = noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        while noise.shape[2] > 8:
            rolled_x = torch.roll(noise, shifts=1, dims=3)
            rolled_y = torch.roll(noise, shifts=1, dims=2)
            reg_loss += (noise * rolled_x).mean() ** 2
            reg_loss += (noise * rolled_y).mean() ** 2
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss

def apply_noise(w_opts, noise_buffs, w_noise_scale, device):
    noise = torch.cat([torch.randn_like(w_opts, device=device) * w_noise_scale for _ in noise_buffs.values()], dim=0)
    return w_opts + noise[:w_opts.size(0)]

def preprocess_images(targets, device, img_resolution):
    # Convert PIL images to tensors and preprocess
    tensors = [torch.tensor(np.array(target.resize((img_resolution, img_resolution), PIL.Image.BILINEAR)).transpose(2, 0, 1), device=device).float().unsqueeze(0) for target in targets]
    return torch.cat(tensors)  # Normalize from [0, 255] to [-1, 1]

# ----------------------------------------------------------------------------

def project_batch(
        G,
        targets: List[PIL.Image.Image],
        *,
        projection_seed: int,
        truncation_psi: float,
        num_steps: int = 1000,
        w_avg_samples: int = 10000,
        initial_learning_rate: float = 0.1,
        initial_noise_factor: float = 0.05,
        constant_learning_rate: bool = False,
        lr_rampdown_length: float = 0.25,
        lr_rampup_length: float = 0.05,
        noise_ramp_length: float = 0.75,
        regularize_noise_weight: float = 1e5,
        project_in_wplus: bool = False,
        loss_paper: str = 'sgan2',
        normed: bool = False,
        sqrt_normed: bool = False,
        start_wavg: bool = True,
        z_samples,
        w_samples,
        w_avg,
        w_std,
        noise_buffs,
        buf,
        device: torch.device,
        D=None) -> Tuple[torch.Tensor, dict]:

    targets_tensor = preprocess_images(targets, device, G.img_resolution)
    vgg16 = metric_utils.get_feature_detector("https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl", device)
    target_features = vgg16(targets_tensor, resize_images=False, return_lpips=True)
    if w_avg.dim() == 1:
        w_avg = w_avg.unsqueeze(0)  # Make it [1, C]
    # Repeat w_avg across the batch size, no additional layers
    w_opts = w_avg.repeat(len(targets), 1, 1)  # Shape: [batch_size, 1, C]
    # Set requires_grad to True for optimization
    w_opts = w_opts.clone().detach().requires_grad_(True)
    w_out = torch.zeros([num_steps, len(targets)] + list(w_opts.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opts] + list(noise_buffs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    lr_ramps = torch.linspace(0, 1, num_steps, device=device)
    noise_scales = w_std * initial_noise_factor * torch.maximum(torch.tensor(0.0, device=device), 1.0 - lr_ramps / noise_ramp_length) ** 2

    for step in range(num_steps):
        w_noise = apply_noise(w_opts, noise_buffs, noise_scales[step], device)
        synth_images = G.synthesis(w_noise if project_in_wplus else w_noise.repeat(1, G.mapping.num_ws, 1), noise_mode='const')
        synth_images = (synth_images + 1) * (255/2)
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum(dim=1)
        reg_loss = compute_regularization_loss(noise_buffs, device, len(targets))

        total_loss = dist + reg_loss * regularize_noise_weight
        optimizer.zero_grad(set_to_none=True)
        total_loss.mean().backward()
        optimizer.step()
        w_out[step] = w_opts.detach()

        if step % 10 == 0 or step == num_steps - 1:
            print(f'Step {step + 1}/{num_steps}, Total Loss {total_loss.mean().item():.7e}', end='\r')

    return w_out if project_in_wplus else w_out.repeat(1, G.mapping.num_ws, 1)

# ----------------------------------------------------------------------------
# Configs for the projector
save_visualization= True
out_vis_dir = os.path.join(os.getcwd(), 'out', "projecting4", "visualizations")
steps = 1000
w_avg_samples = 10000
init_lr = 0.1
noise_weight_regularization = 1e5 #was 1e5
constant_learning_rate = False # Add flag to use a constant learning rate throughout the optimization (turn off the rampup/rampdown)'
batch_size = 1 # Define a suitable batch size based on GPU capacity and image dimensions
network_pkl = r"E:\thesis\repos\Stylegan3\training_runs\00000-stylegan2-stylegan-processed-images-512-gpus1-batch16-gamma6.5536\network-snapshot-002800.pkl"
target_dir = r"E:\thesis\datasets\all-train-images-512-front" #r"C:\Users\Rasmu\Repos\StyleGAN\stylegan3-fun-main\out\projection\00025-manual_test\rows"
out_image_dir = os.path.join(os.getcwd(), 'out', "projecting4", "images")
out_latent_dir = os.path.join(os.getcwd(), 'out', "projecting4", "latents")
out_name_addon = '_proj'
# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('--network', '-net', 'network_pkl', default=network_pkl, help='Network pickle filename', required=True)
@click.option('--cfg', help='Config of the network, used only if you want to use one of the models that are in torch_utils.gen_utils.resume_specs', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']))
@click.option('--target_dir', '-t', 'target_fname_directory', type=click.Path(exists=True, dir_okay=True), default=target_dir,help='Target image directory containing files to project to', required=True, metavar='FILE')
# Optimization options
@click.option('--num-steps', '-nsteps', help='Number of optimization steps', type=click.IntRange(min=0), default=steps, show_default=True)
@click.option('--init-lr', '-lr', 'initial_learning_rate', type=float, help='Initial learning rate of the optimization process', default=init_lr, show_default=True)
@click.option('--constant-lr', 'constant_learning_rate', default=constant_learning_rate, help='Add flag to use a constant learning rate throughout the optimization (turn off the rampup/rampdown)')
@click.option('--reg-noise-weight', '-regw', 'regularize_noise_weight', type=float, help='Noise weight regularization', default=noise_weight_regularization, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=303, show_default=True)
# Video options
# Options on which space to project to (W or W+) and where to start: the middle point of W (w_avg) or a specific seed
@click.option('--project-in-wplus', '-wplus', default=True, help='Project in the W+ latent space')
@click.option('--start-wavg', '-wavg', type=bool, help='Start with the average W vector, ootherwise will start from a random seed (provided by user)', default=True, show_default=True)
@click.option('--projection-seed', type=int, help='Seed to start projection from', default=None, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi to use in projection when using a projection seed', default=0.7, show_default=True)
# Decide the loss to use when projecting (all other apart from o.g. StyleGAN2's are experimental, you can select the VGG16 features/layers to use in the im2sgan loss)
@click.option('--loss-paper', '-loss', type=click.Choice(['sgan2', 'im2sgan', 'discriminator', 'clip']), help='Loss to use (if using "im2sgan", make sure to norm the VGG16 features)', default='sgan2', show_default=True)
# im2sgan loss options (try with and without them, though I've found --vgg-normed to work best for me)
@click.option('--vgg-normed', 'normed', is_flag=True, help='Add flag to norm the VGG16 features by the number of elements per layer that was used')
@click.option('--vgg-sqrt-normed', 'sqrt_normed', is_flag=True, help='Add flag to norm the VGG16 features by the square root of the number of elements per layer that was used')
def run_projection(
        ctx: click.Context,
        network_pkl: str,
        cfg: str,
        target_fname_directory: str,
        num_steps: int,
        initial_learning_rate: float,
        constant_learning_rate: bool,
        regularize_noise_weight: float,
        seed: int,
        project_in_wplus: bool,
        start_wavg: bool,
        projection_seed: int,
        truncation_psi: float,
        loss_paper: str,
        normed: bool,
        sqrt_normed: bool,
):
    
    # Creating the output directories
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    if not os.path.exists(out_latent_dir):
        os.makedirs(out_latent_dir)
    if save_visualization and not os.path.exists(out_vis_dir):
        os.makedirs(out_vis_dir)
    # """Project given image to the latent space of pretrained network pickle.
    torch.manual_seed(seed)
    # If we're not starting from the W midpoint, assert the user fed a seed to start from
    if not start_wavg:
        if projection_seed is None:
            ctx.fail('Provide a seed to start from if not starting from the midpoint. Use "--projection-seed" to do so')

    # Load networks.
    # If model name exists in the gen_utils.resume_specs dictionary, use it instead of the full url
    try:
        network_pkl = gen_utils.resume_specs[cfg][network_pkl]
    except KeyError:
        # Otherwise, it's a local file or an url
        pass
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
    if loss_paper == 'discriminator':
        # We must also load the Discriminator
        with dnnlib.util.open_url(network_pkl) as fp:
            D = legacy.load_network_pkl(fp)['D'].requires_grad_(False).to(device)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    # Compute w stats.
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    if project_in_wplus:  # Thanks to @pbaylies for a clean way on how to do this
        print('Projecting in W+ latent space...')
        if start_wavg:
            print(f'Starting from W midpoint using {w_avg_samples} samples...')
            w_avg = torch.mean(w_samples, dim=0, keepdim=True)  # [1, L, C]
        else:
            print(f'Starting from a random vector (seed: {projection_seed})...')
            z = np.random.RandomState(projection_seed).randn(1, G.z_dim)
            w_avg = G.mapping(torch.from_numpy(z).to(device), None)  # [1, L, C]
            w_avg = G.mapping.w_avg + truncation_psi * (w_avg - G.mapping.w_avg)
    else:
        print('Projecting in W latent space...')
        w_samples = w_samples[:, :1, :]  # [N, 1, C]
        if start_wavg:
            print(f'Starting from W midpoint using {w_avg_samples} samples...')
            w_avg = torch.mean(w_samples, dim=0, keepdim=True)  # [1, 1, C]
        else:
            print(f'Starting from a random vector (seed: {projection_seed})...')
            z = np.random.RandomState(projection_seed).randn(1, G.z_dim)
            w_avg = G.mapping(torch.from_numpy(z).to(device), None)[:, :1, :]  # [1, 1, C]; fake w_avg
            w_avg = G.mapping.w_avg + truncation_psi * (w_avg - G.mapping.w_avg)
    w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    # Setup noise inputs (only for StyleGAN2 models)
    noise_buffs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    # Init noise.
    for buf in noise_buffs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    # Prepare for image processing
    image_paths = glob.glob(os.path.join(target_fname_directory, '*.png'))
    image_batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    for batch in tqdm(image_batches, desc='Processing batches', unit='batch'):
        images = [PIL.Image.open(fname).convert('RGB') for fname in batch]
        target_images = []
        for target in images:
            w, h = target.size
            s = min(w, h)
            target_cropped = target.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_resized = target_cropped.resize((G.img_resolution, G.img_resolution), PIL.Image.BILINEAR)
            target_images.append(target_resized)
        filenames = [os.path.basename(path) for path in batch]  # Extract filenames to maintain correspondence

        projected_w_steps = project_batch(
            G, targets=target_images, num_steps=num_steps, initial_learning_rate=initial_learning_rate,
            constant_learning_rate=constant_learning_rate, regularize_noise_weight=regularize_noise_weight,
            project_in_wplus=project_in_wplus, start_wavg=start_wavg, projection_seed=projection_seed,
            truncation_psi=truncation_psi, loss_paper=loss_paper, normed=normed, sqrt_normed=sqrt_normed,
            z_samples=z_samples, w_samples=w_samples, w_avg=w_avg, w_std=w_std, noise_buffs=noise_buffs,
            buf=buf, device=device,  D=D if loss_paper == 'discriminator' else None)
        
        # Handle output for each image in the batch
        for idx, (image, filename) in enumerate(zip(target_images, filenames)):
            final_w = projected_w_steps[-1, idx]  # Get the last step's latent vector for the current image
            synth_image = gen_utils.w_to_img(G, dlatents=final_w, noise_mode='const')[0]  # Generate image from W
            out_img_path = os.path.join(out_image_dir, f"{os.path.splitext(filename)[0]+out_name_addon}.png")
            out_latent_path = os.path.join(out_latent_dir, f"{os.path.splitext(filename)[0]+out_name_addon}.npy")
            # Save the synthesized image and the corresponding W vector
            PIL.Image.fromarray(synth_image, 'RGB').save(out_img_path)
            np.save(out_latent_path, final_w.cpu().numpy())
            # Optionally save the target image
            if save_visualization:
                visualization = np.concatenate([np.array(image), synth_image], axis=1)
                out_vis_path = os.path.join(out_vis_dir, f"{os.path.splitext(filename)[0]+out_name_addon}.png")
                PIL.Image.fromarray(visualization, 'RGB').save(out_vis_path)

        print("Batch processing complete.")

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------
