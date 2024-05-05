import os
from typing import List, Optional, Tuple
import click
from torch_utils import gen_utils
import numpy as np
import PIL.Image
import torch
from configurationLoader import returnRepoConfig
repoConfig = returnRepoConfig('stylegan_cfg.yaml')




os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


# ----------------------------------------------------------------------------


# TODO: this is no longer true for StyleGAN3, we have 14 layers irrespective of resolution
def parse_styles(s: str) -> List[int]:
    """
    Helper function for parsing style layers. s will be a comma-separated list of values, and these can be
    either ranges ('a-b'), ints ('a', 'b', 'c', ...), or the style layer names ('coarse', 'middle', 'fine').

    A combination of these can also be used. For example, if the user wishes to mix the 'coarse' and 'fine'
    layers, then the input can be: 'coarse,fine'. If just the 'middle' and '14-17' layers are to be used,
    then 'middle,14-17' or '14-17,middle' can be the used as input.

    The repeated styles will be deleted, as these won't add anything to our final result.
    """
    style_layers_dict = {'coarse': list(range(0, 4)), 'middle': list(range(4, 8)), 'fine': list(range(8, 18))}
    str_list = s.split(',')
    nums = []
    for el in str_list:
        if el in style_layers_dict:
            nums.extend(style_layers_dict[el])
        else:
            nums.extend(gen_utils.num_range(el, remove_repeated=True))
    # Sanity check: delete repeating numbers and limit values between 0 and 17
    nums = list(set([max(min(x, 17), 0) for x in nums]))
    return nums


# TODO: For StyleGAN3, there's only 'coarse' and 'fine' groups, though the boundary is not 100% clear
def style_names(max_style: int, file_name: str, desc: str, col_styles: List[int]) -> Tuple[str, str]:
    """
    Add the styles if they are being used (from the StyleGAN paper)
    to both the file name and the new directory to be created.
    """
    if list(range(0, 4)) == col_styles:
        styles = 'coarse_styles'
    elif list(range(4, 8)) == col_styles:
        styles = 'middle_styles'
    elif list(range(8, max_style)) == col_styles:
        styles = 'fine_styles'
    elif list(range(0, 8)) == col_styles:
        styles = 'coarse+middle_styles'
    elif list(range(4, max_style)) == col_styles:
        styles = 'middle+fine_styles'
    elif list(range(0, 4)) + list(range(8, max_style)) == col_styles:
        styles = 'coarse+fine_styles'
    else:
        styles = 'custom_styles'

    file_name = f'{file_name}-{styles}'
    desc = f'{desc}-{styles}'

    return file_name, desc


def _parse_cols(s: str, G, device: torch.device, truncation_psi: float) -> torch.Tensor:
    """s can be a path to a npy/npz file or a seed number (int)."""
    if os.path.isfile(s[0]):
        # Collect all latent vectors from files into a list
        all_w_arrays = []
        for el in s:
            w_el = gen_utils.get_latent_from_file(el)  # Load np.ndarray from file
            all_w_arrays.append(np.squeeze(w_el))
        # Stack all arrays along a new axis to create a batch dimension
        all_w = np.stack(all_w_arrays)
        # Convert the numpy array to a PyTorch tensor and send to device
        all_w = torch.from_numpy(all_w).to(device)
    else:
        # Process seed numbers, converting each to an integer and generating Z vectors
        all_z = np.stack([np.random.RandomState(int(seed)).randn(G.z_dim) for seed in s])
        all_w = G.mapping(torch.from_numpy(all_z).to(device), None)

    # Apply the truncation trick
    w_avg = G.mapping.w_avg
    w = w_avg + (all_w - w_avg) * truncation_psi

    return w


# ----------------------------------------------------------------------------


# We group the different types of style-mixing (grid and video) into a main function
@click.group()
def main():
    pass


# ----------------------------------------------------------------------------
# Configs for the style_mixer
only_process_diagonal = repoConfig.mixing.only_process_diagonal
save_grid = repoConfig.mixing.save_grid
limit_grid_size = repoConfig.mixing.limit_grid_size
col_styles = repoConfig.mixing.col_styles 
network_pkl = repoConfig.inference.model
reference_image_dir = repoConfig.mixing.ref_image_dir #r"C:\Users\Rasmu\Repos\StyleGAN\stylegan3-fun-main\out\projection\00025-manual_test\rows"
content_image_dir = repoConfig.mixing.content_image_dir #row_seeds_dir
out_dir = repoConfig.mixing.out_dir
out_name_addon = repoConfig.mixing.out_name_addon

# ----------------------------------------------------------------------------
@main.command(name="style_mix")
@click.pass_context
@click.option('--network', 'network_pkl', default=network_pkl, help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
@click.option('--device', help='Device to use for image generation; using the CPU is slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
# Synthesis options
@click.option('--row-seeds_dir', default=content_image_dir, help='Random seeds to use for image rows', required=True)
@click.option('--col-seeds_dir', default=reference_image_dir, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=parse_styles, help='Style layers to use; can pass "coarse", "middle", "fine", or a list or range of ints', default=col_styles, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=out_dir, show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='', show_default=True)
def generate_style_mix(
        ctx: click.Context,
        network_pkl: str,
        cfg: Optional[str],
        device: Optional[str],
        row_seeds_dir: Optional[str],
        col_seeds_dir: Optional[str],
        col_styles: List[int],
        truncation_psi: float,
        noise_mode: str,
        anchor_latent_space: bool,
        outdir: str,
        description: str,
):
    # """Generate style-mixing images using pretrained network pickle.

    # Examples:2,13,19,200,400
 
    # python style_mixing.py grid -rows 13,15,17 -cols 24,6,73 --network=C:\Users\Rasmu\StyleGAN\stylegan3-main\training_runs\00003-stylegan2-GCI_Front_100_formatted-gpus1-batch16-gamma0.8192\network-snapshot-001200.pkl
    # python style_mixing.py grid -rows 24,26,9 -cols 13,15,17 --network=C:\Users\Rasmu\StyleGAN\stylegan3-main\training_runs\00003-stylegan2-GCI_Front_100_formatted-gpus1-batch16-gamma0.8192\network-snapshot-001200.pkl
    # python style_mixing.py grid -rows=85,100,75,458,1500 -cols=55,821,1789,293 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    # python style_mixing.py grid -rows=6,11,18,20,100 -cols=2,13,19,200,400 --styles "coarse" --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    # """

    row_seeds = [os.path.join(row_seeds_dir, f) for f in os.listdir(row_seeds_dir)]
    col_seeds = [os.path.join(col_seeds_dir, f) for f in os.listdir(col_seeds_dir)]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # TODO: add class_idx
    device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')

    # Load the network
    G = gen_utils.load_network('G_ema', network_pkl, cfg, device)

    # Setup for using CPU
    if device.type == 'cpu':
        gen_utils.use_cpu(G)

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Sanity check: loaded model and selected styles must be compatible
    max_style = G.mapping.num_ws
    if max(col_styles) > max_style:
        click.secho(f'WARNING: Maximum col-style allowed: {max_style - 1} for loaded network "{network_pkl}" '
                    f'of resolution {G.img_resolution}x{G.img_resolution}', fg='red')
        click.secho('Removing col-styles exceeding this value...', fg='blue')
        col_styles[:] = [style for style in col_styles if style < max_style]

    print('Generating W vectors...')
    all_seeds = row_seeds+col_seeds
    all_w = _parse_cols(all_seeds, G, device, truncation_psi)
    all_seeds = [os.path.basename(seed) for seed in all_seeds]
    row_seeds = [os.path.basename(row) for row in row_seeds]
    col_seeds = [os.path.basename(col) for col in col_seeds]
    #print(f'row_seeds: {row_seeds}, col_seeds: {col_seeds}')
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    print('Generating images...')
    all_images = gen_utils.w_to_img(G, all_w, noise_mode)
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for i, row_seed in enumerate(row_seeds):
        for j, col_seed in enumerate(col_seeds):
            if only_process_diagonal and i != j:
                continue
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = gen_utils.w_to_img(G, w, noise_mode)[0]
            image_dict[(row_seed, col_seed)] = image
    # Name of grid and run directory
    grid_name = 'grid'
    description = 'stylemix-grid' if len(description) == 0 else description
    # Add to the name the styles (from the StyleGAN paper) if they are being used
    grid_name, description = style_names(max_style, grid_name, description, col_styles)
    # Create the run dir with the given name description
    run_dir = gen_utils.make_run_dir(outdir, description)

    if save_grid:
        print('Saving image grid...')
        row_limit = len(row_seeds) if len(row_seeds) < limit_grid_size else limit_grid_size
        col_limit = len(col_seeds) if len(col_seeds) < limit_grid_size else limit_grid_size
        print(f'Limiting grid size to {row_limit}x{col_limit} images...')
        W = G.img_resolution
        H = G.img_resolution
        canvas = PIL.Image.new(gen_utils.channels_dict[G.synthesis.img_channels],  # Handle RGBA case
                            (W * (col_limit + 1), H * (row_limit + 1)), 'black')
        
        for row_idx, row_seed in enumerate(([0] + row_seeds)[:row_limit+1]):
            for col_idx, col_seed in enumerate(([0] + col_seeds)[:col_limit+1]):
                if row_idx == 0 and col_idx == 0:
                    continue
                if only_process_diagonal: 
                    if row_idx == col_idx or row_idx == 0 or col_idx == 0:
                        key = (row_seed, col_seed)
                        if row_idx == 0:
                            key = (col_seed, col_seed)
                        if col_idx == 0:
                            key = (row_seed, row_seed)
                        canvas.paste(PIL.Image.fromarray(image_dict[key],
                                                        gen_utils.channels_dict[G.synthesis.img_channels]),
                                    (W * col_idx, H * row_idx))
                else:
                    key = (row_seed, col_seed)
                    if row_idx == 0:
                        key = (col_seed, col_seed)
                    if col_idx == 0:
                        key = (row_seed, row_seed)
                    canvas.paste(PIL.Image.fromarray(image_dict[key],
                                                    gen_utils.channels_dict[G.synthesis.img_channels]),
                                (W * col_idx, H * row_idx))
        canvas.save(os.path.join(out_dir, f'{grid_name}.png'))

    print('Saving individual images...')
    if only_process_diagonal:
        for idx in range(len(row_seeds)):
            key = (row_seeds[idx], col_seeds[idx])
            out_path = os.path.join(out_dir, f'{os.path.splitext(os.path.basename(row_seeds[idx]))[0]+out_name_addon}.png')
            PIL.Image.fromarray(image_dict[key],
                                gen_utils.channels_dict[G.synthesis.img_channels]).save(out_path)
    else:
        for (row_seed, col_seed), image in image_dict.items():
            out_path = os.path.join(out_dir, f'{os.path.splitext(os.path.basename(row_seed))[0]+out_name_addon}.png')
            PIL.Image.fromarray(image,
                                gen_utils.channels_dict[G.synthesis.img_channels]).save(out_path)
    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'row_seeds': row_seeds,
        'col_seeds': col_seeds,
        'col_styles': col_styles,
        'truncation_psi': truncation_psi,
        'noise_mode': noise_mode,
        'run_dir': run_dir,
        'description': description,
    }
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------
