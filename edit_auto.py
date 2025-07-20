import os
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate


def run_edit(model_name,
             output_dir,
             boundary_path,
             input_latent_codes_path='',
             num=1,
             latent_space_type='z',
             start_distance=0,
             end_distance=0,
             steps=1):
    """Runs InterfaceGAN editing with given arguments (no CLI required)."""
    
    # os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir, logger_name='generate_data')

    logger.info(f'Initializing generator: {model_name}')
    gan_type = MODEL_POOL[model_name]['gan_type']

    if gan_type == 'pggan':
        model = PGGANGenerator(model_name, logger)
        kwargs = {}
    elif gan_type == 'stylegan':
        model = StyleGANGenerator(model_name, logger)
        kwargs = {'latent_space_type': latent_space_type}
    else:
        raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

    logger.info('Preparing boundary.')
    if not os.path.isfile(boundary_path):
        raise ValueError(f'Boundary `{boundary_path}` does not exist!')
    boundary = np.load(boundary_path)
    np.save(os.path.join(output_dir, 'boundary.npy'), boundary)

    logger.info('Preparing latent codes.')
    if input_latent_codes_path and os.path.isfile(input_latent_codes_path):
        logger.info(f'  Load latent codes from `{input_latent_codes_path}`.')
        latent_codes = np.load(input_latent_codes_path)
        latent_codes = model.preprocess(latent_codes, **kwargs)
    else:
        logger.info('  Sample latent codes randomly.')
        latent_codes = model.easy_sample(num, **kwargs)

    np.save(os.path.join(output_dir, 'latent_codes.npy'), latent_codes)
    total_num = latent_codes.shape[0]

    logger.info(f'Editing {total_num} samples.')
    for sample_id in tqdm(range(total_num), leave=False):
        interpolations = linear_interpolate(
            latent_codes[sample_id:sample_id + 1],
            boundary,
            start_distance=start_distance,
            end_distance=end_distance,
            steps=steps
        )

        interpolation_id = 0
        for interpolations_batch in model.get_batch_inputs(interpolations):
            if gan_type == 'pggan':
                outputs = model.easy_synthesize(interpolations_batch)
   
            elif gan_type == 'stylegan':
                outputs = model.easy_synthesize(interpolations_batch, **kwargs)

            for image in outputs['image']:
                save_path = os.path.join(output_dir,
                                         f'{sample_id:03d}_{interpolation_id:03d}.jpg')
                cv2.imwrite(save_path, image[:, :, ::-1])
                interpolation_id += 1
        assert interpolation_id == steps
        logger.debug(f'  Finished sample {sample_id:3d}.')
    
    logger.info(f'Successfully edited {total_num} samples.')


# Optional: CLI entry point
if __name__ == '__main__':
    model_name = 'stylegan_celebahq'  # can be 'stylegan_celebahq' or 'stylegan_ffhq'
    boundary_dir = 'boundaries/stylegan_celebahq_smile_boundary.npy'  # contains *_boundary.npy
    output_dir = 'results/smile_edited_output'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default=model_name)
    parser.add_argument('-o', '--output_dir', type=str, default=output_dir)
    parser.add_argument('-b', '--boundary_path', type=str,default=boundary_dir)
    parser.add_argument('-i', '--input_latent_codes_path', type=str, default='')
    parser.add_argument('-n', '--num', type=int, default=1)
    parser.add_argument('-s', '--latent_space_type', type=str, default='z')
    parser.add_argument('--start_distance', type=float, default=0)
    parser.add_argument('--end_distance', type=float, default=0)
    parser.add_argument('--steps', type=int, default=1)

    args = parser.parse_args()
    run_edit(**vars(args))
