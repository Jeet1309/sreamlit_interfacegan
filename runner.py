import os
import numpy as np
import cv2
from models.model_settings import MODEL_POOL
from utils.manipulator import linear_interpolate
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from itertools import product

def load_generator(model_name, latent_space_type='z'):
    if model_name == 'pggan_celebahq':
        generator = PGGANGenerator(model_name)
        kwargs = {}
    elif model_name in ['stylegan_celebahq', 'stylegan_ffhq']:
        generator = StyleGANGenerator(model_name)
        kwargs = {'latent_space_type': latent_space_type}
    else:
        raise NotImplementedError(f'Model {model_name} is not supported.')

    generator.model_name = model_name
    generator.load()
    return generator, kwargs


def sample_latent(generator, n_images=1, kwargs=None):
    codes = generator.easy_sample(n_images)
    if kwargs:
        codes = generator.preprocess(codes, **kwargs)
    return codes


def synthesize_image(generator, latent_codes, kwargs=None):
    if kwargs:
        outputs = generator.easy_synthesize(latent_codes, **kwargs)
    else:
        outputs = generator.easy_synthesize(latent_codes)
    return outputs['image'][0]  # return raw image array


def save_image(image_array, path):
    image_bgr = image_array[:, :, ::-1]  # Convert RGB to BGR
    cv2.imwrite(path, image_bgr)


def load_boundaries(boundary_dir, model_name):
    smile = np.load(os.path.join(boundary_dir, f'{model_name}_smile_boundary.npy'))
    age = np.load(os.path.join(boundary_dir, f'{model_name}_age_boundary.npy'))
    gender = np.load(os.path.join(boundary_dir, f'{model_name}_gender_boundary.npy'))
    return smile, age, gender


def apply_edits(latent_code, smile_val, age_val, gender_val, smile_boundary, age_boundary, gender_boundary):
    direction = (smile_val * smile_boundary +
                 age_val * age_boundary +
                 gender_val * gender_boundary)
    return latent_code + direction.reshape(1, -1)


def run_editor(model_name='stylegan_ffhq',
               smile_val=0, age_val=0, gender_val=0,
               output_dir='results/edited_output',
               boundary_dir='boundaries',
               n_images=1,
               latent_override=None):

    os.makedirs(output_dir, exist_ok=True)

    generator, kwargs = load_generator(model_name)
    latent_code = latent_override if latent_override is not None else sample_latent(generator, n_images=n_images, kwargs=kwargs)

    np.save(os.path.join(output_dir, 'latent.npy'), latent_code)
    original_img = synthesize_image(generator, latent_code, kwargs=kwargs)
    save_image(original_img, os.path.join(output_dir, 'original.png'))

    smile_b, age_b, gender_b = load_boundaries(boundary_dir, model_name)
    edited_code = apply_edits(latent_code, smile_val, age_val, gender_val, smile_b, age_b, gender_b)
    np.save(os.path.join(output_dir, 'latent_edited.npy'), edited_code)

    edited_img = synthesize_image(generator, edited_code, kwargs=kwargs)
    save_image(edited_img, os.path.join(output_dir, 'edited.png'))

    print(f"[✔] Images saved in {output_dir}")
    return original_img, edited_img


def generate_latent_and_original(model_name, output_dir='results/edited_output'):
    os.makedirs(output_dir, exist_ok=True)
    generator, kwargs = load_generator(model_name)
    latent_code = sample_latent(generator, n_images=1, kwargs=kwargs)
    original_img = synthesize_image(generator, latent_code, kwargs=kwargs)
    return generator, kwargs, latent_code, original_img

def edit_latent_image(generator, kwargs, latent_code, smile_val, age_val, gender_val, boundary_dir='boundaries'):
    smile_b, age_b, gender_b = load_boundaries(boundary_dir, generator.model_name)
    edited_code = apply_edits(latent_code, smile_val, age_val, gender_val, smile_b, age_b, gender_b)
    edited_img = synthesize_image(generator, edited_code, kwargs=kwargs)
    return edited_img
