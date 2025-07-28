import numpy as np
import matplotlib.pyplot as plt
from runner import load_generator,sample_latent,synthesize_image

def style_mix_latents(source_w, target_w):
    """Mixes styles from target into source at specified layers."""
    mixed = (source_w +target_w)/2
    return mixed

def show_images(images, titles):
    """Displays images side-by-side using matplotlib."""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        # img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        # img = (img * 255).clip(0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def main(model_name):
    generator,kwargs = load_generator(model_name,'Z')
    print(kwargs)
    source = sample_latent(generator, n_images=1, kwargs=kwargs)
    target = sample_latent(generator, n_images=1, kwargs=kwargs)
    
    s_img = synthesize_image(generator, source, kwargs=kwargs) 
    t_img = synthesize_image(generator, target, kwargs=kwargs)   
    # s_bgr  = s_img[:, :, ::-1]
    # t_bgr = t_img[:, :, ::-1]
    # # Load generator
    mixed = style_mix_latents(source,target)
    m_img = synthesize_image(generator, mixed, kwargs=kwargs) 
    # show_images(images=[s_img,m_img,t_img],titles=["source","mixed","target"])
    return s_img,t_img,m_img
        
    
    # mixed_img  = generator.synthesize(mixed_wp,  latent_space_type='WP')['image'][0]

    # # Visualize
   

if __name__ == '__main__':
    main('stylegan_celebahq')
