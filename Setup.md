

# InterfaceGAN Latent Space Editing (PGGAN / StyleGAN)

This repository provides a simple interface to perform latent space manipulation using pretrained GANs such as **PGGAN** and **StyleGAN**, based on [InterfaceGAN](https://github.com/ShenYujun/InterFaceGAN).

## 🚀 Features

- Edit facial attributes like **smile**, **age**, and **gender** in GAN-generated images.
- Supports:
  - PGGAN-CelebA-HQ
  - StyleGAN-CelebA-HQ
  - StyleGAN-FFHQ
- Latent vector generation, saving, and interactive manipulation using sliders.
- Lightweight **Streamlit UI** for manual attribute control.

---

## 🧠 Requirements

- Python 3.6+
- PyTorch 1.7.1 with CUDA 11.0 (custom install)
- NVIDIA GPU (needs to have support of at least CUDA 10+)

---

## ⚙️ Setup

### 1. Create and activate virtual environment

```bash
python -m venv env
source env/bin/activate     # On Windows: env\Scripts\activate
````

### 2. Install PyTorch 1.7.1 with CUDA 11.0

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
Pillow
matplotlib
streamlit
scikit-learn
```

---

## 🖼️ Usage

### Edit latent vector with specific attribute

```bash
python edit.py \
  -m pggan_celebahq \
  -b boundaries/pggan_celebahq_smile_boundary.npy \
  -n 1 \ ##optional
  -o results/pggan_celebahq_smile_editing \ 
  --start_distance -3.0 \  ##optional 
  --end_distance 3.0 \  ##optional
  --step 10  ##optional
```

### Run the Streamlit UI

```bash
streamlit run app_streamlit.py
```

Then open the local URL shown (usually `http://localhost:8501`).

---

## 📦 Models & Latent Boundaries

Place your model files and boundary `.npy` files in the `boundaries/` folder.

Pretrained boundaries:

* `pggan_celebahq_smile_boundary.npy`
* `pggan_celebahq_age_boundary.npy`
* `pggan_celebahq_gender_boundary.npy`

---

## 🧪 Troubleshooting

### CUDA not detected?

Make sure:

* You're running on a machine with an NVIDIA GPU.
* PyTorch was installed with the correct CUDA version.
* Driver version ≥ 450.

Run this check in Python:

```python
import torch
print(torch.cuda.is_available())           # Should be True
print(torch.cuda.get_device_name(0))       # Should show your GPU name
```

---

## 📝 Citation

This project is based on:

> Shen, Yujun, et al. "Interpreting the Latent Space of GANs for Semantic Face Editing." CVPR 2020.

---

## 🧑‍💻 Author

Built and extended by \Jeet Gurbani
Adapted for PyTorch 1.7.1 and CUDA 11.0

```


