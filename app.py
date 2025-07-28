import streamlit as st
import numpy as np
from runner import (
    generate_latent_and_original,
    edit_latent_image
)
from style_mixing import main

# -----------------------
# Page Setup
# -----------------------
st.set_page_config(page_title="InterfaceGAN Editor", layout="centered")
st.title("🎭 InterfaceGAN Latent Editor")

# -----------------------
# Utility Functions
# -----------------------
def load_new_latent():
    st.session_state.generator, st.session_state.kwargs, st.session_state.latent, img = generate_latent_and_original(st.session_state.model)
    st.session_state.original_img = img

def reset_sliders():
    st.session_state.smile = 0
    st.session_state.age = 0
    st.session_state.gender = 0
    st.session_state.glasses = 0

def update_edited_image():
    st.session_state.edited_img = edit_latent_image(
        generator=st.session_state.generator,
        kwargs=st.session_state.kwargs,
        latent_code=st.session_state.latent,
        smile_val=st.session_state.smile,
        age_val=st.session_state.age,
        gender_val=st.session_state.gender,
        glass_val=st.session_state.glasses
    )

# -----------------------
# Model Selection
# -----------------------
model = st.selectbox("🎨 Choose GAN Model", ['pggan_celebahq', 'stylegan_celebahq', 'stylegan_ffhq'])

# Initialize Session State
if 'model' not in st.session_state or st.session_state.model != model:
    st.session_state.model = model
    reset_sliders()
    load_new_latent()
    update_edited_image()

# -----------------------
# Top Controls: Random Face + Reset
# -----------------------
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🎲 Random Face"):
        load_new_latent()
        update_edited_image()
with col2:
    if st.button("♻️ Reset Sliders"):
        reset_sliders()
        update_edited_image()

# -----------------------
# Image Outputs: Original vs Edited
# -----------------------
col_img1, col_img2 = st.columns([1, 1])
with col_img1:
    st.subheader("🖼️ Original Image")
    st.image(st.session_state.original_img, caption="Unedited", width=300)
with col_img2:
    st.subheader("✨ Edited Image")
    st.image(st.session_state.edited_img, caption="Modified", width=300)

# -----------------------
# Latent Editing Sliders
# -----------------------
st.subheader("🎚️ Adjust Latent Attributes")
st.slider("Smile", -5, 5, key="smile", on_change=update_edited_image)
st.slider("Age", -5, 5, key="age", on_change=update_edited_image)
st.slider("Gender", -5, 5, key="gender", on_change=update_edited_image)
st.slider("Glasses", -5, 5, key="glasses", on_change=update_edited_image)

# -----------------------
# Style Mixing Section
# -----------------------
st.markdown("---")
st.header("🔀 Style Mixing")

if model in ['stylegan_celebahq', 'stylegan_ffhq']:
    if st.button("🔁 Perform Style Mixing"):
        with st.spinner("Generating mixed styles..."):
            src_img, tgt_img, mixed_img = main(model)
            st.session_state.src_img = src_img
            st.session_state.tgt_img = tgt_img
            st.session_state.mixed_img = mixed_img

    if 'mixed_img' in st.session_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(st.session_state.src_img, caption="🎨 Source Style", width=250)
        with col2:
            st.image(st.session_state.tgt_img, caption="🧠 Target Structure", width=250)
        with col3:
            st.image(st.session_state.mixed_img, caption="🧬 Mixed Output", width=250)
else:
    st.info("⚠️ Style mixing is only available for StyleGAN models (`stylegan_celebahq`, `stylegan_ffhq`).")
