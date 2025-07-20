import streamlit as st
import numpy as np
from runner import generate_latent_and_original, edit_latent_image

st.set_page_config(page_title="InterfaceGAN Editor", layout="centered")
st.title("🎭 InterfaceGAN Latent Editor")

# -----------------------
# Utility functions
# -----------------------
def load_new_latent():
    st.session_state.generator, st.session_state.kwargs, st.session_state.latent, img = generate_latent_and_original(st.session_state.model)
    st.session_state.original_img = img

def reset_sliders():
    st.session_state.smile = 0
    st.session_state.age = 0
    st.session_state.gender = 0

def update_edited_image():
    st.session_state.edited_img = edit_latent_image(
        generator=st.session_state.generator,
        kwargs=st.session_state.kwargs,
        latent_code=st.session_state.latent,
        smile_val=st.session_state.smile,
        age_val=st.session_state.age,
        gender_val=st.session_state.gender,
    )

# -----------------------
# Model selection
# -----------------------
model = st.selectbox("🎨 Choose GAN Model", ['pggan_celebahq', 'stylegan_celebahq', 'stylegan_ffhq'])

# Init session state on first run or when model changes
if 'model' not in st.session_state or st.session_state.model != model:
    st.session_state.model = model
    st.session_state.smile = 0
    st.session_state.age = 0
    st.session_state.gender = 0
    load_new_latent()
    update_edited_image()

# -----------------------
# Buttons: Random + Reset
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
col_img1, col_img2 = st.columns([1, 1])
# -----------------------
# Original Image
# -----------------------
with col_img1:
    st.subheader("🖼️ Original Image")
    st.image(st.session_state.original_img, caption="Unedited", width=300)
# -----------------------
# Edited Image
# -----------------------
with col_img2:
    st.subheader("✨ Edited Image")
    st.image(st.session_state.edited_img, caption="Modified", width=300)

# -----------------------
# Sliders for Editing
# -----------------------
st.subheader("🎚️ Adjust Attributes")
st.slider("Smile", -5, 5, key="smile", on_change=update_edited_image)
st.slider("Age", -5, 5, key="age", on_change=update_edited_image)
st.slider("Gender", -5, 5, key="gender", on_change=update_edited_image)

