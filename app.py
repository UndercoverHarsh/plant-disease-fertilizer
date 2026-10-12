import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Plant Disease & Fertilizer Assistant", page_icon="🌿", layout="wide")
st.title("🌿 AI-Based Plant Disease Detection & Fertilizer Recommendation")
st.caption("Student MVP prototype. Demo mode shows the workflow; image diagnosis requires a trained model.")

DISEASES = ["Tomato — Early blight", "Tomato — Late blight", "Tomato — Healthy",
            "Potato — Early blight", "Potato — Late blight", "Potato — Healthy",
            "Pepper — Bacterial spot", "Pepper — Healthy"]

GUIDANCE = {
    "Early blight": ("Possible fungal leaf disease. Confirm with a local agricultural expert.",
        ["Remove heavily affected leaves where appropriate.", "Avoid overhead watering and improve airflow.",
         "Use only locally approved treatments according to the label and expert advice."],
        "Fertilizer does not cure disease. Base application on soil testing and crop growth stage; avoid excess nitrogen."),
    "Late blight": ("Potentially serious disease that may progress quickly. Confirm promptly.",
        ["Inspect affected plants and avoid moving wet plant material through the crop.",
         "Reduce leaf wetness where practical.", "Contact a local agricultural expert about approved management options."],
        "Fertilizer will not cure late blight. Use soil-test-based nutrition and avoid excess nitrogen."),
    "Bacterial spot": ("Possible bacterial leaf disease; other conditions can look similar.",
        ["Avoid handling plants while foliage is wet.", "Sanitize tools and manage affected plant material appropriately.",
         "Ask a local extension officer about crop- and region-specific management."],
        "Use a soil test and crop-stage recommendation; fertilizer does not cure bacterial infection."),
    "Healthy": ("The selected label indicates a healthy plant. Continue monitoring.",
        ["Inspect leaves regularly for new spots, yellowing, or wilting.", "Avoid waterlogging and maintain good crop hygiene.",
         "Maintain balanced nutrition based on soil-test results."],
        "Apply fertilizer according to crop, growth stage, and soil test; avoid excessive application.")
}
def get_guidance(label):
    key = next((k for k in GUIDANCE if k.lower() in label.lower()), "Healthy")
    return GUIDANCE[key]

st.sidebar.header("Prototype status")
st.sidebar.write("Working: image upload, recommendation workflow, soil-input form.")
st.sidebar.write("Not yet complete: automatic AI prediction unless you add a trained model.")
st.sidebar.info("Educational prototype only; not a validated diagnostic or fertilizer prescription.")

a,b = st.columns(2)
with a:
    st.subheader("1. Upload leaf image")
    upload = st.file_uploader("Choose a JPG or PNG", type=["jpg","jpeg","png"])
    image = Image.open(upload) if upload else None
    if image:
        st.image(image, caption="Uploaded leaf image", use_container_width=True)
    else:
        st.write("Use a clear, well-lit photo of one leaf.")
with b:
    st.subheader("2. Disease label and recommendation")
    model_path = Path("models/plant_disease_model.keras")
    mode = st.radio("Mode", ["Demo/manual mode", "Use trained TensorFlow model"])
    if mode == "Use trained TensorFlow model":
        if not upload:
            st.warning("Upload an image first.")
        elif not model_path.exists():
            st.error("No model found. Add a compatible trained model at models/plant_disease_model.keras.")
        else:
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                arr = np.asarray(image.convert("RGB").resize((224,224)), dtype=np.float32)/255.0
                scores = model.predict(np.expand_dims(arr,0), verbose=0)[0]
                idx = int(np.argmax(scores))
                label = DISEASES[idx]
                st.success(f"Predicted class: {label}")
                st.metric("Model score", f"{float(scores[idx])*100:.1f}%")
                st.caption("Score is not a guarantee of diagnosis. Class order must match the model training labels.")
                summary, actions, fertilizer = get_guidance(label)
                st.write(summary)
                for item in actions: st.write("• " + item)
                st.write("**Fertilizer guidance:** " + fertilizer)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
    else:
        st.warning("Demo mode does not diagnose the uploaded image. Select a sample label to demonstrate recommendations.")
        label = st.selectbox("Sample disease label", DISEASES)
        if st.button("Generate recommendation", type="primary"):
            summary, actions, fertilizer = get_guidance(label)
            st.success(f"Demo label: {label}")
            st.write(summary)
            for item in actions: st.write("• " + item)
            st.write("**Fertilizer guidance:** " + fertilizer)

st.divider()
st.subheader("3. Soil information")
c1,c2,c3,c4 = st.columns(4)
with c1: crop = st.selectbox("Crop", ["Tomato","Potato","Pepper","Other"])
with c2: ph = st.number_input("Soil pH (if known)", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
with c3: n_value = st.number_input("Nitrogen test value (lab units)", min_value=0.0, value=0.0, step=1.0)
with c4: has_npk = st.checkbox("NPK test results available")
if st.button("Summarize soil inputs"):
    st.write(f"Crop: **{crop}** | Entered soil pH: **{ph:.1f}**")
    if has_npk:
        st.write(f"N test value entered: **{n_value:g}**. Interpret using the laboratory's units and local agronomic guidance.")
    else:
        st.write("No NPK test results confirmed. Obtain a soil test before deciding fertilizer rates.")
    if ph < 5.5: st.warning("Acidic pH entered; confirm with a soil test and crop-specific advice.")
    elif ph > 7.5: st.warning("Alkaline pH entered; confirm with a soil test and crop-specific advice.")
    else: st.info("This pH is within a broad range for many crops, but crop-specific suitability varies.")
st.caption("Exact fertilizer doses are intentionally not prescribed without verified soil-test units, crop stage, region, and agronomic advice.")
