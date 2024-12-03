import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="nmoxDf7wfxJEIYLkLFDA"
)

# Streamlit application
st.title("PPE Kit Detection")
st.write("Upload an image to detect PPE items (e.g., mask, gloves, helmet).")

# File upload widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Save the image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file, format="JPEG")
            temp_file_path = temp_file.name

        # Send the image to Roboflow for inference
        result = CLIENT.infer(temp_file_path, model_id="ppe-kit-detection-ieadm/3")

        # Check for the presence of PPE items
        items_to_check = ['mask', 'vest', 'shoes', 'gloves', 'helmet', 'googles']
        presence = {item: False for item in items_to_check}

        for pred in result['predictions']:
            if pred['class'] in presence:
                presence[pred['class']] = True

        # Display detection results
        st.subheader("Detection Results:")
        for item, is_present in presence.items():
            st.write(f"- **{item.capitalize()}**: {'✅ Present' if is_present else '❌ Not Present'}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
