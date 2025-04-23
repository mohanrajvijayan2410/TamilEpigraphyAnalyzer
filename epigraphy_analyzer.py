

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import VGG16
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pytesseract
import Levenshtein
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tempfile
import base64

# Configuration
IMAGE_SIZE = (224, 224)
DATASET_PATH ="content\8thcentury dataset"

# Specify the path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

class EpigraphyAnalyzer:
    def __init__(self):
        self.script_model = load_model("content\epigraphy__model.h5")
        self.feature_extractor = self._init_feature_extractor()
        self.class_labels = [p.name for p in Path(DATASET_PATH).iterdir()]
        self.reference_features = self._load_reference_features()

    def _init_feature_extractor(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
        return Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    def _load_reference_features(self):
        features = {}
        for root, _, files in os.walk(DATASET_PATH):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(root, file))
                    if img is not None:
                        feat = self._extract_features(img)
                        if feat is not None:
                            features[file] = feat
        return features

    def _extract_features(self, img):
        try:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            return self.feature_extractor.predict(np.expand_dims(img/255.0, axis=0))
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None

    def _preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to read image")
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            img = cv2.medianBlur(img, 3)
            return img
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None

    def _segment_characters(self, processed_img):
        try:
            contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            chars = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > 50:
                    char = processed_img[y:y+h, x:x+w]
                    char = cv2.copyMakeBorder(char, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                    chars.append(cv2.resize(char, (64, 64)))
            return chars
        except Exception as e:
            print(f"Segmentation error: {str(e)}")
            return []

    def analyze_inscription(self, img_path, ground_truth_text=None):
        result = {
            'script_type': 'Unknown',
            'century': 'Unknown',
            'segmented_chars': [],
            'extracted_text': '',
            'translated_text': '',
            'ocr_accuracy': 'Unknown'
        }
        try:
            # Script Classification
            img = cv2.resize(cv2.imread(img_path), IMAGE_SIZE) / 255.0
            pred = self.script_model.predict(np.expand_dims(img, axis=0))
            result['script_type'] = self.class_labels[np.argmax(pred)]

            # Preprocessing & Segmentation
            processed = self._preprocess_image(img_path)
            if processed is None:
                return result
            result['segmented_chars'] = self._segment_characters(processed)

            # OCR & Translation
            extracted_text = pytesseract.image_to_string(processed, lang='tam', config='--psm 6')
            result['extracted_text'] = extracted_text.strip()
            if extracted_text.strip():
                result['translated_text'] = GoogleTranslator(source='ta', target='en').translate(extracted_text)

            # OCR Accuracy Calculation
            if ground_truth_text:
                max_len = max(len(extracted_text), len(ground_truth_text))
                if max_len > 0:
                    distance = Levenshtein.distance(extracted_text, ground_truth_text)
                    result['ocr_accuracy'] = (1 - (distance / max_len)) * 100
                else:
                    result['ocr_accuracy'] = 100

            # Century Classification
            century_matches = []
            for char in result['segmented_chars']:
                feat = self._extract_features(char)
                if feat is not None:
                    similarities = [cosine_similarity(feat.reshape(1,-1), ref.reshape(1,-1))[0][0] for ref in self.reference_features.values()]
                    century_matches.append(max(similarities) > 0.7)
            result['century'] = "8th Century" if century_matches and (sum(century_matches)/len(century_matches) > 0.25) else "Unknown"
        except Exception as e:
            print(f"Processing Error: {str(e)}")
        return result


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def run_streamlit_app():
    st.set_page_config(page_title="Tamil Epigraphy Analyzer", layout="centered")

    # 🔥 Use local image as background
    bg_image_path = "assets\bg.png"  # Update as needed
    bg_image_base64 = get_base64_image(bg_image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_image_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🪵 Tamil Epigraphy Analyzer")
    st.markdown("Upload an inscription image to analyze its script, segment characters, and get Tamil to English translation.")

    uploaded_file = st.file_uploader("Upload Epigraphy Image", type=["jpg", "jpeg", "png"])
    ground_truth_text = st.text_area("Optional: Enter Ground Truth Text for OCR Accuracy", "")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_img_path = temp_file.name

        analyzer = EpigraphyAnalyzer()
        analysis = analyzer.analyze_inscription(temp_img_path, ground_truth_text.strip() or None)

        st.subheader("🖋️ Script Classification")
        st.success(f"**Script Type:** {analysis['script_type']}")
        st.success(f"**Century Prediction:** {analysis['century']}")

        st.subheader("🧠 Extracted Text")
        st.code(analysis['extracted_text'] or "No text extracted.")

        st.subheader("🌍 English Translation")
        st.code(analysis['translated_text'] or "No translation available.")

        st.subheader("📊 OCR Accuracy")
        st.info(f"{analysis['ocr_accuracy']}%" if analysis['ocr_accuracy'] != 'Unknown' else "Ground truth not provided.")

        st.subheader("🔡 Segmented Characters")
        if analysis['segmented_chars']:
            cols = st.columns(min(len(analysis['segmented_chars']), 5))
            for i, char_img in enumerate(analysis['segmented_chars']):
                cols[i % 5].image(char_img, use_container_width=True, clamp=True)
        else:
            st.warning("No characters segmented.")


if __name__ == "__main__":
    run_streamlit_app()
