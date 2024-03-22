import os
import streamlit as st
from PIL import Image
from fashion_recommendation.fashion_recommender import FashionRecommender
from image_feature_extractor.image_feature_extractor import ImageFeatureExtractor

st.markdown("<h1 style='text-align: center;'>Vogue Vaani</h1>", unsafe_allow_html=True)
image_path = "app_images/vogue.jpeg"
try:
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Vogue Vaani', use_column_width=True)
    else:
        st.error("Image file not found at specified path.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Load FashionRecommender
fashion_recommender = FashionRecommender('data/extracted_feature_embeddings.pkl', 'data/image_file_names.pkl')


def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.mkdir('uploads')

        file_path = os.path.join('uploads', uploaded_file.name)

        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return file_path

    except OSError:
        return None


# File upload
user_uploaded_file = st.file_uploader("Choose an image: ")

if user_uploaded_file is not None:
    uploaded_file_path = save_uploaded_file(user_uploaded_file)
    if uploaded_file_path:
        # Display the uploaded image
        display_image = Image.open(user_uploaded_file)
        st.image(display_image)

        # Extract the features for the uploaded image using the ImageFeatureExtractor
        image_extractor = ImageFeatureExtractor()
        extracted_features = image_extractor.extract_image_features(uploaded_file_path)

        # Get the recommended images
        recommended_image_paths = fashion_recommender.recommend_similar_images(extracted_features)

        num_images = len(recommended_image_paths)

        # Create columns
        cols = st.columns(num_images)

        # Display images in each column
        for i in range(num_images):
            with cols[i]:
                st.image(recommended_image_paths[i], use_column_width=False)
