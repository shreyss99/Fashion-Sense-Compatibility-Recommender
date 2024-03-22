import os
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
import pickle


class ImageFeatureExtractor:
    def __init__(self):
        self.resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.resnet.trainable = False

        self.resnet = tensorflow.keras.Sequential([
            self.resnet,
            GlobalMaxPooling2D()
        ])

    def extract_image_features(self, input_path):
        img = image.load_img(input_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img_array = preprocess_input(expanded_img_array)
        result = self.resnet.predict(preprocessed_img_array).flatten()
        normalized_result = result / norm(result)

        return normalized_result

    def extract_features_for_images(self, image_folder_path):
        file_names = []
        features_list = []

        for image_name in tqdm(os.listdir(image_folder_path)):
            file_path = os.path.join(image_folder_path, image_name)
            file_names.append(file_path)
            features_list.append(self.extract_image_features(file_path))

        return file_names, features_list


if __name__ == "__main__":
    image_extractor = ImageFeatureExtractor()
    image_folder = 'images'
    file_names, features_list = image_extractor.extract_features_for_images(image_folder)

    pickle.dump(features_list, open('data/extracted_feature_embeddings.pkl', 'wb'))
    pickle.dump(file_names, open('data/image_file_names.pkl', 'wb'))
