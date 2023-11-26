import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# Create a resnet model
# weights are trained from the ImageNet dataset
# include_top is False as we will add our own top layer
# input shape of images is resized to (224, 224, 3)
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet.trainable = False

resnet = tensorflow.keras.Sequential([
    resnet,
    GlobalMaxPooling2D()
])

print(resnet.summary())
