# Vogue-Vaani

## Introduction

A fashion recommendation system that delivers customized suggestions based on user input by combining the power of transfer learning with the ResNet50 architecture and Annoy, an enhanced K-Nearest Neighbors algorithm. I successfully analyzed the image data by using transfer learning with ResNet-50 to perform feature extraction on a sizable dataset of more than 45,000 images. Using K-Nearest Neighbors, I implemented a similarity search approach to find the top 5 closest matches to a user's input, thereby offering personalized fashion recommendations. The system is easy to use and intuitive, making it possible to analyze image data accurately and successfully.

This recommendation system showcases the versatility and power of transfer learning, similarity search, and convolutional neural networks (CNNs).


## Recommendation Engine: Proposed Methodology 

In this project, we present a model that combines the Nearest Neighbor backed recommender with Convolutional Neural Network. According to the figure After training the neural networks, an inventory is chosen to generate recommendations, and a database is made for the inventory items. Based on the input image, the nearest neighbor's algorithm locates the most pertinent products and generates recommendations.

![work-model](https://user-images.githubusercontent.com/89743011/170476738-cdfcd048-8bfd-450c-ad58-20ec025d5b7c.png)


## Application Flow-Chart

The suggested method makes use of Sklearn Nearest Neighbors to produce recommendations. By doing this, we are able to determine who the input image's closest neighbors are. The Cosine Similarity measure was employed in this project as the similarity metric. The database is queried to extract the top 5 recommendations, and the resulting images are shown.

![flow-chart](https://user-images.githubusercontent.com/89743011/170476148-5c472690-675b-4907-91c4-9b9804668f6f.png)


## Convolution-Based Neural Systems

- A specialized neural network made for visual data, like photos and videos, is the convolutional neural network. CNNs are effective with non-image data as well, particularly in text classification and natural language processing.
- Its idea is comparable to that of a multilayer perceptron, or a basic neural network, as both use the same forwarding and backward propagation general principle.
- After pre-processing the data, transfer learning from ResNet50 is used to train the neural networks. To fine-tune the network model to serve the current problem, more layers are added in the final layers, replacing the architecture and weights from ResNet50. The architecture of ResNet50 is depicted in the figure.


## Getting the data

The images from Kaggle Fashion Product Images Dataset. The inventory is then run through the neural networks to classify and generate embeddings and the output  is then used to generate recommendations.

### The Figure shows a sample set of inventory data

![dataset-cover](https://user-images.githubusercontent.com/89743011/170478150-9204c659-06a4-48bf-8420-5fee02a3c4d3.png)


## Experiment and results

To get around the problems with the small size of the Fashion dataset, the idea of transfer learning is applied. 
Consequently, we utilize the DeepFashion dataset, which comprises 44,441 garment images, to pre-train the classification models. Using the obtained dataset, the networks are trained and verified. The model's high accuracy, low error, loss, and good f-score are demonstrated by the training results.


## Dataset Link

 - [Kaggle Dataset Big size 25 GB](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
 - [Kaggle Dataset Small size 593 MB](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)


## Screenshots

### App Home

![Screenshot (107)](https://github.com/shreyss99/Vogue-Vaani/blob/1ba2b111d1fd852592dae06fa7eef8a82a854bd8/Screenshots/Vogue%20Vaani%20-%20App.png)

### Outfits generated for Shoes

![Screenshot (107)](https://github.com/shreyss99/Vogue-Vaani/blob/3e4e5a865fca91ce3d04135d62d2c8086cd60e18/Screenshots/Shoes.png)

### Outfits generated for Shirt

![Screenshot (107)](https://github.com/shreyss99/Vogue-Vaani/blob/3e4e5a865fca91ce3d04135d62d2c8086cd60e18/Screenshots/Yellow%20Shirt.png)

### Outfits generated for Dress

![Screenshot (107)](https://github.com/shreyss99/Vogue-Vaani/blob/3e4e5a865fca91ce3d04135d62d2c8086cd60e18/Screenshots/Red%20Dress.png)


## Installation

Use pip to install the requirements.

~~~bash
pip install -r requirements.txt
~~~


## Usage

To run the web server, simply execute streamlit with the main recommender app:

```bash
streamlit run main.py
```


## Dependencies

- **OpenCV** - Open Source Computer Vision and Machine Learning software library
- **Tensorflow** - TensorFlow is an end-to-end open source platform for machine learning.
- **Tqdm** - tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable.
- **streamlit** - Streamlit is an open-source app framework for Machine Learning and Data Science teams. Create beautiful data apps in hours, not weeks.
- **pandas** - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
- **Pillow** - PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
- **scikit-learn** - Scikit-learn is a free software machine learning library for the Python programming language.
- **opencv-python** - OpenCV is a huge open-source library for computer vision, machine learning, and image processing.
