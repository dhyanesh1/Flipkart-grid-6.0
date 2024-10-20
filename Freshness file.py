import cv2
import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.layers import Activation
import tensorflow as tf
import matplotlib.pyplot as plt

# Classify fresh/rotten
def print_fresh(res):
    threshold_fresh = 0.10  # set according to standards
    threshold_medium = 0.35  # set according to standards
    if res < threshold_fresh:
        print("The item is FRESH!")
    elif threshold_fresh < res < threshold_medium:
        print("The item is MEDIUM FRESH")
    else:
        print("The item is NOT FRESH")


def pre_proc_img(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0) # Add an extra dimension for the batch
    return img


def evaluate_rotten_vs_fresh(image_path):
    # Use CustomObjectScope to handle potential custom objects or layers
     from keras.models import load_model
     from keras.utils import CustomObjectScope
     from keras.layers import Activation

    # Define the custom objects
     model = load_model('/content/rottenvsfresh (1).h5')

img_path = '/content/img0.png'
img = pre_proc_img(img_path)
plt.imshow(img[0])
is_rotten = model.predict(img)[0][0] # Predict using the preprocessed image
print(f'Prediction: {is_rotten}')
print_fresh(is_rotten)
