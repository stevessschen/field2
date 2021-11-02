import streamlit as st 

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility

%matplotlib inline
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

voc_classes = ['x','p','t']
NUM_CLASSES = len(voc_classes) + 1

st.write('Build ssd model...')
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
st.write('Load weight...')
model.load_weights('./checkpoints/weights.04-2.13.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

