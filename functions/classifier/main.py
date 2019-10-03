import tensorflow as tf
import numpy as np
import uuid
from flask import jsonify
import cv2

from google.cloud import storage
from google.cloud import firestore

import os

model = None
BUCKET = os.environ.get('GCS_BUCKET')
storage_client = storage.Client()
firestore_client = firestore.Client()

classes = ['jCKM0uQFyKeF8gieQPti', 'jCKM0uQFyKeF8gieQPti','jCKM0uQFyKeF8gieQPti', 'jCKM0uQFyKeF8gieQPti','jCKM0uQFyKeF8gieQPti', 'jCKM0uQFyKeF8gieQPti','jCKM0uQFyKeF8gieQPti', 'jCKM0uQFyKeF8gieQPti','jCKM0uQFyKeF8gieQPti']
FILENAME_TEMPLATE = '{}.jpg'

if not os.path.exists('/tmp/model'):
    os.makedirs('/tmp/model')
    
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image = image / 255.0  # normalize to [0,1] range

    return image


def download_blob(bucket_name, src_blob_name, dst_file_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(src_blob_name)

    blob.download_to_filename(dst_file_name)

    print('Blob {} downloaded to {}.'.format(
        src_blob_name,
        dst_file_name))

def load_model():
    global model
    if not os.path.exists('/tmp/model/dog_cat_M.h5'):
        download_blob(BUCKET, 'dog_cat_M.h5', '/tmp/model/dog_cat_M.h5')

    path = '/tmp/model/dog_cat_M.h5'
    model = tf.keras.models.load_model(path)

def get_product(product_id):
    product = firestore_client.collection('stores').document(product_id).get()
    return product.to_dict()

def classifier(request):
    global model

    # Set up CORS to allow requests from arbitrary origins.
    # See https://cloud.google.com/functions/docs/writing/http#handling_cors_requests
    # for more information.
    # For maxiumum security, set Access-Control-Allow-Origin to the domain
    # of your own.
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {
        'Access-Control-Allow-Origin': '*'
    }
	    
    if model is None:
        load_model()
    
    f = request.files['image']
    raw_img = f.read()
    img_preprocessed = preprocess_image(raw_img)
    outputs = model.predict(tf.expand_dims(img_preprocessed, 0))[0]
    pred_class = classes[np.argmax(outputs)]
    product = get_product(pred_class)

    return (jsonify([product['name'],product['desciption'],product['image']]), 200, headers)