import os
from flask import Blueprint, render_template, request, send_file, jsonify

from models import product_catalog
from middlewares.auth import auth_optional

import base64
import json
import tensorflow as tf
import numpy as np
import re
import glob
import pathlib

import cv2

from matplotlib import pyplot as plt
import uuid

from google.cloud import storage

import tensorflow as tf
import numpy as np
import uuid
import cv2
from flask import jsonify

from google.cloud import storage
from google.cloud import firestore

model = None
BUCKET = os.environ.get('GCS_BUCKET')
storage_client = storage.Client()
firestore_client = firestore.Client()

FILENAME_TEMPLATE = '{}.jpg'

uploader_page = Blueprint('uploader_page', __name__)

# dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = "static/images"
STATIC_FOLDER = "static"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'static/models/frozen_inference_graph1.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'static/models/data/object-detection.pbtxt'

# Load model
# cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "dog_cat_M.h5")
# classes = ['cat', 'dog']


classes = ['TqE3sw0EuLVO6xHPBlog','ibrfh344pxsOkbTf6pLf','Jhuge8GdZ4geGXgnm7wr','jCKM0uQFyKeF8gieQPti','Hk8OLqr0qwTsaknX4phW']
category_index = {1: {'id': 1, 'name': 'modernhustle'},
                 2: {'id': 2, 'name': 'koi'},
                 3: {'id': 3, 'name': 'alley'},
                 4: {'id': 4, 'name': 'starbucks'},
                 5: {'id': 5, 'name': 'phuclong'}}

# IMAGE_SIZE = 192
# IMAGE_SIZE = (12, 8)

# # Preprocess an image
# def preprocess_image(image):
#     image_reader = tf.image.decode_jpeg(image, channels=3)
#     float_caster = tf.cast(image_reader, tf.float32)
#     dims_expander = tf.expand_dims(float_caster, 0)
#     image = tf.image.resize_images(dims_expander, [12, 8])
#     # image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
#     image = image / 255.0  # normalize to [0,1] range

#     return image

def upload_blob(bucket_name, src_file, dst_file_name):
    """Upload a file to the bucket"""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('fansipan-website-290191')
    blob = bucket.blob('uploaded/'+dst_file_name)
    blob.upload_from_string(src_file, content_type='image/jpg')
    print('File uploaded to uploaded/{}.'.format(dst_file_name))

def preprocess_image(image_raw):
    image = cv2.imdecode(np.fromstring(image_raw, dtype='uint8'), cv2.COLOR_BGR2RGB)
    return np.array([image]), image.shape[:2]

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            # if 'detection_masks' in tensor_dict:
            #     # The following processing is only for single image
            #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            #     detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            #     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            #     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            #         detection_masks, detection_boxes, image.shape[1], image.shape[2])
            #     detection_masks_reframed = tf.cast(
            #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            #     # Follow the convention by adding back the batch dimension
            #     tensor_dict['detection_masks'] = tf.expand_dims(
            #         detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            print('ok')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
              output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# # Read the image from path and preprocess
# def load_and_preprocess_image(path):
#     image = tf.io.read_file(path)

#     return preprocess_image(image)


# # Predict & classify image
# def classify(model, image_path):

#     preprocessed_imgage = load_and_preprocess_image(image_path)
#     preprocessed_imgage = tf.reshape(
#         preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
#     )

#     prob = cnn_model.predict(preprocessed_imgage)
#     label = "Cat" if prob[0][0] >= 0.5 else "Dog"
#     classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]

#     return label, classified_prob

def get_product(product_id):
    product = firestore_client.collection('stores').document(product_id).get()
    return product.to_dict()

def list_products(name_of_product):
    products = firestore_client.collection(name_of_product).get()
    return products


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

@uploader_page.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    # if request.method == 'POST':
    #    f = request.files['image']
    #    f.save("images/" + f.filename)
    #    # myImage = [image for image in ['stores/' + im for im in os.listdir('stores')]]
    #    label = [im.split('.')[0] for im in os.listdir('static/stores')]
    #    print(label)
    # return render_template("uploader.html", label=label)

    # products = product_catalog.get_product('jCKM0uQFyKeF8gieQPti')
    f = request.files.get('image')
    raw_img = f.read()
    img_preprocessed, (rows,cols) = preprocess_image(raw_img)
    # Actual detection.
    output_dict = run_inference_for_single_image(img_preprocessed, detection_graph)
    output = output_dict['detection_classes'][0]
    store_name = category_index[output]['name']

    x = output_dict['detection_boxes'][0][1] * cols
    y = output_dict['detection_boxes'][0][0] * rows
    right = output_dict['detection_boxes'][0][3] * cols
    bottom = output_dict['detection_boxes'][0][2] * rows
    # x,y,w,h = output_dict['detection_boxes'][0]
    # x,w = int(x*height), int(w*height)
    # y,h = int(y*weight), int(h*weight)
    # print(height, weight)
    image = cv2.rectangle(img_preprocessed[0], (int(x),int(y)), (int(right),int(bottom)), (0,255,0), 5)
    print('hey', category_index[output_dict['detection_classes'][0]]['name'])
    image = cv2.putText(image, category_index[output_dict['detection_classes'][0]]['name'], (int(y), int(x-8)), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0))
    cv2.imwrite('static/images/image.jpg', image)
    link='static/images/image.jpg'

    pred_class = classes[output-1]
    product = get_product(pred_class)
    
    menus = list_products(store_name)
    menu = []
    for item in list(menus):
        prod=item.to_dict()
        prod['id']=store_name + "@" + item.id
        menu.append(prod)

    # menu= [prod.to_dict() for prod in list(menus)]

    img_encode = cv2.imencode('.jpg', image)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()

    # upload file to storage
    id = uuid.uuid4().hex
    filename = FILENAME_TEMPLATE.format(id)
    upload_blob(BUCKET, str_encode, filename)

#    outputs = cnn_model.predict(tf.expand_dims(img_preprocessed, 0))[0]
#    pred_class = classes[np.argmax(outputs)]

   # f.save("images/" + f.filename)



    return render_template('uploader.html',
                    name=product['name'],
                    image=product['image'],
                    description=product['description'],
                    averageprice=str(product['average_price']),
                    products=menu,
                    image_menu='uploaded/'+filename)    

    # return jsonify([product['name'],product['description'],product['image'],"uploaded/"+filename,str(product['average_price']),menu])   

