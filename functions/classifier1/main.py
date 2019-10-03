import tensorflow as tf
import numpy as np
import uuid
from flask import jsonify

from google.cloud import storage
from google.cloud import firestore

import os

model = None
BUCKET = os.environ.get('GCS_BUCKET')
storage_client = storage.Client()
firestore_client = firestore.Client()

category_index = {1: {'id': 1, 'name': 'starbucks'},
                 2: {'id': 2, 'name': 'coffeehouse'}}
classes = {1: 'jCKM0uQFyKeF8gieQPti', 2: 'tqffouSurLWDrCJdkyox'}

FILENAME_TEMPLATE = '{}.jpg'

if not os.path.exists('/tmp/model'):
    os.makedirs('/tmp/model')
    
def preprocess_image(image_raw):
    image = cv2.imdecode(np.fromstring(image_raw, dtype='uint8'), cv2.COLOR_BGR2RGB)
    return np.array([image]), image.shape[:2]


def download_blob(bucket_name, src_blob_name, dst_file_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(src_blob_name)

    blob.download_to_filename(dst_file_name)

    print('Blob {} downloaded to {}.'.format(
        src_blob_name,
        dst_file_name))

def upload_blob(bucket_name, src_file, dst_file_name):
    """Upload a file to the bucket"""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('fansipan-website-290191')
    blob = bucket.blob('uploaded/'+dst_file_name)
    blob.upload_from_string(src_file, content_type='image/jpg')
    print('File uploaded to uploaded/{}.'.format(dst_file_name))


def load_model():    
    global model
    if not os.path.exists('/tmp/model/frozen_inference_graph.pb'):
        download_blob(BUCKET, 'frozen_inference_graph.pb', '/tmp/model/frozen_inference_graph.pb')

    path = '/tmp/model/frozen_inference_graph.pb'
    
    model = tf.Graph()
    with model.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


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
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

def get_product(product_id):
    product = firestore_client.collection('stores').document(product_id).get()
    return product.to_dict()

def classifier1(request):
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

    f = request.files.get('image')
    raw_img = f.read()
    img_preprocessed, (height, weight) = preprocess_image(raw_img)
    # Actual detection.
    output_dict = run_inference_for_single_image(img_preprocessed, model)
    output = output_dict['detection_classes'][0]
    store_name = category_index[output_dict['detection_classes'][0]]['name']

    x,y,w,h = output_dict['detection_boxes'][0]
    x,w = int(x*height), int(w*height)
    y,h = int(y*weight), int(h*weight)
    image = cv2.rectangle(img_preprocessed[0], (y,x), (h,w), (0,255,0), 5)
    image = cv2.putText(image, store_name, (y, x-8), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0))
	
    # Encode image to base64
    img_encode = cv2.imencode('.jpg', image)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()

    # upload file to storage
    id = uuid.uuid4().hex
    filename = FILENAME_TEMPLATE.format(id)
    upload_blob(BUCKET, str_encode, filename)

    pred_class = classes[output]
    product = get_product(pred_class)

    return (jsonify([product['name'],product['description'],product['image'],"uploaded/"+filename]), 200, headers)