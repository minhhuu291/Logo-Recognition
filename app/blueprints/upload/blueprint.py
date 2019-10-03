import os
import uuid

from flask import Blueprint, request
from middlewares.auth import auth_required

from google.cloud import storage

client = storage.Client()

BUCKET = os.environ.get('GCS_BUCKET')
FILENAME_TEMPLATE = '{}.png'

upload_api = Blueprint('upload_api', __name__)

@upload_api.route('/upload', methods=['POST'])
@auth_required
def upload_image(auth_context):
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

    file = request.files.get('filepond')
    if not file:
        return ("File is not found in the request.", 400, headers)

    image = file.read()

    id = uuid.uuid4().hex
    filename = FILENAME_TEMPLATE.format(id)

    bucket = client.get_bucket(BUCKET)
    blob = bucket.blob(filename)

    blob.upload_from_string(image, content_type='image/png')

    return (id, 200, headers)