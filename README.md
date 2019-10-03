# Vietnam Dong Classifier

In this project, I implement a classifier for the currency of Vietnam, the Vietnam Dong (VND). The idea is a website that user can capture an image of VND note which could be 1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k, and 500k (9 classes), and then the app will recognize the value of the note.

Demo app: https://cs-currency-classifier.appspot.com/#

The technical problems to solve:
* How to train a deep learning model to classify the note?
* How to deploy the model in a web application?
* How can a user capture photos on the web?
* How to store the images that users submit?

## Modeling

The model was trained on 1675 images:

| Note value | #image |
|-|-|
| 1000 | 68|
| 10000 | 164|
| 100000 | 200|
| 2000 | 176|
| 20000 | 185|
| 200000 | 203|
| 5000 | 253|
| 50000 | 219|
| 500000 | 207|

Pre-trained model: `tf.keras.applications.MobileNetV2`

Fine-tuning at layer 140

Model summary:
```
Model: "sequential_18"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_192 (Model) (None, 6, 6, 1280)        2257984   
_________________________________________________________________
global_average_pooling2d_18  (None, 1280)              0         
_________________________________________________________________
dense_18 (Dense)             (None, 9)                 11529     
=================================================================
Total params: 2,269,513
Trainable params: 1,051,529
Non-trainable params: 1,217,984
```

Validation accuracy: 88.54%

## Deployment

This demo app was deployed on Google Cloud Platform, using
[Google App Engine](https://cloud.google.com/appengine/),
[Google Cloud Functions](https://cloud.google.com/functions/),
[Google Cloud Storage](https://cloud.google.com/storage/)

**Before you begin:** Create a Google Cloud Platform project with billing enabled and install Google Cloud SDK.
Set up project:

```
gcloud auth login
gcloud config set project PROJECT_ID
```

**Deploy to App Engine:**
Open a terminal and switch to the `app/` directory. Edit `app/app.yaml` and replace `YOUR-PROJECT`,
`YOUR-BUCKET`, with values of your own. `app.yaml`
is the App Engine configuration file. Then run the command below

```
gcloud app deploy
```

**Deploy Cloud Function:**
Open a terminal and switch to directory `functions/classifier`:

```
gcloud functions deploy classifier2 \
    --runtime python37 \
    --trigger-http \
    --set-env-vars GCS_BUCKET=fansipan-website-290191 \
    --region us-central1 \
    --memory 2048
```
