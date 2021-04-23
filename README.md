# CitizenScienceTFL
Training Tensorflow Lite models for ML powered Citizen Science mobile apps

The steps Needed to train and use a Tensorflow Lite Object detection model are as below

**1. Dataset creation and annotation**

The annotated dataset for rip current detection is given in this repo. There is two folders, "train" and "test", respectively.
Datasets are labeled using a tool called labelImg https://github.com/tzutalin/labelImg

**2. Setting up Tensorflow and Dependencies**

Install Tensorflow 1.15 using this command:

```pip install tensorflow==1.15```

Install these dependencies:

```
pip install Cython 
pip install contextlib2
pip install pillow 
pip install lxml 
pip install jupyter 
pip install matplotlib 
pip install tf_slim 
pip install pycocotools
```

Install Protobuf:
https://grpc.io/docs/protoc-installation/

Clone the object detection models repository

```git clone https://github.com/tensorflow/models.git```

Compile Protobuf

```protoc object_detection/protos/*.proto --python_out=.```

Update PYTHONPATH variable:

```export PYTHONPATH=$PYTHONPATH:"/research":"/research/slim"```

Test the installation of object detection API by runnning the command below from models/researchfolder

```python object_detection/builders/model_builder_tf1_test.py```

**3. Generating TFRecord**

Generat the TFRecord files

**4. Selecting a Pre-trained model**

We used a pretrained model as our initial checkpoint ssd mobilenet v2. It can be downloaded from https://github.com/practical-learning/object-detection-on-android/releases/download/v1.0/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

Extract the .tar.gz file to a folder named pretrained_model.

**5. Training the model**

Train the model using the pretrained model as our initial checkpoint.

**6. Converting the model to .tflite**

Convert the TensorFlow model and generates a TensorFlow Lite model using the instruction from the link below
https://www.tensorflow.org/lite/convert

**7. Building and running the Android/iOS app with the .tflite.**

Include the .tflite file to your Android/iOS app development project in Android Studio or Xcode, then build and run the app.
