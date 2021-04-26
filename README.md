# CitizenScienceTFL

The steps Needed to train and use a Tensorflow Lite rip current detection model are as below.

**1. Dataset creation and annotation**

Link to the annotated dataset for rip current detection is given in this repo. There is two folders, "train" and "test", respectively.
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

Test the installation of object detection API by runnning the command below from models/research folder

```python object_detection/builders/model_builder_tf1_test.py```

**3. Generating TFRecord**

First, run "the xml_to_csv_test.py" and "xml_to_csv_train.py" files in the dataset directory to generate "test.csv" and "train.csv" respectively. Then, generate the TFRecord files by running the following python script inside the dataset directory,

```python generate_tfrecord.py --csv_input=train.csv  --output_path=train.record --image_dir=train/images

python generate_tfrecord.py --csv_input=test.csv  --output_path=test.record --image_dir=test/images```

**4. Selecting a Pre-trained model**

We used a pretrained model as our initial checkpoint ssd mobilenet v2. It can be downloaded from https://github.com/practical-learning/object-detection-on-android/releases/download/v1.0/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

Extract the .tar.gz file to the folder named "pretrained_model".

Copy and replace with the given "pipeline.config" in "pretrained_model" to the extracted folder. Inside the extracted folder, update the paths of the pretrained model in line 157 and update the path of the dataset in line 162, 164, 174, 178 in the "pipeline.config" file.

**5. Training the model**

Train the model using the pretrained model as our initial checkpoint. Use the trained_model directory as the training folder. Run the following script inside "models/research/object_detection/legacy/" directory.

```python train.py --logtostderr --train_dir=<path to "trained_model"> --pipeline_config_path=<path to pipeline.config file>```

Run the training until converge and then run the checkpoint for the next step.

**6. Converting the model to .tflite**

Convert the TensorFlow model and generates a TensorFlow Lite model using the instruction from the link below
https://www.tensorflow.org/lite/convert

For example, if you are using the 5000th checkpoint, run the follwing command from "models/research/object_detection/" directory.

```python export_tflite_ssd_graph.py --pipeline_config_path=<path to pipeline.config file> --trained_checkpoint_prefix="../trained_model/model.ckpt-5000" --output_directory="../trained_model/tflite" --add_postprocessing_op=true```

**7. Building and running the Android/iOS app with the .tflite.**

Include the .tflite file to your Android/iOS app development project in Android Studio or Xcode, then build and run the app.
