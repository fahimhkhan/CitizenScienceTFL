# CitizenScienceTFL

This repository is intended for "step 2" to create an object detection app using the citizen science platform from https://sites.google.com/ucsc.edu/csmlappstudio/detection. 

The steps needed to train and use a Tensorflow Lite rip current detection model are below. The repository has the directory structure needed for the training, which can be cloned/downloaded to be used as the starting point.

**1. Dataset creation and annotation**

**Note:** If you are here from https://sites.google.com/ucsc.edu/csmlappstudio/detection, you may already finished your "Dataset creation and annotation". 
Datasets are labeled using a tool called labelImg https://github.com/tzutalin/labelImg. Detailed instruction for can be found here: https://sites.google.com/ucsc.edu/csmlappstudio/label

This repository has no dataset included in it. you need to put your labeled dataset (from step 1) in the "train" and "test" directory under the "dataset" directory. You also need to update the label_map.pbtxt file in the dataset "directory" with the id and name of your classes.

**Note:** If you are trying to train the rip current detection model from this paper (https://doi.org/10.1145/3462204.3481743), link to the annotated dataset for rip current detection is given in this "train" and "test" in a text file named download link.

**2. Setting up Tensorflow and Dependencies**

Install Tensorflow 1.15 using this command:

```pip install tensorflow==1.15```

Install these dependencies:

```
pip install numpy==1.19
pip install Cython 
pip install contextlib2
pip install pillow 
pip install lxml 
pip install jupyter 
pip install matplotlib 
pip install tf_slim 
pip install pycocotools
pip install scipy
```

Install Protobuf appropriate for your operationg system:
https://grpc.io/docs/protoc-installation/

Clone the object detection models repository

```git clone https://github.com/tensorflow/models.git```

Compile Protobuf from "models/research" folder

```protoc object_detection/protos/*.proto --python_out=.```

In the \models\research\slim directory run

```
python setup.py build
python setup.py install
```

Update PYTHONPATH variable:

```export PYTHONPATH=$PYTHONPATH:"/research":"/research/slim"```
or
```export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim```

Install object detection. From within TensorFlow/models/research/

```
cp object_detection/packages/tf1/setup.py .
python -m pip install .
```

Test the installation of object detection API by runnning the command below from "models/research" folder

```python object_detection/builders/model_builder_tf1_test.py```

**3. Generating TFRecord**

First, run the following commands in the dataset directory to generate "test.csv" and "train.csv" respectively. 

```python xml_to_csv_test.py```

```python xml_to_csv_train.py```


Then, generate the TFRecord files by running the following python script inside the dataset directory,

```python generate_tfrecord.py --csv_input=train.csv  --output_path=train.record --image_dir=train/images```

```python generate_tfrecord.py --csv_input=test.csv  --output_path=test.record --image_dir=test/images```

**4. Selecting a Pre-trained model**

We used a pretrained model as our initial checkpoint ssd mobilenet v2. It can be downloaded from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

Extract the .tar.gz file to the folder named "pretrained_model". (It should be pretrained_model/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03)

Update the "pipeline.config" in inside the extracted folder as follows,

*update the number of classes in line 3
*update the paths of the pretrained model in line 157
*update the path of the dataset in line 162, 164, 174, 178

![alt text](pipeline_config.png?raw=true)

(Tip: you can use an editor such as Atom, VS Code, etc to see the line numbers.)

**5. Training the model**

Train the model using the pretrained model as our initial checkpoint. Use the trained_model directory as the training folder. Run the following script inside "models/research/object_detection/legacy/" directory.

```python train.py --logtostderr --train_dir=<path to "trained_model"> --pipeline_config_path=<path to pipeline.config file>```

Run the training until converge and then run the checkpoint for the next step. It is considered by many literature that the model converged when the loss is below 2.

**6. Converting the model to .tflite**

Convert the TensorFlow model and generates a TensorFlow Lite model using the instruction from the link below
https://www.tensorflow.org/lite/convert

For example, if you are using the 5000th checkpoint, run the follwing command from "models/research/object_detection/" directory.

```python export_tflite_ssd_graph.py --pipeline_config_path=<path to pipeline.config file> --trained_checkpoint_prefix="../trained_model/model.ckpt-5000" --output_directory="../trained_model/tflite" --add_postprocessing_op=true```

run the follwing command from "trained_model" directory

```
tflite_convert \
--graph_def_file=tflite/tflite_graph.pb \
--output_file=tflite/model.tflite \
--output_format=TFLITE \
--input_shapes=1,300,300,3 \
--input_array=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_dev_values=127 \
--change_concat_input_ranges=false \
--allow_custom_ops
```

This command will create a file named "model.tflite" in a directory named "tflite". Here, create a text file with your class names in seperate lines and name it "labelmap.txt". Now, run the following command to include the labelmap as metadata with your .tflite file,

```python metadata_writer.py```

It'll create a file named "detect.tflite"

**7. Building and running the Android/iOS app with the .tflite.**

Add an additional line "???" to your "labelmap.txt". Now, use your "detect.tflite" and "labelmap.txt" with your Android/iOS app development project in Android Studio or Xcode to build and run the app following the instructions from "Step 3" here, 

https://sites.google.com/ucsc.edu/csmlappstudio/detection
