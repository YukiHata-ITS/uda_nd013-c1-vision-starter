# [Submission_report] Object Detection in an Urban Environment

### Project overview
The purpose of this project is that the environment is built and the accuracy and recall of object detection in an urban environment are improved.
The procedure is as the follows.
1. Exploratory Data Analysis: We analysys the datasets and recognize features and trends.
2. Set pipeline: We modify the pipeline config of model, training and eval.
3. Training: We train the model with datasets.
4. Evaluation: We evaluate the model with datasets.
5. Improve model: We repeat steps from 2 to 4 and compare the output
6. Report:we export trained model and create video for model's inferences

### Set up
I used the prepared workspace in UDACITY lesson.
Therefore dataset is already prepared.



Precedure 1: Exploratory Data Analysis

Install Chrome browser
```
sudo apt-get update
sudo apt-get install chromium-browser
sudo chromium-browser --no-sandbox
```
Start Jupyter notebook
```
cd /home/workspace/
jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root
```
Enter address of Jupyter notebook on Chrome.
Open "Exploratory Data Analysis.ipynb" on Jupyter notebook.

Precedure 2:
1. Split dataset to the group of training, evaluation and test.
The ratio of 100 tfrecords in dataset is that training has 70 files, evaluation has 20 files and test has 10 files.
```
python create_splits.py --data-dir /home/workspace/data
```
2. Download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

3. Edit the config file from pretrained model. A new config file `pipeline_new.config` has been created in `home/workspace`.
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
4. Move the file `pipeline_new.config` to `/home/workspace/experiments/reference`

Precedure 3:

Do Training
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```

Precedure 4:

Do Evaluation
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference
```
See in TensoBoard
```
python -m tensorboard.main --logdir experiments/reference/
```

Change checkpoints: Change the first line with parameter name as 'model_checkpoint_path' in `/home/workspace/experiments/reference/checkpoint`.

Precedure 5:

Create a new folder named `experiment0`, `experiment1`, and so on in `/home/workspace/experiments`.  
Change the path of command from `reference` to `experiment0`.
Repeat steps from 2 to 4.


Precedure 6:

Export the trained model
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```
Create video
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

ここまで記入完了

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
