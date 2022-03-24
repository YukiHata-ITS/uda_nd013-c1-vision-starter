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

#### reference
- folder: [reference](experiments\reference)  
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](experiments\reference\pipeline_new.config)    

Result
![r_Loss](00_report_data\reference\Loss.PNG)
![r_Precision](00_report_data\reference\DetectionBoxes_Precision.PNG)
![r_Recall](00_report_data\reference\DetectionBoxes_Recall.PNG)

Eval metrics at step 2500
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.003
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.008
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.005
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.016
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.008
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.140
DetectionBoxes_Precision/mAP: 0.000780
DetectionBoxes_Precision/mAP@.50IOU: 0.002309
DetectionBoxes_Precision/mAP@.75IOU: 0.000541
DetectionBoxes_Precision/mAP (small): 0.000082
DetectionBoxes_Precision/mAP (medium): 0.003173
DetectionBoxes_Precision/mAP (large): 0.008268
DetectionBoxes_Recall/AR@1: 0.001590
DetectionBoxes_Recall/AR@10: 0.004735
DetectionBoxes_Recall/AR@100: 0.015630
DetectionBoxes_Recall/AR@100 (small): 0.008226
DetectionBoxes_Recall/AR@100 (medium): 0.012240
DetectionBoxes_Recall/AR@100 (large): 0.139973
Loss/localization_loss: 0.847028
Loss/classification_loss: 0.746803
Loss/regularization_loss: 1.198902
Loss/total_loss: 2.792733
```

I compared this reference model and the following experiment0 model. The result is written the below.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

#### experiment0
- folder: [experiment0](experiments\experiment0)  
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](experiments\experiment0\pipeline_new.config)  
  I add 3 data_augmentation_options on the reference pipeline.  
  1. random_adjust_brightness
  2. random_adjust_contrast
  3. random_distort_color


Result
![r_Loss](00_report_data\experiment0\Loss.PNG)
![r_Precision](00_report_data\experiment0\DetectionBoxes_Precision.PNG)
![r_Recall](00_report_data\experiment0\DetectionBoxes_Recall.PNG)

Eval metrics at step 2500
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.005
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.070
DetectionBoxes_Precision/mAP: 0.000038
DetectionBoxes_Precision/mAP@.50IOU: 0.000170
DetectionBoxes_Precision/mAP@.75IOU: 0.000006
DetectionBoxes_Precision/mAP (small): 0.000000
DetectionBoxes_Precision/mAP (medium): 0.000035
DetectionBoxes_Precision/mAP (large): 0.000710
DetectionBoxes_Recall/AR@1: 0.000048
DetectionBoxes_Recall/AR@10: 0.000795
DetectionBoxes_Recall/AR@100: 0.005331
DetectionBoxes_Recall/AR@100 (small): 0.000000
DetectionBoxes_Recall/AR@100 (medium): 0.009718
DetectionBoxes_Recall/AR@100 (large): 0.069919
Loss/localization_loss: 1.089438
Loss/classification_loss: 0.959590
Loss/regularization_loss: 520.413696
Loss/total_loss: 522.462769
```

#### experiment3
- folder: [experiment3](experiments\experiment3)  
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](experiments\experiment3\pipeline_new.config)  
  I add 2 data_augmentation_options on the reference pipeline.  
  1. random_adjust_brightness
  2. random_adjust_contrast

Result
![r_Loss](00_report_data\experiment3\Loss.PNG)
![r_Precision](00_report_data\experiment3\DetectionBoxes_Precision.PNG)
![r_Recall](00_report_data\experiment3\DetectionBoxes_Recall.PNG)

Eval metrics at step 2500
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
DetectionBoxes_Precision/mAP: 0.000000
DetectionBoxes_Precision/mAP@.50IOU: 0.000000
DetectionBoxes_Precision/mAP@.75IOU: 0.000000
DetectionBoxes_Precision/mAP (small): 0.000000
DetectionBoxes_Precision/mAP (medium): 0.000000
DetectionBoxes_Precision/mAP (large): 0.000000
DetectionBoxes_Recall/AR@1: 0.000000
DetectionBoxes_Recall/AR@10: 0.000000
DetectionBoxes_Recall/AR@100: 0.000000
DetectionBoxes_Recall/AR@100 (small): 0.000000
DetectionBoxes_Recall/AR@100 (medium): 0.000000
DetectionBoxes_Recall/AR@100 (large): 0.000000
Loss/localization_loss: 1.287755
Loss/classification_loss: 1.130766
Loss/regularization_loss: 0.244385
Loss/total_loss: 2.662905
```

Comparing them, the performance of experiment0 and experiment3 pipeline is lower than the one of reference pipeline. The reference pipeline cannnot detect objects correctly. Therefore with more complex dataset I use, the performance decrease.
I have to improve overall performance.

#### experiment4
- folder: [experiment4](experiments\experiment4)  
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](experiments\experiment4\pipeline_new.config)  
  I change batch_size from 2 to 4 on the reference pipeline.

Result
![r_Loss](00_report_data\experiment4\Loss.PNG)
![r_Precision](00_report_data\experiment4\DetectionBoxes_Precision.PNG)
![r_Recall](00_report_data\experiment4\DetectionBoxes_Recall.PNG)

Eval metrics at step 2500
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.026
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.108
DetectionBoxes_Precision/mAP: 0.000191
DetectionBoxes_Precision/mAP@.50IOU: 0.000771
DetectionBoxes_Precision/mAP@.75IOU: 0.000074
DetectionBoxes_Precision/mAP (small): 0.000068
DetectionBoxes_Precision/mAP (medium): 0.000247
DetectionBoxes_Precision/mAP (large): 0.001641
DetectionBoxes_Recall/AR@1: 0.000425
DetectionBoxes_Recall/AR@10: 0.001967
DetectionBoxes_Recall/AR@100: 0.010292
DetectionBoxes_Recall/AR@100 (small): 0.000125
DetectionBoxes_Recall/AR@100 (medium): 0.025631
DetectionBoxes_Recall/AR@100 (large): 0.108266
Loss/localization_loss: 0.857064
Loss/classification_loss: 0.778067
Loss/regularization_loss: 28.522840
Loss/total_loss: 30.157974
```

The batch size change alone had little effect for performance. I will change the dataset size next.

#### experiment5
- folder: [experiment5](experiments\experiment5)  
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](experiments\experiment5\pipeline_new.config)  
  I change the following steps of training_config.  
  total_steps: 12500  
  warmup_steps: 1000  
  num_steps: 12500

Result
![r_Loss](00_report_data\experiment5\Loss.PNG)
![r_Precision](00_report_data\experiment5\DetectionBoxes_Precision.PNG)
![r_Recall](00_report_data\experiment5\DetectionBoxes_Recall.PNG)

This training output is too large.  
I have reached the storage limit of 3GB.  

Eval metrics at step 6000
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.041
DetectionBoxes_Precision/mAP: 0.000009
DetectionBoxes_Precision/mAP@.50IOU: 0.000046
DetectionBoxes_Precision/mAP@.75IOU: 0.000001
DetectionBoxes_Precision/mAP (small): 0.000000
DetectionBoxes_Precision/mAP (medium): 0.001980
DetectionBoxes_Precision/mAP (large): 0.000099
DetectionBoxes_Recall/AR@1: 0.000000
DetectionBoxes_Recall/AR@10: 0.000007
DetectionBoxes_Recall/AR@100: 0.002234
DetectionBoxes_Recall/AR@100 (small): 0.000000
DetectionBoxes_Recall/AR@100 (medium): 0.000890
DetectionBoxes_Recall/AR@100 (large): 0.040921
Loss/localization_loss: 1.112553
Loss/classification_loss: 4996.385742
Loss/regularization_loss: 1026282356736.000000
Loss/total_loss: 1026282356736.000000
```
The performance has not be inproved.

