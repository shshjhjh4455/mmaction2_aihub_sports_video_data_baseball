# AI Hub Sports Video Dataset: Baseball

## Overview

This repository hosts the AI Hub's sports video dataset focused on baseball. The dataset comprises labeled video data specifically designed for training and validating AI models in sports analytics, particularly baseball. You can find the original dataset [here](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=488).

## Dataset Description

### Labeled Data Categories

The dataset is organized into multiple categories, each corresponding to specific baseball actions and movements:

- `baseball_ra`: Includes only 'ra' annotations.
- `baseball_ra_ct`: Catcher throwing actions.
- `baseball_ra_ff`: Fielder catching a fly ball.
- `baseball_ra_fg`: Fielder catching a ground ball.
- `baseball_ra_ft`: Fielder throwing actions.
- `baseball_ra_hb`: Hitter bunting.
- `baseball_ra_hh`: Hitter hitting.
- `baseball_ra_hs`: Hitter swinging.
- `baseball_ra_pb`: Pitcher balk movements.
- `baseball_ra_po`: Pitcher overhand throws.
- `baseball_ra_pp`: Pitcher pick-off throws.
- `baseball_ra_ps`: Pitcher side-arm throws.
- `baseball_ra_pu`: Pitcher underhand throws.
- `baseball_ra_rr`: Runner running.

### Data Structure

The dataset includes:

- **Training Set**: 3,888 video clips.
- **Validation Set**: 504 video clips.

Each action is represented by a series of image frames, which are provided in separate folders.

#### Sample Data Distribution
![Training and Validation Data Distribution](https://github.com/shshjhjh4455/aihub_sports_video_data_baseball/assets/44297309/c71045a6-ae82-4d86-8ce9-618df4810f0b)

### Data Columns

The dataset consists of 13 distinct columns, capturing various aspects of each video frame.

#### Dataset Columns Overview
![Dataset Columns](https://github.com/shshjhjh4455/aihub_sports_video_data_baseball/assets/44297309/0c58e641-b220-486a-b4e3-29f6ed5a4104)

## Data Preprocessing

### Video Conversion

Given that the dataset is initially structured as image frames, we provide a script `img2video.py` for converting these frames into continuous video files. This conversion is essential for further processing, including skeleton value extraction using tools like MMAction2.

#### Processed Video Sample
![Processed Video](https://github.com/shshjhjh4455/aihub_sports_video_data_baseball/assets/44297309/2aea2ece-5a6d-44ca-831f-9abebc43040a)

## Usage

Detailed usage instructions for dataset preprocessing, analysis, and model training are forthcoming.

## Skeleton Value Extraction using MMAction2

![MMAction2 Logo](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/14c33060-849b-4565-8b22-ddd84e5bf73f)

### Overview

We utilize the [MMAction2 library](https://github.com/open-mmlab/mmaction2) for extracting skeleton values from each action video. The process involves using the `ntu_pose_extraction.py` script available in MMAction2. This script is specifically designed to extract human poses in a format suitable for action recognition tasks.

### Using `ntu_pose_extraction.py`

The `ntu_pose_extraction.py` script ([source code](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/ntu_pose_extraction.py)) is employed for processing each video file in our dataset. 

#### Usage Example

```bash
python ntu_pose_extraction.py --input video_file.avi --output output.pkl
```

### Data Preprocessing - PKL File Generation

To facilitate the extraction process across all video files in our dataset, we have created a script, `extract_pkl.py`. This script automates the pose extraction for each video and compiles the results into a single PKL file, which can be directly used as input for modeling.

### PKL Format

The PKL files follow the format specified in the MMAction2 documentation, which can be found [here](https://github.com/open-mmlab/mmaction2/tree/main/tools/data/skeleton#the-format-of-annotations).

### Sample PKL File Output

![PKL File Screenshot](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/9061a4bf-bb90-4948-a85a-aff9c6ce7f37)

### Next Steps

Further instructions on how to use the extracted skeleton data for training models using MMAction2 will be provided in subsequent updates.



## PoseC3D for Action Recognition

### Introduction to PoseC3D

PoseC3D is a skeleton-based action recognition model, which can be found in the [MMAction2 model zoo](https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html#posec3d). This model leverages the power of 3D pose estimation to recognize and classify human actions in videos.

#### PoseC3D Architecture
![PoseC3D Architecture](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/18f7e20c-43a1-4300-bbcb-154da8b7c3b9)
![PoseC3D Details](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/a5c96455-6672-411a-ac0b-ac84cda522a4)

### Using PoseC3D with Custom Dataset (AI_hub)

The aim is to train the PoseC3D model using our custom dataset derived from AI_hub. The process involves:

- Utilizing pre-trained human detectors and pose estimators.
- Training the model on the custom dataset to generate labels for video objects.

### Preprocessing Steps for Model Training

To prepare the AI_hub dataset for PoseC3D model training, follow these steps:

1. **Video Conversion**: Since the AI_hub dataset is composed of individual image frames, it is necessary to convert these frames into video format, corresponding to each action.

2. **Data Extraction and Compression**: Extract skeleton and other relevant information from the converted videos. This data is then compressed into a single file.

3. **Model Training**: Use the compressed file to train the PoseC3D model.

### Model Configuration and Training

Modify the existing model configuration file `slowonly_r50_8xb16-u48-240e_gym-keypoint.py` located at `configs/skeleton/posec3d/`. Adjustments include setting the `num_classes` to 13 and specifying the annotation file (`ann_file`) as `data/skeleton/output/custom.pkl`.

By following these steps and configurations, the PoseC3D model can be effectively trained on the AI_hub baseball dataset, enabling accurate action recognition in sports video analysis.


## Training PoseC3D with Custom Dataset

### Executing Training on Colab

To train the PoseC3D model on the custom dataset (AI_hub), follow these steps in Google Colab:

1. **Run the Training Command**:
   Execute the training process using the following command in Colab:

   ```bash
   !python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_customdata_xsub_keypoint.py \
   --work-dir work_dirs/slowonly_r50_u48_240e_customdata_xsub_keypoint \
   --seed 0 \
   --resume
   ```

## PoseC3D Model Training and Evaluation

### Model Specification
- **Model**: PoseC3D
- **Input**: `custom.pkl` (generated from the AI_hub dataset)

### Training Environment
- **Platform**: Google Colab Pro+
- **GPU**: NVIDIA A100
- **Training Duration**: Approximately 6.3 hours

### Performance Evaluation
The model's performance was evaluated using the validation dataset as the test data. The results are as follows:
- **Accuracy (Top-1)**: 0.9636
- **Accuracy (Top-5)**: 0.9838
- **Mean Accuracy**: 0.9717

### Checkpoint and Results
The best performing model checkpoint is saved as `best_acc_top1_epoch_22.pth`.

![Model Checkpoint](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/8d777e74-abdc-4471-9871-e7b2ed56c5bd)

### Label Mapping
The following labels were used for the classification task, as specified in `label_map_custom.txt`:
1. Catcher throw
2. Fielder catch a fly ball
3. Fielder catch a ground ball
4. Fielder fielder throw
5. Hitter bunt
6. Hitter hitting
7. Hitter swing
8. Pitcher balk
9. Pitcher overhand throw
10. Pitcher pick off throw
11. Pitcher side arm throw
12. Pitcher underhand throw
13. Runner run

This section outlines the detailed process of training the PoseC3D model using the custom dataset from AI_hub, including the training environment, duration, performance metrics, and the classification labels used.


## Skeleton-based Action Recognition Demo

This section details the procedure to use the trained PoseC3D model for skeleton-based action recognition in video files. The demo is designed to be run in a Google Colab environment.

### Components
- **Human Detector**: Faster-RCNN
- **Pose Estimator**: HRNetw32
- **Skeleton-based Action Recognizer**: PoseC3D-CUSTOM-keypoint
- **Model Checkpoint**: `best_acc_top1_epoch_22.pth`

### Demo Execution
The following is an example of how to run the demo on a specific video file (`ZX0QNPTZO56M.mp4`). This script will process the video and generate an output video (`demo_output_ZX0QNPTZO56M.mp4`) with recognized actions.

```bash
!python demo/demo_skeleton.py \
    video/mlb/ZX0QNPTZO56M.mp4 \
    demo/demo_output_ZX0QNPTZO56M.mp4 \
    --config configs/skeleton/posec3d/slowonly_r50_u48_240e_customdata_xsub_keypoint.py \
    --checkpoint work_dirs/slowonly_r50_u48_240e_customdata_xsub_keypoint/best_acc_top1_epoch_22.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_custom.txt
```
This script includes the paths to the necessary configuration files, checkpoints for the human detector and pose estimator, and the custom label map for the trained PoseC3D model.


 ## Example Use Cases and Troubleshooting

### Successful Cases
- **Input**: Video of a baseball pitch
- **Output**: Label for the action in the video and a skeleton-annotated video
- **Result**: Skeleton successfully displayed, and the label for the input video correctly identified.
![Successful Skeleton and Label Extraction](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/a4ba0abb-73c9-4a98-ad18-57e26cc53dd1)

- **Test with Baseball Broadcast Video**
  - **Result**: Successful prediction of label for a single person in the video.
![Baseball Broadcast Video Test](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/db0ae673-97a3-4e2d-a008-9e215841ffc2)

### Identified Issues and Solutions

#### Misidentification Issue
- **Problem**: The model misidentified all characters (pitcher, batter, catcher, umpire) in the frame as a batter.
![Misidentification in Action Recognition](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/4c91a586-6108-45e0-93ce-aedae82f4a2b)
- **Cause**: The model, designed to assess action based on a minimum of 15 frames, was only provided with a single frame, leading to incorrect inference.
![Error Due to Single Frame Analysis](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/d70560e9-c6de-4dde-86e1-7f9326fbeceb)

#### Solution
- **Approach**: Divide the video into 48-frame chunks. Distinguish between individuals using cosine similarity, followed by PoseC3D pose estimation.
![Proposed Solution for Misidentification](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/a66ded35-5ea1-4764-8737-574f744a3390)

## Demonstration Code

To run a demonstration of the PoseC3D model on a custom video, use the following command. This script processes a specified video, applying skeleton-based action recognition, and outputs the results in the designated directory.

```bash
!python demo/custom_demo_skeleton.py \
    ./data/cut_video.mp4 \
    --output-dir ./demo/middle \
    --output-file output \
    --config ./configs/skeleton/posec3d/slowonly_r50_u48_240e_customdata_xsub_keypoint.py \
    --checkpoint ./work_dirs/slowonly_r50_u48_240e_customdata_xsub_keypoint/best_acc_top1_epoch_22.pth \
    --output-fps 15
```

### Parameters Explained
- `./data/cut_video.mp4`: Path to the input video file.
- `--output-dir ./demo/middle`: Directory where the output will be saved.
- `--output-file output`: Name of the output file.
- `--config`: Path to the model configuration file.
- `--checkpoint`: Path to the model checkpoint file.
- `--output-fps 15`: Frame rate of the output video.

This command generates a video in the `./demo/middle` directory with recognized actions and corresponding skeletons overlaid on the input video frames.


## Pitcher Recognition and Action Identification

The PoseC3D model has been effectively trained to recognize and classify actions specific to a baseball pitcher. Below are some examples demonstrating the model's ability to identify a pitcher and their specific actions from video frames.

### Pitcher Recognition
The model successfully detects the pitcher in various frames, highlighting the robustness of the pose estimation and action recognition capabilities. 

![Pitcher Recognition 1](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/fd509ebf-e007-4876-a0bb-365bb7be7f39)
![Pitcher Recognition 2](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/88dda875-6189-44f0-ad0d-3703f878506d)
![Pitcher Recognition 3](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/c721b93c-5328-49bf-8f87-877eb2e4976d)

### Action Identification
Once the pitcher is recognized, the model further analyzes the specific actions performed by the pitcher. This feature is crucial for detailed analysis and understanding of the game. The following images showcase the model's capability in identifying distinct actions of a pitcher.

![Pitcher Action 1](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/8e21784a-cd22-4bb7-8e6b-ecd6a41e926a)
![Pitcher Action 2](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/5684663e-90d8-4a58-a359-38746b4b1d18)

These examples demonstrate the model's efficacy in not only detecting the presence of a pitcher but also in accurately classifying their specific movements and actions.



