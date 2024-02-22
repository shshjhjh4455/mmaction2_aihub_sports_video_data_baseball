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



PoseC3D https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html#posec3d
![142995620-21b5536c-8cda-48cd-9cb9-50b70cab7a89](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/18f7e20c-43a1-4300-bbcb-154da8b7c3b9)
![116531676-04cd4900-a912-11eb-8db4-a93343bedd01](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/a5c96455-6672-411a-ac0b-ac84cda522a4)

스켈레톤 기반 행동인식 poseC3D 모델 사용 방안
커스텀 데이터셋(AI_hub)으로 poseC3D 모델을 학습
pre trained된 Human detector와 pose estimator를 사용
학습된 모델로 영상 객체에 라벨 생성

커스텀 데이터셋으로 모델 학습을 위한 전처리 단계
다운로드한 AI_hub 데이터셋은 동영상포맷이 아닌 이미지 프레임별로 나누어져 있어서, 행동별로 나누어진 모든 폴더의 이미지를 각각의 동영상으로 변환해야한다.
변환한 동영상으로부터 스켈레톤을 포함한 여러 정보를 추출하여 하나의 파일로 압축한다.
압축한 파일을 사용해서 PoseC3D 모델을 학습 시킨다.


기존에 모델 코드 slowonly_r50_8xb16-u48-240e_gym-keypoint.py  수정하여 사용
경로 configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py
num_classes=13, ann_file = 'data/skeleton/output/custom.pkl'

커스텀 데이터셋으로 모델 학습
코랩에서 실행
!python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_customdata_xsub_keypoint.py --work-dir work_dirs/slowonly_r50_u48_240e_customdata_xsub_keypoint --seed 0 --resume

모델 : PoseC3D
인풋 : custom.pkl (AI_hub 데이터셋으로 생성한 파일)

학습 환경 : Colab pro+ ,GPU : A100
학습 소요시간 : 6.3시간

성능평가 : test 데이터는 validation 데이터 사용
acc/top1: 0.9636 acc/top5: 0.9838 acc/mean1: 0.9717

checkpoint : best_acc_top1_epoch_22.pth
![Screenshot 2024-02-22 at 2 27 49 PM](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/8d777e74-abdc-4471-9871-e7b2ed56c5bd)

label_map_custom.txt
catcher throw
fielder catch a fly ball
fielder catch a ground ball
fielder filder throw
hitter bunt
hitter hitting
hitter swing
pitcher bulk
pitcher overhand throw
pitcher pick off throw
pitcher side arm throw
pitcher underhand throw
runner run

학습된 모델로 영상 객체에 라벨 생성 
코랩에서 실행

Skeleton-based Action Recognition Demo
human detector : Faster-RCNN
pose estimator : HRNetw32
skeleton-based action recognizer : PoseC3D-CUSTOM-keypoint
checkpoint : best_acc_top1_epoch_22.pth

# ZX0QNPTZO56M.mp4
# Skeleton-based Action Recognition Demo

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

   *예시* 
입력 : 야구 공 던지는 
동영상 / 출력 : 해당 영상에 해당하는 레이블 값과 스켈레톤 표시 동영상
![Screenshot 2024-02-22 at 2 36 46 PM](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/a4ba0abb-73c9-4a98-ad18-57e26cc53dd1)
스켈레톤이 정상적으로 표시됨
입력 영상에 해당하는 레이블 값 추출 성공

야구 중계 영상 테스트![Screenshot 2024-02-22 at 2 37 28 PM](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/db0ae673-97a3-4e2d-a008-9e215841ffc2)
사람 1명에 대한 레이블 예측 성공

문제점![Screenshot 2024-02-22 at 2 38 05 PM](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/4c91a586-6108-45e0-93ce-aedae82f4a2b)
투수, 타자, 포수, 심판
모두를 타자라고 오인식

오인식 이유 파악
최소 15개의 프레임을 보고 행동을 판단하는 모델에게 1개의 프레임만을 보여줌으로써
행동을 추론하는 과정에서 오류 발생
![Screenshot 2024-02-22 at 2 38 50 PM](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/d70560e9-c6de-4dde-86e1-7f9326fbeceb)

문제점 해결방안![Screenshot 2024-02-22 at 2 39 27 PM](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/a66ded35-5ea1-4764-8737-574f744a3390)
영상 프레임을 48프레임 청크로 나눈다.

인식된 사람들을 코사인 유사도를 통해 각각의 사람으로 구분하고 posec3d 포즈추정 과정을 진행한다.
![Screenshot 2024-02-22 at 2 39 53 PM](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/00dfd2f7-d379-4b4e-97d8-952ed2190134)

데모 코드
!python demo/custom_demo_skeleton.py ./data/cut_video.mp4 --output-dir ./demo/middle --output-file output --config ./configs/skeleton/posec3d/slowonly_r50_u48_240e_customdata_xsub_keypoint.py --checkpoint ./work_dirs/slowonly_r50_u48_240e_customdata_xsub_keypoint/best_acc_top1_epoch_22.pth --output-fps 15

투수 인식
![image](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/fd509ebf-e007-4876-a0bb-365bb7be7f39)
![image](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/88dda875-6189-44f0-ad0d-3703f878506d)
![image](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/c721b93c-5328-49bf-8f87-877eb2e4976d)
투수의 행동인식
![image](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/8e21784a-cd22-4bb7-8e6b-ecd6a41e926a)
![image](https://github.com/shshjhjh4455/mmaction2_aihub_sports_video_data_baseball/assets/44297309/5684663e-90d8-4a58-a359-38746b4b1d18)






