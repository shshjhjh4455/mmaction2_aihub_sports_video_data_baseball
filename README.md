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




