# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile

import os
import cv2
import mmcv
import mmengine
import torch
import numpy as np

from scipy.spatial.distance import cosine
from tqdm import tqdm
from mmengine import DictAction
from mmengine.utils import track_iter_progress

from mmaction.apis import (detection_inference, inference_skeleton,
                           init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', help='video file/url')
    # parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='the category id for human detection')
    parser.add_argument(
        '--pose-config',
        default='demo/demo_configs/'
        'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_custom.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--output-fps',
        type=float,
        default=15.0,
        help='fps')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./',
        help='output directory')
    parser.add_argument(
        '--output-file',
        type=str,
        help='output file')
    args = parser.parse_args()
    return args

def get_middle_values(lst, num):
    middle_values = []
    for i in range(0, len(lst), num):
        sublist = lst[i:i+5]
        middle_index = len(sublist) // 2
        middle_values.append(sublist[middle_index])
    return middle_values

def visualize(args, frames, data_samples, action_label, frame_num, people_num):
    pose_config = mmengine.Config.fromfile(args.pose_config)
    visualizer = VISUALIZERS.build(pose_config.visualizer)
    visualizer.set_dataset_meta(data_samples[0].dataset_meta)

    vis_frames = []
    print('Drawing skeleton for each frame')
    for d, f in track_iter_progress(list(zip(data_samples, frames))):
        f = mmcv.imconvert(f, 'bgr', 'rgb')
        visualizer.add_datasample(
            'result',
            f,
            data_sample=d,
            draw_gt=False,
            draw_heatmap=True,
            draw_bbox=True,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=0.3)
            
        vis_frame = visualizer.get_image()
        # cv2.putText(vis_frame, action_label, (10, 30), FONTFACE, FONTSCALE,
        #             FONTCOLOR, THICKNESS, LINETYPE)
        for i, (pred_instance, keypoint_scores) in enumerate(zip(d.pred_instances.bboxes, d.pred_instances.keypoint_scores)):
          bbox = pred_instance  # 각 사람의 바운딩 박스 정

          # 바운딩 박스 그리기
          cv2.rectangle(vis_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

          # 행동 레이블 텍스트 추가
          cv2.putText(vis_frame, action_label, (int(bbox[0]), int(bbox[1]) - 10), FONTFACE, FONTSCALE,
                      FONTCOLOR, THICKNESS, LINETYPE)
        vis_frames.append(vis_frame)
    if len(vis_frames) >= 15:
      vid = mpy.ImageSequenceClip(vis_frames, fps=len(vis_frames)//4)
      
      if 'pitcher' in action_label.split(":")[0]:
        if 'bulk' in action_label.split(":")[0]:
          if not os.path.exists(f"{args.output_dir}/pitcher/bulk"):
            os.makedirs(f"{args.output_dir}/pitcher/bulk")
            output_file = f"{args.output_dir}/pitcher/bulk/pitcher_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
        
          else:
            output_file = f"{args.output_dir}/pitcher/bulk/pitcher_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
        
        elif 'throw' in action_label.split(":")[0]:
          if not os.path.exists(f"{args.output_dir}/pitcher/throw"):
            os.makedirs(f"{args.output_dir}/pitcher/throw")
            output_file = f"{args.output_dir}/pitcher/thorw/pitcher_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
        
          else:
            output_file = f"{args.output_dir}/pitcher/throw/pitcher_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
      
      elif 'hitter' in action_label.split(":")[0]:
        if 'bunt' in action_label.split(":")[0]:
          if not os.path.exists(f"{args.output_dir}/hitter/bunt"):
            os.makedirs(f"{args.output_dir}/hitter/bunt")
            output_file = f"{args.output_dir}/hitter/bunt/hitter_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
        
          else:
            output_file = f"{args.output_dir}/hitter/bunt/hitter_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)

        elif 'hitting' in action_label.split(":")[0]:
          if not os.path.exists(f"{args.output_dir}/hitter/hitting"):
            os.makedirs(f"{args.output_dir}/hitter/hitting")
            output_file = f"{args.output_dir}/hitter/hitting/hitter_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
        
          else:
            output_file = f"{args.output_dir}/hitter/hitting/hitter_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)

        elif 'swing' in action_label.split(":")[0]:
          if not os.path.exists(f"{args.output_dir}/hitter/swing"):
            os.makedirs(f"{args.output_dir}/hitter/swing")
            output_file = f"{args.output_dir}/hitter/swing/hitter_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
        
          else:
            output_file = f"{args.output_dir}/hitter/swing/hitter_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)

      elif "filder" in action_label.split(":")[0]:
        if 'throw' in action_label.split(":")[0]:
          if not os.path.exists(f"{args.output_dir}/filder/throw"):
            os.makedirs(f"{args.output_dir}/filder/throw")
            output_file = f"{args.output_dir}/filder/throw/filder_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
        
          else:
            output_file = f"{args.output_dir}/filder/throw/filder_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)

        if 'catch' in action_label.split(":")[0]:
          if not os.path.exists(f"{args.output_dir}/filder/catch"):
            os.makedirs(f"{args.output_dir}/filder/catch")
            output_file = f"{args.output_dir}/filder/catch/filder_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)
        
          else:
            output_file = f"{args.output_dir}/filder/catch/filder_frame_{frame_num}_person_{people_num}.mp4"
            vid.write_videofile(output_file, codec="libx264", remove_temp=True)

def divide_list(input_list, chunk_size):
  divided_list = []
  for i in range(len(input_list)-(chunk_size-1)):
    sublist = input_list[i:i+chunk_size]
    if len(sublist) >= 15:
      divided_list.append(sublist)
  return divided_list

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1 (numpy.ndarray): First vector.
        vec2 (numpy.ndarray): Second vector.

    Returns:
        float: Cosine similarity between the two vectors.
    """
    vec1, vec2 = np.array(vec1), np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm_vec1 * norm_vec2)

    return similarity

def split_list_into_chunks(input_list, chunk_size):
    # 입력 리스트를 chunk_size만큼씩 나누어 저장할 리스트 초기화
    chunks = []

    # 입력 리스트를 chunk_size만큼씩 나누어 저장
    for i in range(0, len(input_list), chunk_size):
        chunks.append(input_list[i:i + chunk_size])

    return chunks

def average_pairs(lst):
    averages = []
    for i in range(0, len(lst), 2):
        pair = lst[i:i + 2]
        avg = sum(pair) / len(pair)
        averages.append(avg)
    return averages

def count_of_lists_at_max_index(lists):
    """
    Find the count of lists in the list at the index with the maximum number of lists.

    Args:
        lists (list): List of lists.

    Returns:
        int: Count of lists in the list at the index with the maximum number of lists.
    """
    max_index = -1
    max_size = -1

    for i, sublist in enumerate(lists):
        current_size = len(sublist)
        if current_size > max_size:
            max_size = current_size
            max_index = i

    if max_index != -1:
        return len(lists[max_index])
    else:
        return 0

def main():
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    
    frame_paths, frames = frame_extract(args.video, args.short_side, tmp_dir.name)
    chunk_size = 240
    middle_count = 5
    # frames = split_list_into_chunks(frames, chunk_size=chunk_size)
    # frame_paths = split_list_into_chunks(frame_paths, chunk_size=chunk_size)
    frames = divide_list(frames, chunk_size)
    frame_paths = divide_list(frame_paths, chunk_size)
    
    for i in range(len(frames)):
      frames[i] = get_middle_values(frames[i], middle_count)
    
    for i in range(len(frames)):
      frame_paths[i] = get_middle_values(frame_paths[i], middle_count)
    
    h, w, _ = frames[0][0].shape

    for frame_path in range(len(frame_paths)):
    # Get Human detection results.
      det_results, _ = detection_inference(args.det_config, args.det_checkpoint,
                                          frame_paths[frame_path], args.det_score_thr,
                                          args.det_cat_id, args.device)
                        
      person_count = count_of_lists_at_max_index(det_results)
      people = {i:[] for i in range(person_count)}
      for i in range(len(det_results[0])):
        people[i].append(det_results[0][i])

      for det in range(1, len(det_results)):
        for person in range(len(det_results[0])):
          for j in range(len(det_results[det])):
            if cosine_similarity(people[person][-1], det_results[det][j]) > 0.9999:
              # print(people[person][-1], det_results[det][j])
              people[person].append(det_results[det][j])

      for i in people:
        # print(i)
        people[i] = [np.array(item).reshape(1, -1) for item in people[i]]
      #   print(len(people[i]))
      #   print(people[i])
      # exit()
        torch.cuda.empty_cache()

      # for i in people:
        if len(people) >= 3:
          pose_results, pose_data_samples = pose_inference(args.pose_config,
                                                          args.pose_checkpoint,
                                                          frame_paths[frame_path], people[0],
                                                          args.device)
          
          # pose_data_samples = split_list_into_chunks(pose_data_samples, chunk_size=15)
          torch.cuda.empty_cache()

          config = mmengine.Config.fromfile(args.config)
          config.merge_from_dict(args.cfg_options)

          model = init_recognizer(config, args.checkpoint, args.device)
          # print(len(pose_results))
          # print(pose_results)
          if len(pose_results):
            result = inference_skeleton(model, pose_results, (h, w))
            max_pred_index = result.pred_score.argmax().item()
            label_map = [x.strip() for x in open(args.label_map).readlines()]
            action_label = label_map[max_pred_index]
            visualize(args, frames[frame_path], pose_data_samples, f"{action_label}: {torch.max(result.pred_score)}", frame_path, i)

    tmp_dir.cleanup()


if __name__ == '__main__':
    main()
    print("다 했음")
