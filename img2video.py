import cv2
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def create_video_from_images(image_folder, output_video_file, fps=15):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg") or img.endswith(".png")]
    if not images:
        return  # 이미지가 없는 경우 함수 종료

    if os.path.exists(output_video_file):
        return  # 이미 비디오 파일이 존재하는 경우 함수 종료

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for image in tqdm(images, desc=f"Processing {image_folder}", leave=False):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

def process_folder(args):
    root_folder, output_folder, label_prefix, fps = args
    for root, dirs, files in os.walk(root_folder):
        if not dirs:  # 하위 폴더가 없는 경우
            relative_path = os.path.relpath(root, root_folder)
            parts = relative_path.split(os.sep)[1:]  # 첫 번째 폴더명을 제외한 나머지 경로
            cleaned_relative_path = os.path.join(*parts)
            output_video_file = os.path.join(output_folder, f"{label_prefix}_{cleaned_relative_path.replace(os.sep, '_')}.avi")
            create_video_from_images(root, output_video_file, fps)

def process_all_folders(base_folder, output_base_folder, label_mapping, fps=15, max_workers=12):
    for category in ["1.Training", "2.Validation"]:
        output_folder = os.path.join(output_base_folder, category.split('.')[1].lower())
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        folder_path = os.path.join(base_folder, category, "원천데이터")
        tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for label, label_prefix in tqdm(label_mapping.items(), desc=f"Processing {category}", leave=False):
                root_folder = os.path.join(folder_path, label)
                if os.path.exists(root_folder):
                    tasks.append(executor.submit(process_folder, (root_folder, output_folder, label_prefix, fps)))

            for task in tqdm(tasks, desc=f"Overall Progress - {category}", leave=False):
                task.result()

# 라벨 매핑
label_mapping = {
    "baseball_ra_ct": "A001",
    "baseball_ra_ff": "A002",
    "baseball_ra_fg": "A003",
    "baseball_ra_ft": "A004",
    "baseball_ra_hb": "A005",
    "baseball_ra_hh": "A006",
    "baseball_ra_hs": "A007",
    "baseball_ra_pb": "A008",
    "baseball_ra_po": "A009",
    "baseball_ra_pp": "A010",
    "baseball_ra_ps": "A011",
    "baseball_ra_pu": "A012",
    "baseball_ra_rr": "A013"
}

# 원천 데이터 폴더와 출력 기본 폴더 설정
base_folder = ''
output_base_folder = 'output'

process_all_folders(base_folder, output_base_folder, label_mapping)
