import os
import subprocess as sb
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

# 비디오 처리 함수
def process_video(ntu_path, video_path, output_path):
    try:
        sb.run(f"python {ntu_path} {video_path} {output_path}", shell=True)
        return f"{video_path} 처리 완료"
    except Exception as e:
        return f"{video_path} 처리 실패: {e}"

# 폴더 내 파일명 리스트 반환 함수
def get_file_names(folder_path):
    try:
        file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return file_names
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# 지정된 폴더의 비디오 파일 처리 함수
def process_folder(ntu_path, avi_folder_path, output_folder_path):
    file_names = get_file_names(avi_folder_path)
    if file_names:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_video, ntu_path, f"{avi_folder_path}/{file_name}", f"{output_folder_path}/{file_name.split('.')[0]}.pkl") for file_name in file_names]
            for future in as_completed(futures):
                print(future.result())
    else:
        print(f"{avi_folder_path}에서 파일명을 가져오는 데에 오류가 발생했습니다.")

# .pkl 파일 로드 함수
def load_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"{file_path} 파일을 불러오는 데 실패했습니다: {e}")
        return None

# .pkl 파일 병합 함수
def merge_pkl_files_parallel(folder_path, output_file_path):
    pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
    merged_data = []

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(load_pkl, file): file for file in pkl_files}
        for future in as_completed(future_to_file):
            data = future.result()
            if data is not None:
                merged_data.append(data)

    try:
        with open(output_file_path, 'wb') as output_file:
            pickle.dump(merged_data, output_file)
        print(f"병합된 파일이 저장되었습니다: {output_file_path}")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")

# 메인 실행 부분
ntu_path = 'tools/data/skeleton/ntu_pose_extraction.py'
base_folder_path = '/content/drive/MyDrive/project_baseball/mmaction2/data/skeleton/output'
folder_types = ['validation', 'training']

for folder_type in folder_types:
    avi_folder_path = os.path.join(base_folder_path, folder_type)
    output_folder_path = f"{base_folder_path}/{folder_type}_pkl"
    print(f"{folder_type} 폴더 처리 시작...")
    process_folder(ntu_path, avi_folder_path, output_folder_path)

    # 병합 과정 시작
    input_folder_path = output_folder_path
    output_file_path = f"{base_folder_path}/{folder_type}.pkl"
    merge_pkl_files_parallel(input_folder_path, output_file_path)
