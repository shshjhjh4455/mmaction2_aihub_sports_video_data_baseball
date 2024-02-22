# aihub_sports_video_data_baseball


스포츠 영상 데이터 (야구)
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=488

라벨링 정리

baseball_ra <-ra 만 어노테이션 값 포함되어 있음

baseball_ra_ct <- catcher, throw 

baseball_ra_ff <- fielder, catch_a_fly_ball

baseball_ra_fg <- fielder, catch_a_ground_ball

baseball_ra_ft <- fielder, fielder_throw

baseball_ra_hb <- hitter, bunt

baseball_ra_hh <- hitter, hitting

baseball_ra_hs <- hitter, swing

baseball_ra_pb <- pitcher, balk

baseball_ra_po <- pitcher, overhand_throw

baseball_ra_pp <- pitcher, pick_off_throw

baseball_ra_ps <- pitcher, side_arm_throw

baseball_ra_pu <- pitcher, underhand_throw

baseball_ra_rr <- runner, run

img2video.py
이미지 프레임 별로 나눠저 있기 때문에 각 폴더별로 동영상으로 변환


데이터셋 설명
컬럼 : 13개
![Screenshot 2024-02-22 at 1 45 45 PM](https://github.com/shshjhjh4455/aihub_sports_video_data_baseball/assets/44297309/0c58e641-b220-486a-b4e3-29f6ed5a4104)

각 동작에 해당하는 영상 - 이미지 프레임 
Training : 3888개
Validation : 504개
![Screenshot 2024-02-22 at 1 46 24 PM](https://github.com/shshjhjh4455/aihub_sports_video_data_baseball/assets/44297309/c71045a6-ae82-4d86-8ce9-618df4810f0b)

데이터 전처리 과정 : 동영상 변환
mmaction2에서 스켈레톤 값 추출을 위한 인풋으로 동영상을 받기 때문에 이미지 별로 나눠진 파일들을 영상으로 합친다. 
AI_hub의 각 폴더의 이미지 -> 각각의 동영상으로 변환
![Screenshot 2024-02-22 at 1 48 25 PM](https://github.com/shshjhjh4455/aihub_sports_video_data_baseball/assets/44297309/928a1d8e-eb3d-4126-aa8d-69b5392c8d23)

처리 완료된 동영상
![Screenshot 2024-02-22 at 1 49 26 PM](https://github.com/shshjhjh4455/aihub_sports_video_data_baseball/assets/44297309/2aea2ece-5a6d-44ca-831f-9abebc43040a)

