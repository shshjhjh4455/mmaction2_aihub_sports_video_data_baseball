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
