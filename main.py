import UtilityFunctions as uf
import numpy as np

vid_names = [
    r'D:\AzulaTrials_30_12_2020\NimrodTrial1_30_12_2_20201230T180622\block1\trial1\videos\back_20201230T180623.mp4',
    r'D:\AzulaTrials_30_12_2020\NimrodTrial1_30_12_2_20201230T180622\block1\trial1\videos\left_20201230T180623.mp4',
    r'D:\AzulaTrials_30_12_2020\NimrodTrial1_30_12_2_20201230T180622\block1\trial1\videos\realtime_20201230T180623.mp4',
    r'D:\AzulaTrials_30_12_2020\NimrodTrial1_30_12_2_20201230T180622\block1\trial1\videos\right_20201230T180623.mp4'
            ]

frame_val_list = []
for vid in vid_names:
    print(f'working on video {vid}')
    frame_val = uf.arena_video_initial_thr(vid, 250)
    frame_val_list.append(frame_val)

