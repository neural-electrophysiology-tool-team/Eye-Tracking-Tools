import subprocess as sp
import cv2
import pandas as pd
import glob
import os

def convert_h264_mp4(path):
    files_to_convert = glob.glob(path + r'\**\*.h264', recursive=True)
    converted_files = glob.glob(path + r'\**\*.mp4', recursive=True)
    for file in files_to_convert:
        fps = file[file.find('hz') - 2:file.find('hz')]
        if len(fps) != 2:
            fps = 60
            print('could not determine fps, using 60...')
        if not str(fr'{file[:-5]}.mp4') in converted_files:
            sp.run(f'MP4Box -fps {fps} -add {file} {file[:-5]}.mp4')
            print(fr'{file} converted ')
        else:
            print(f'The file {file[:-5]}.mp4 already exists, no conversion necessary')


def validate_no_framedrop(path):
    videos_to_inspect = glob.glob(path + r'\**\*.mp4', recursive=True)
    timestamps_to_inspect = glob.glob(path + r'\**\*.csv', recursive=True)
    for vid in range(len(videos_to_inspect)):
        timestamps = pd.read_csv(timestamps_to_inspect[vid])
        num_reported = timestamps.shape[0]
        cap = cv2.VideoCapture(videos_to_inspect[vid])
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'The video named {os.path.split(videos_to_inspect[vid])[1]} has reported {num_reported} frames '
              f'and has {length} frames, it has dropped {num_reported - length} frames')
        cap.release()

def stamp_diff_videos(path_to_stamp,stamp):
    videos_to_stamp = glob.glob(path_to_stamp + r'\**\*.mp4', recursive=True)
    for vid in videos_to_stamp:
        os.rename(vid, fr'{vid[:-4]}_{stamp}{vid[-4:]}')

path = r'D:\AzulaTrial_21_2_2021\EyeVids\LE'
convert_h264_mp4(path)
validate_no_framedrop(path)


