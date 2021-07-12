import wx
import numpy as np
import pandas as pd
import glob
import subprocess as sp
import os
from pathlib import Path
from matplotlib import pyplot as plt
from ellipse import LsqEllipse
import cv2
import math
import scipy.stats as stats
from tqdm import tqdm
from matplotlib.patches import Ellipse
from bokeh.plotting import figure, show
from bokeh.models import CustomJS, Slider
import bokeh.layouts

'''
This script defines the BlockSync class which takes all of the relevant data for a given trial and can be utilized
to produce a synchronized dataframe for all video sources to be used for further analysis
'''

class BlockSync:
    """
    This class designed to allow parsing and synchronization of the different files acquired in a given experimental
    block. The class expects a certain file system paradigm:
     - Data will be arranged into block folders under animal folders, where each block contains the next structure:

                                           /----> arena_videos  ->[config.yaml , info.yaml] videos -> [video files, output.log] timestamps -> [csv of timestamps]

    Animal_x ->date(xx_xx_xxxx) -> block_x -----> eye_videos >> LE\RE -> video folder with name -> [video.h264 , video.mp4 , params.json , timestamps.csv]

                                           \----> oe_files >> date_time(xxxx_xx_xx_xx-xx-xx) --> [events.csv] internal open ephys structure from here (NWB format only!!!)

    """

    def __init__(self, animal_num, experiment_date, block_num, path_to_animal_folder):
        """
            defines the relevant block for analysis

            Parameters
            ----------
            animal_num :  str
                the number tag for the animal in the experiment

            experiment_date :  str
                the date of the experiment in DD_MM_YYYY format

            block_num, :  str
                block number to analyze

            path_to_animal_folder :  str
                path to the folder where animal_{animal_num} is located

        """
        self.animal_num = animal_num
        self.experiment_date = experiment_date
        self.block_num = block_num
        self.path_to_animal_folder = path_to_animal_folder
        self.block_path = Path(
            rf'{self.path_to_animal_folder}Animal_{self.animal_num}\{self.experiment_date}\block_{self.block_num}')
        print(f'instantiated block number {self.block_num} at Path: {self.block_path}')
        self.arena_path = self.block_path / 'arena_videos'
        self.arena_files = [x for x in self.arena_path.iterdir()]
        self.arena_videos = None
        self.arena_vidnames = None
        self.arena_timestamps = None
        self.re_videos = None
        self.le_videos = None
        self.arena_sync_df = None
        self.anchor_vid_name = None
        self.arena_frame_val_list = None
        self.arena_brightness_df = None

    def handle_arena_files(self):
        """
        method to fix arena files names and append them to separate video and timestamp files
        this is a preliminary stage for arena internal synchronization

        sets the following attributes:
        self.arena_videos: list
            list of videos after name correction
        self.arena_timestamps : list
            list of .csv files associated with
        """
        # fix names
        for i in self.arena_files:
            if '-' in i.name:
                newname = i.name.replace('-', '_')
                newpath = i.parent / newname
                i.replace(newpath)
        self.arena_videos = [x for x in self.arena_files if x.suffix == '.mp4']
        self.arena_timestamps = [x for x in self.arena_files if x.suffix == '.csv']
        self.arena_vidnames = [i.name for i in self.arena_videos]
        print(f'Arena video Names:')
        print(*self.arena_vidnames, sep='\n')

    def handle_eye_videos(self):
        """
        This method converts and renames the eye tracking videos in the files tree ino workable .mp4 files
        """
        eye_vid_path = self.block_path / 'eye_videos'
        print('converting videos...')
        files_to_convert = glob.glob(str(eye_vid_path) + r'\**\*.h264', recursive=True)
        converted_files = glob.glob(str(eye_vid_path) + r'\**\*.mp4', recursive=True)
        print(f'converting files: {files_to_convert}')
        if len(files_to_convert) == 0:
            print('found no eye videos to handle...')
            return None
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
        print('Validating videos...')
        videos_to_inspect = glob.glob(str(eye_vid_path) + r'\**\*.mp4', recursive=True)
        timestamps_to_inspect = glob.glob(str(eye_vid_path) + r'\**\*.csv', recursive=True)
        for vid in range(len(videos_to_inspect)):
            timestamps = pd.read_csv(timestamps_to_inspect[vid])
            num_reported = timestamps.shape[0]
            cap = cv2.VideoCapture(videos_to_inspect[vid])
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'The video named {os.path.split(videos_to_inspect[vid])[1]} has reported {num_reported} frames '
                  f'and has {length} frames, it has dropped {num_reported - length} frames')
            cap.release()
        print('stamping LE video')
        stamp = 'LE'
        path_to_stamp = eye_vid_path / stamp
        videos_to_stamp = glob.glob(str(path_to_stamp) + r'\**\*.mp4', recursive=True)
        for vid in videos_to_stamp:
            if stamp not in str(vid):
                os.rename(vid, fr'{vid[:-4]}_{stamp}{vid[-4:]}')
        self.le_videos = glob.glob(str(self.block_path) + r'\eye_videos\LE\**\*.mp4')
        self.re_videos = glob.glob(str(self.block_path) + r'\eye_videos\RE\**\*.mp4')

    @staticmethod
    def get_closest_frame(timestamp, vid_timeseries, report_acc=None):
        """
        This function extracts a frame from a series so that it is as close as possible to a given timestamp
        :param timestamp: The time to match a frame to
        :param vid_timeseries: The time frames series to look at for a match
        :param report_acc: if set to 1, will report the accuracy of the match
        :return: index_of_lowest_diff , accuracy of match (if requested)
        """
        array = np.abs((vid_timeseries.to_numpy()) - timestamp)
        index_of_lowest_diff = np.argmin(array)
        if report_acc == 1:
            accuracy = abs(vid_timeseries[index_of_lowest_diff] - timestamp)
            return index_of_lowest_diff, accuracy
        else:
            return index_of_lowest_diff

    def synchronize_arena_timestamps(self, return_dfs=False):
        """
        This function reads the different arena timestamps files, chooses the longest as an anchor and fits
        frames corresponding with the closest timestamp to the anchor.
        It creates self.arena_sync_df and self.anchor_vid_name
        """
        # read the timestamp files
        len_list = []
        df_list = []
        for p in self.arena_timestamps:
            df = pd.read_csv(p)
            df_list.append(df)
            len_list.append(len(df))

        # pick the longest as an anchor
        anchor_ind = len_list.index(max(len_list))
        anchor_vid = df_list[anchor_ind]
        self.anchor_vid_name = self.arena_vidnames[anchor_ind]

        # construct a synchronization dataframe
        self.arena_sync_df = pd.DataFrame(data=[],
                                          columns=self.arena_vidnames,
                                          index=range(len(anchor_vid)))

        # populate the df, starting with the anchor:
        self.arena_sync_df[self.arena_sync_df.columns[anchor_ind]] = range(len(anchor_vid))
        vids_to_sync = list(self.arena_sync_df.drop(axis=1, labels=self.anchor_vid_name).columns)# CHECK ME !!!!
        anchor_df = df_list.pop(anchor_ind)
        df_to_sync = df_list
        # iterate over rows and videos to find the corresponding frames
        print('Synchronizing the different arena videos')
        for row in tqdm(self.arena_sync_df.index):
            anchor = anchor_vid.timestamp[row]
            for vid in range(len(df_to_sync)):
                frame_num = self.get_closest_frame(anchor, df_to_sync[vid])
                self.arena_sync_df.loc[row, vids_to_sync[vid]] = frame_num
        print(f'The anchor video used was "{self.anchor_vid_name}"')

        if return_dfs:
            return self.arena_sync_df, self.anchor_vid_name

    @staticmethod
    def arena_video_initial_thr(vid_path, threshold_value, show_frames=False):
        """
            This function works through an arena video to determine where the LEDs are on and when off
            :param threshold_value: value of the frame threshold
            :param show_frames: if true will show the video after thresholding
            :param  vid_path: Path to video. When ShowFrames is True a projection of the frames after threshold is presented

            :return: np.array with frame numbers and mean values after threshold
            """
        cap = cv2.VideoCapture(vid_path)
        all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        flag = 0
        i = 0
        mean_values = []
        indexes = []
        while flag == 0:
            print('Frame number {} of {}'.format(i, all_frames), end='\r', flush=True)
            ret, frame = cap.read()
            if not ret:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey[grey < threshold_value] = 0
            mean_values.append(np.mean(grey))
            indexes.append(i)
            if show_frames:
                cv2.imshow('Thresholded_Frames', grey)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            i += 1
        cap.release()
        cv2.destroyAllWindows()
        frame_val = np.array((indexes, mean_values))
        return frame_val

    @staticmethod
    def produce_frame_val_list(vid_paths, threshold_value):
        """
        Parameters
        ----------
        vid_paths: list of str
            a list of str paths to videos for analysis

        threshold_value: float
            the threshold to use in order to concentrate on LEDs

        Returns
        ----------
        frame_val_list:
            a list of mean pixel values for each frame after threshold

        """
        frame_val_list = []
        for vid in vid_paths:
            print(f'working on video {vid}')
            frame_val = BlockSync.arena_video_initial_thr(str(vid), threshold_value)
            frame_val_list.append(frame_val)
        print(f'done, frame_val_list contains {len(frame_val_list)} objects', flush=True)

        return frame_val_list

    def create_arena_brightness_df(self, threshold_value, return_df=False):
        """
        This is a validation function for the previous synchronization steps and will produce self.ar_frame_val_list
        to plot a brightness trace after

        Parameters
        ----------
        threshold_value: float
            the threshold to use in order to concentrate on LEDs

        return_df: binary
            if set to True, will return the arena brightness dataframe
        """
        if self.arena_frame_val_list is None:
            self.arena_frame_val_list = BlockSync.produce_frame_val_list(self.arena_videos, threshold_value)

        # arrange into dataframe:
        self.arena_brightness_df = pd.DataFrame(index=self.arena_sync_df[self.anchor_vid_name])
        for ind, vid in enumerate(self.arena_vidnames):
            vid_val_arr = stats.zscore(self.arena_frame_val_list[ind][1])
            sync_list = self.arena_sync_df[vid].astype(int)
            sync_list[sync_list >= len(vid_val_arr)] = len(vid_val_arr) - 1
            self.arena_brightness_df.insert(loc=0,
                                            column=str(vid),
                                            value=vid_val_arr[sync_list])

    def validate_arena_synchronization(self):
        if self.arena_brightness_df is None:
            print('No arena_brightness_df, run the create_arena_brightness_df method')
        x_axis = self.arena_brightness_df.index.values
        columns = self.arena_brightness_df.columns
        bokeh_fig = figure(title='Arena Video Synchronization Verify',
                           x_axis_label='Frame',
                           y_axis_label='Z_Score',
                           plot_width=1500,
                           plot_height=700
                           )
        color_list = ['orange', 'purple', 'teal', 'green', 'red']
        for ind, video in enumerate(columns):
            bokeh_fig.line(x_axis, self.arena_brightness_df[video],
                           legend_label=video,
                           line_width=1,
                           line_color=color_list[ind])
        show(bokeh_fig)
