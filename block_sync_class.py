import glob
import os
import subprocess as sp
from pathlib import Path
from ellipse import LsqEllipse
import math
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
from bokeh.plotting import figure, show
from tqdm import tqdm
import glob

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
        self.exp_date_time = os.listdir(fr'{self.block_path}\oe_files')[0]
        self.arena_path = self.block_path / 'arena_videos'
        self.arena_files = None
        self.arena_videos = None
        self.arena_vidnames = None
        self.arena_timestamps = None
        self.re_videos = None
        self.le_videos = None
        self.arena_sync_df = None
        self.anchor_vid_name = None
        self.arena_frame_val_list = None
        self.arena_brightness_df = None
        self.channeldict = {
            5: 'L_eye_TTL',
            6: 'Arena_TTL',
            7: 'Logical ON/OFF',
            8: 'R_eye_TTL'
        }
        self.oe_events = None
        self.oe_off_events = None
        self.ts_dict = None
        self.block_starts = None
        self.block_ends = None
        self.block_length = None
        self.synced_videos = None
        self.accuracy_report = None
        self.anchor_signal = None
        self.le_frame_val_list = None
        self.re_frame_val_list = None
        self.eye_brightness_df = None
        self.l_eye_values = None
        self.r_eye_values = None
        self.arena_first_ttl_frame = None
        self.synced_videos_validated = None
        self.le_csv = None
        self.re_csv = None
        self.le_ellipses = None
        self.re_ellipses = None
        self.arena_bdf = None

    def __str__(self):
        return str(f'animal {self.animal_num}, block {self.block_num}, on {self.exp_date_time}')

    def __repr__(self):
        return str(f'BlockSync object with block_num {self.block_num}')

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
        self.arena_files = [x for x in self.arena_path.iterdir()]
        # fix names
        for i in self.arena_files:
            if '-' in i.name:
                newname = i.name.replace('-', '_')
                newpath = i.parent / newname
                i.replace(newpath)
        self.arena_files = [x for x in self.arena_path.iterdir()]
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
            if str(fr'{file[:-5]}.mp4') not in converted_files:
                if str(fr'{file[:-5]}_LE.mp4') not in converted_files:
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
        stamp = 'LE'
        path_to_stamp = eye_vid_path / stamp
        videos_to_stamp = glob.glob(str(path_to_stamp) + r'\**\*.mp4', recursive=True)
        for vid in videos_to_stamp:
            if stamp + '.mp4' not in str(vid):
                print('stamping LE video')
                os.rename(vid, fr'{vid[:-4]}_{stamp}{vid[-4:]}')
        self.le_videos = glob.glob(str(self.block_path) + r'\eye_videos\LE\**\*.mp4')
        self.re_videos = glob.glob(str(self.block_path) + r'\eye_videos\RE\**\*.mp4')

    @staticmethod
    def get_closest_frame(timestamp, vid_timeseries, report_acc=None):
        """
        This function extracts a frame from a series so that it is as close as possible to a given timestamp

        Parameters
        ----------
        timestamp: float
            The time to match a frame to
        vid_timeseries: pd.Series
            The time frames series to look at for a match
        report_acc: boolean
            if set to 1, will report the accuracy of the match, index_of_lowest_diff , accuracy of match (if requested)
        """
        array = np.abs((vid_timeseries.to_numpy()) - timestamp)
        index_of_lowest_diff = np.argmin(array)
        if report_acc == 1:
            accuracy = abs(vid_timeseries[index_of_lowest_diff] - timestamp)
            return index_of_lowest_diff, accuracy
        else:
            return index_of_lowest_diff

    def synchronize_arena_timestamps(self, return_dfs=False, export_sync_df=False):
        """
        This function reads the different arena timestamps files, chooses the longest as an anchor and fits
        frames corresponding with the closest timestamp to the anchor.
        It creates self.arena_sync_df and self.anchor_vid_name
        """
        # read the timestamp files
        len_list = []
        df_list = []
        for p in self.arena_timestamps:
            if p.name != 'events.csv':
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
        if export_sync_df:
            self.arena_sync_df.to_csv(self.block_path / 'arena_synchronization.csv')

    @staticmethod
    def arena_video_initial_thr(vid_path, threshold_value, show_frames=False):
        """
            This function works through an arena video to determine where the LEDs are on and when off

            Parameters
            ----------
            threshold_value: float
                value of the frame threshold
            show_frames: Binary
                if true will show the video after thresholding
            vid_path: Path
                Path to video. When ShowFrames is True a projection of the frames after threshold is presented

            Returns
            ----------
            frame_val: np.array
                 frame numbers and mean values after threshold
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

    def create_arena_brightness_df(self, threshold_value, export=True):
        """
        This is a validation function for the previous synchronization steps and will produce
        self.arena_brightness_df to plot a brightness trace and check synchronization

        Parameters
        ----------
        threshold_value: float
            the threshold to use in order to concentrate on LEDs

        export: binary
            if set to true, will export a dataframe to the analysis folder inside the block directory
        """
        if self.arena_frame_val_list is None:
            self.arena_frame_val_list = BlockSync.produce_frame_val_list(self.arena_videos, threshold_value)

        # arrange into dataframe:
        self.arena_brightness_df = pd.DataFrame(index=self.arena_sync_df[self.anchor_vid_name].values)
        for ind, vid in enumerate(self.arena_vidnames):
            vid_val_arr = stats.zscore(self.arena_frame_val_list[ind][1])
            sync_list = self.arena_sync_df[vid].astype(int)
            sync_list[sync_list >= len(vid_val_arr)] = len(vid_val_arr) - 1
            self.arena_brightness_df.insert(loc=0,
                                            column=str(vid),
                                            value=vid_val_arr[sync_list])
        if export:
            self.arena_brightness_df.to_csv(self.block_path / 'analysis/arena_brightness.csv')

    def validate_arena_synchronization(self):
        if self.arena_brightness_df is None:
            print('No arena_brightness_df, run the create_arena_brightness_df method')
        x_axis = self.arena_brightness_df.index.values
        columns = self.arena_brightness_df.columns
        bokeh_fig = figure(title=f'Block Number {self.block_num} Arena Video Synchronization Verify',
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


    @staticmethod
    def get_frame_timeseries(df, channel):
        index_range = range(0, len(df[channel][df[channel].notna()]))
        timeseries = pd.Series(df[channel][df[channel].notna()])
        timeseries = pd.Series(timeseries.values, index=index_range, name=channel)
        return timeseries


    @staticmethod
    def oe_events_parser(open_ephys_csv_path, channel_names, export_path=None):
        """
        :param open_ephys_csv_path: The path to an open ephys analysis tools exported csv (using TrialReporter.ipynb)
        :param channel_names: a dictionary of the form -
                        { 1 : 'channel name' (L_eye_camera)
                          2 : 'channel name' (Arena_TTL)
                          etc..
                        }
        :param export_path: default None, if a path is specified a csv file will be saved
        :returns open_ephys_events: a pandas DataFrame object where each column has the ON events of one channel
                                    and has a title from channel_names
        :returns open_ephys_off_events: same but for the OFF states (only important for the logical start-stop signal)

        """

        # Infer the active channels:
        df = pd.read_csv(open_ephys_csv_path)
        channels = df['channel'].to_numpy(copy=True)
        channels = np.unique(channels)
        df_onstate = df[df['state'] == 1]  # cut the df to represent only rising edges
        df_offstate = df[df['state'] == 0]  # This one is important for the ON/OFF signal of the arena
        list = []
        off_list = []
        for chan in channels:  # extract a pandas series of the ON stats timestamps for each channel
            Sname = channel_names[chan]
            s = pd.Series(df_onstate['timestamp'][df_onstate['channel'] == chan], name=Sname)
            offs = pd.Series(df_offstate['timestamp'][df_offstate['channel'] == chan], name=Sname)
            list.append(s)
            off_list.append(offs)
        open_ephys_events = pd.concat(list, axis=1)
        open_ephys_off_events = pd.concat(off_list, axis=1)
        if export_path is not None:
            if export_path not in os.listdir(str(open_ephys_csv_path).split('events.csv')[0][:-1]):
                open_ephys_events.to_csv(export_path)
        return open_ephys_events, open_ephys_off_events

    def import_open_ephys_events(self):
        """
        Method for importing the Open Ephys events.csv results of the OETrialReporter.ipynb mini-pipe
        defines

        """
        # first, parse the events of the open-ephys recording
        self.oe_events, self.oe_off_events = BlockSync.oe_events_parser(
            self.block_path / rf'oe_files\{self.exp_date_time}\events.csv',
            self.channeldict,
            export_path=self.block_path / rf'oe_files\{self.exp_date_time}\parsed_events.csv')
        # now, arrange them in distinct columns of a timeseries dataframe
        self.ts_dict = {}
        for channel in list(self.oe_events.columns):
            ts = pd.Series(data=BlockSync.get_frame_timeseries(self.oe_events, str(channel)), name=channel)
            self.ts_dict[f'{channel}'] = ts
            if channel == 'Arena_TTL':
                ttl_breaks = np.where(np.diff(ts.values) > 0.5)
                self.block_starts = ts[ttl_breaks[0][0]+1]
                self.block_ends = ts[ttl_breaks[0][1]]
                self.block_length = self.block_ends - self.block_starts
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

    def synchronize_block(self):
        """
        This method defines the synchronization dataframe for the block, where frames from the different video sources
        are aligned with an anchor signal spanning the synchronized experiment timeframe
        """
        # define the anchor signal
        self.anchor_signal = np.arange(self.block_starts, self.block_ends, 1 / 60)
        # define the dataframe for the synchronized video frames output
        self.synced_videos = pd.DataFrame(data=None,
                                          index=range(len(self.anchor_signal)),
                                          columns=self.oe_events.columns)
        # define the accuracy report for all frames
        self.accuracy_report = pd.DataFrame(data=None,
                                            index=range(len(self.anchor_signal)),
                                            columns=self.oe_events.columns)
        print(f'synchronizing {self.block_ends - self.block_starts} seconds of videos...')
        for frame in tqdm(range(len(self.anchor_signal))):
            anchor_time = self.anchor_signal[frame]
            for vid in list(self.ts_dict.keys()):
                f, a = BlockSync.get_closest_frame(anchor_time, self.ts_dict[vid], report_acc=1)
                self.synced_videos[vid][frame] = f
                self.accuracy_report[vid][frame] = a

    def update_arena_bdf(self, arena_col='Arena_TTL'):
        """
        method to update the arena brightness df with the new synced video base
        """
        self.arena_first_ttl_frame = self.synced_videos[arena_col][0]
        self.arena_bdf = pd.DataFrame(data=None,
                                      index=self.anchor_signal)
        for col in self.arena_brightness_df.columns:
            arena_brightness_col = pd.Series(index=range(len(self.anchor_signal)))
            for frame in range(len(self.anchor_signal) - 1):
                arena_brightness_col[frame] = self.arena_brightness_df[
                    f'{col}'][self.synced_videos[arena_col][frame] - self.arena_first_ttl_frame]
            self.arena_bdf.insert(loc=0,
                                  column=col,
                                  value=arena_brightness_col.values)

    def create_eye_brightness_df(self, threshold_value,
                                 right_eye_col='R_eye_TTL',
                                 left_eye_col='L_eye_TTL',
                                 arena_col='Arena_TTL',
                                 arena_vid_to_use=None):
        """
        This is a validation function for the preceding synchronization steps - It utilizes a Z score of
        the image brightness after thresholding per frame to find the LED blinks incorporated into the recording

        Parameters
        ----------
        threshold_value: float
            the threshold to use for the analysis
        right_eye_col: str
            the column representing the right eye in the self.synced_videos attribute
        left_eye_col: str
            the column representing the left eye in the self.synced_videos attribute
        arena_col: str
            the column representing the arena in the self.synced_videos attribute
        arena_vid_to_use: None or str
            if None, will use the anchor video, if string -
            has to be the video name of an arena video file that is synced
        """
        if arena_vid_to_use is None:
            arena_vid = self.anchor_vid_name
        elif arena_vid_to_use in self.arena_vidnames:
            arena_vid = arena_vid_to_use
        else:
            raise NameError('There is no arena video with the name specified as arena_vid_to_use')

        if self.le_frame_val_list is None:
            self.le_frame_val_list = BlockSync.produce_frame_val_list(self.le_videos, threshold_value)
        if self.re_frame_val_list is None:
            self.re_frame_val_list = BlockSync.produce_frame_val_list(self.re_videos, threshold_value)

        self.l_eye_values = stats.zscore(self.le_frame_val_list[0][1])
        self.r_eye_values = stats.zscore(self.re_frame_val_list[0][1])

        self.eye_brightness_df = pd.DataFrame(index=self.anchor_signal)
        self.eye_brightness_df.insert(loc=0,
                                      column=right_eye_col,
                                      value=self.r_eye_values[self.synced_videos[f'{right_eye_col}'].values.astype(int)][
                                            0:len(self.anchor_signal)])
        self.eye_brightness_df.insert(loc=0,
                                      column=left_eye_col,
                                      value=self.l_eye_values[self.synced_videos[f'{left_eye_col}'].values.astype(int)][
                                            0:len(self.anchor_signal)])

        # self.arena_first_ttl_frame = self.synced_videos[arena_col][0]
        # self.arena_brightness_col = pd.Series(index=range(len(self.anchor_signal)))
        # for frame in range(len(self.anchor_signal)-1):
        #     self.arena_brightness_col[frame] = self.arena_brightness_df[
        #         f'{arena_vid}'][self.synced_videos[arena_col][frame] - self.arena_first_ttl_frame]
        #
        # self.eye_brightness_df.insert(loc=0,
        #                               column=arena_col,
        #                               value=self.arena_brightness_col.values)

    def validate_eye_synchronization(self, arena_vid_to_use=None):
        if self.eye_brightness_df is None:
            print('No eye_brightness_df, run the create_eye_brightness_df method')
        x_axis = range(len(self.eye_brightness_df))
        columns = self.eye_brightness_df.columns
        bokeh_fig = figure(title=f'Block number {self.block_num} Eye Video Synchronization Verify',
                           x_axis_label='Time',
                           y_axis_label='Z_Score',
                           plot_width=1500,
                           plot_height=700
                           )
        color_list = ['blue', 'red', 'teal']
        for ind, video in enumerate(columns):
            bokeh_fig.line(x_axis, self.eye_brightness_df[video],
                           legend_label=video,
                           line_width=1,
                           line_color=color_list[ind])

        if arena_vid_to_use is None:
            arena_vid_to_use = self.arena_bdf.columns[0]
        bokeh_fig.line(x_axis, self.arena_bdf[arena_vid_to_use],
                       legend_label=arena_vid_to_use,
                       line_width=1,
                       line_color='purple')

        show(bokeh_fig)

    def synchronize_by_led_blink(self, arena_vid_to_use=None, threshold_value_for_eyes=30):
        """
        This method determines where the minimum points of the video brightnesses are and moves the synced_videos
        dataframe series to create synchronization between the different video sources - This should be run after
        validate_eye_synchronization when you know that you have good LED blinks in the brightness dataframes

        Parameters
        ----------
        arena_vid_to_use: None or str
            if None, will use the anchor video, if string -
            has to be the video name of an arena video file that is synced

        threshold_value_for_eyes: float
            the threshold to use for the analysis
        """

        a = self.arena_bdf
        flag = False
        threshold = -4.0
        while not flag:
            print(threshold)
            arena_mask = a.mask(a < threshold, other=True)
            arena_mask = arena_mask.mask(a > threshold, other=False)
            suspect_list = []
            for ind in range(len(arena_mask)):
                if sum(arena_mask.iloc[ind].values) >= 3:
                    suspect_list.append(ind)
            if len(suspect_list) != 0:
                flag = True
            else:
                threshold += 0.2
        print(threshold)
        if len(suspect_list) > 10 or threshold >= -1:
            print('bad arena brightness df, manual synchronization required')
            self.validate_arena_synchronization()
            arena_blink_ind = input('please identify and enter the led blink frame:')
        else:
            arena_blink_ind = suspect_list[0]
        search_range = range(arena_blink_ind-50, arena_blink_ind+50)
        print(f'The suspect list is: {suspect_list}')
        print(f'the search range is {search_range}')

        # both eyes find blink
        le_sync_range = self.eye_brightness_df.L_eye_TTL.iloc[search_range]
        re_sync_range = self.eye_brightness_df.R_eye_TTL.iloc[search_range]
        ls = [le_sync_range, re_sync_range]
        for ind, eye in enumerate(ls):
            flag = False
            threshold = -50
            while not flag:
                mask = eye.mask(eye <= threshold, other=True)
                mask = mask.mask(eye > threshold, other=False)
                if sum(mask) != 0:
                    print(f'index inside range = {np.where(mask == True)}')
                    print(f'threshold used: {threshold}')
                    flag = 1
                else:
                    threshold += 1
            if ind == 0:
                le_blink_at = mask.iloc[np.where(mask == True)].index.values[0]
                le_blink_ind = self.eye_brightness_df.index.get_loc(le_blink_at)
                print(f'done, left ind is : {le_blink_ind}')
            elif ind == 1:
                re_blink_at = mask.iloc[np.where(mask == True)].index.values[0]
                re_blink_ind = self.eye_brightness_df.index.get_loc(re_blink_at)
                print(f'done, right ind is : {re_blink_ind}')

        re_drift = re_blink_ind - arena_blink_ind
        le_drift = le_blink_ind - arena_blink_ind

        print(f'right eye drift is {re_drift} frames')
        print(f'left eye drift is {le_drift} frames')
        print('correcting synced_videos...')
        print(f'The method intends to move the right eye video {re_drift} frames and the \nLeft video {le_drift}')
        self.synced_videos['R_eye_TTL'] = self.synced_videos['R_eye_TTL'] + re_drift
        self.synced_videos['L_eye_TTL'] = self.synced_videos['L_eye_TTL'] + le_drift
        self.create_eye_brightness_df(threshold_value=threshold_value_for_eyes)
        self.validate_eye_synchronization()

    def import_arena_brightness_file(self):
        self.arena_brightness_df = pd.read_csv(self.block_path / 'analysis/arena_brightness.csv',
                                               index_col=0)
        #self.arena_brightness_df.drop(columns='Unnamed: 0')
        print(f'block {self.block_num} has imported the arena_brightness_df attribute')
        print('next command to run is import_open_ephys_events()')

    def create_synced_videos_validated(self):
        self.synced_videos_validated = self.synced_videos
        self.synced_videos_validated.insert(loc=0, column='Time', value=self.anchor_signal)

    def export_everything(self):
        """
        Method to save csv files of all internal variables for re-instantiation of the BlockSync class in later sessions
        """
        if self.arena_bdf is not None:
            self.arena_bdf.to_csv(self.block_path / 'analysis/arena_bdf.csv')
            print(f'saved arena_bdf.csv for block number {self.block_num} in the analysis file')
        if self.eye_brightness_df is not None:
            self.eye_brightness_df.to_csv(self.block_path / 'analysis/eye_brightness_df.csv')
            print(f'saved eye_brightness_df.csv for block number {self.block_num} in the analysis file')
        if self.synced_videos_validated is not None:
            self.synced_videos_validated.to_csv(self.block_path / 'analysis/synced_videos_validated.csv')
            print(f'saved sync_videos_validated.csv for block number {self.block_num} in the analysis file')
        if self.re_ellipses is not None:
            self.re_ellipses.to_csv(self.block_path / 'analysis/re_ellipses.csv')
        if self.le_ellipses is not None:
            self.le_ellipses.to_csv(self.block_path / 'analysis/le_ellipses.csv')

    def import_everything(self):
        """
        Method to import all of the cache without the pickle file...
        """
        p_arena_bdf = self.block_path / 'analysis/arena_bdf.csv'
        if p_arena_bdf.exists():
            self.arena_bdf = pd.read_csv(p_arena_bdf,
                                         index_col=0)
        p_eye_brightness_df = self.block_path / 'analysis/eye_brightness_df.csv'
        if p_eye_brightness_df.exists():
            self.eye_brightness_df = pd.read_csv(p_eye_brightness_df,
                                                 index_col=0)
        p_synced_videos_validated = self.block_path / 'analysis/synced_videos_validated.csv'
        if p_synced_videos_validated.exists():
            self.synced_videos_validated = pd.read_csv(p_synced_videos_validated,
                                                       index_col=0)
        p_arena_brightness_df = self.block_path / 'analysis/arena_brightness.csv'
        if p_arena_brightness_df.exists():
            self.arena_brightness_df = pd.read_csv(p_arena_brightness_df,
                                                   index_col=0)
        p_re_ellipses = self.block_path / 'analysis/re_ellipses.csv'
        if p_re_ellipses.exists():
            self.re_ellipses = pd.read_csv(p_re_ellipses,
                                           index_col=0)
        p_le_ellipses = self.block_path / 'analysis/le_ellipses.csv'
        if p_le_ellipses.exists():
            self.le_ellipses = pd.read_csv(p_le_ellipses,
                                           index_col=0)

    def manual_sync(self, threshold_value_for_eyes):
        print('This is a manual synchronization step, a graph should appear now...')
        self.validate_arena_synchronization()
        arena_blink_ind = int(input('At what index does the arena LED blink occur?'))
        self.validate_eye_synchronization()
        re_blink_ind = int(input('At what index closest to the arena LED blink does the right eye LED blink occur?'))
        le_blink_ind = int(input('At what index closest to the arena LED blink does the left eye LED blink occur?'))
        re_drift = re_blink_ind - arena_blink_ind
        le_drift = le_blink_ind - arena_blink_ind
        self.synced_videos['R_eye_TTL'] = self.synced_videos['R_eye_TTL'] + re_drift
        self.synced_videos['L_eye_TTL'] = self.synced_videos['L_eye_TTL'] + le_drift
        self.create_eye_brightness_df(threshold_value=threshold_value_for_eyes)
        self.validate_eye_synchronization()


    @staticmethod
    def eye_tracking_analysis(dlc_video_analysis_csv, uncertainty_thr):
        """
        :param dlc_video_analysis_csv: the csv output of a dlc analysis of one video, already read by pandas with header=1
        :param uncertainty_thr: The confidence P value to use as a threshold for datapoint validity in the analysis
        :returns ellipse_df: a DataFrame of ellipses parameters (center, width, height, phi, size) for each video frame

        """
        # import the dataframe and convert it to floats
        data = dlc_video_analysis_csv
        data = data.iloc[1:].apply(pd.to_numeric)
        # sort the pupil elements to x and y, with p as probability
        pupil_elements = np.array([x for x in data.columns if 'Pupil' in x])
        pupil_xs = data[pupil_elements[np.arange(0, len(pupil_elements), 3)]]
        pupil_ys = data[pupil_elements[np.arange(1, len(pupil_elements), 3)]]
        pupil_ps = data[pupil_elements[np.arange(2, len(pupil_elements), 3)]]
        # rename dataframes for masking with p values of bad points:
        pupil_ps = pupil_ps.rename(columns=dict(zip(pupil_ps.columns, pupil_xs.columns)))
        pupil_ys = pupil_ys.rename(columns=dict(zip(pupil_ys.columns, pupil_xs.columns)))
        good_points = pupil_ps > uncertainty_thr
        pupil_xs = pupil_xs[good_points]
        pupil_ys = pupil_ys[good_points]
        # Do the same for the edges
        # edge_elements = [x for x in data.columns if 'edge' in x]
        # edge_xs = data[edge_elements[np.arange(0,len(edge_elements),3)]]
        # edge_ys = data[edge_elements[np.arange(1,len(edge_elements),3)]]
        # edge_ps = data[edge_elements[np.arange(2,len(edge_elements),3)]]
        # edge_ps = edge_ps.rename(columns=dict(zip(edge_ps.columns,edge_xs.columns)))
        # edge_ys = edge_ys.rename(columns=dict(zip(edge_ys.columns,edge_xs.columns)))
        # e = edge_ps < uncertainty_thr

        # work row by row to figure out the ellipses
        ellipses = []
        caudal_edge_ls = []
        rostral_edge_ls = []
        for row in tqdm(range(1, len(data) - 1)):
            # first, take all of the values, and concatenate them into an X array
            x_values = pupil_xs.loc[row].values
            y_values = pupil_ys.loc[row].values
            X = np.c_[x_values, y_values]
            # now, remove nan values, and check if there are enough points to make the ellipse
            X = X[~ np.isnan(X).any(axis=1)]
            # if there are enough rows for a fit, make an ellipse
            if X.shape[0] > 5:
                el = LsqEllipse().fit(X)
                center, width, height, phi = el.as_parameters()
                center_x = center[0]
                center_y = center[1]
                ellipses.append([center_x, center_y, width, height, phi])
            else:
                ellipses.append([np.nan, np.nan, np.nan, np.nan, np.nan])
            # caudal_edge = [
            #     float(data['Caudal_edge'][row]),
            #     float(data['Caudal_edge.1'][row])
            # ]
            # rostral_edge = [
            #     float(data['Rostral_edge'][row]),
            #     float(data['Rostral_edge.1'][row])
            # ]
            # caudal_edge_ls.append(caudal_edge)
            # rostral_edge_ls.append(rostral_edge)
            # if row % 50 == 0:
            #   print(f'just finished with {row} out of {len(data)-1}', end='\r',flush=True)
        ellipse_df = pd.DataFrame(columns=['center_x', 'center_y', 'width', 'height', 'phi'], data=ellipses)
        a = np.array(ellipse_df['height'][:])
        b = np.array(ellipse_df['width'][:])
        ellipse_size_per_frame = a * b * math.pi
        ellipse_df['ellipse_size'] = ellipse_size_per_frame
        # ellipse_df['rostral_edge'] = rostral_edge_ls
        # ellipse_df['caudal_edge'] = caudal_edge_ls
        print('\n Done')
        return ellipse_df

    def read_dlc_data(self, threshold_to_use=0.999):
        """
        Method to read and analyze the dlc files and fit ellipses to create the le/re ellipses attributes of the block
        """
        generator = self.block_path.glob(r'eye_videos\LE\**\*DLC*.csv')
        p = next(generator)
        self.le_csv = pd.read_csv(p, header=1)
        generator = self.block_path.glob(r'eye_videos\RE\**\*DLC*.csv')
        p = next(generator)
        self.re_csv = pd.read_csv(p, header=1)
        self.le_ellipses = self.eye_tracking_analysis(self.le_csv, threshold_to_use)
        self.re_ellipses = self.eye_tracking_analysis(self.re_csv, threshold_to_use)
        self.le_video_sync_df = self.synced_videos_validated.drop(labels=['Arena_TTL', 'R_eye_TTL'], axis=1)
        for column in list(self.le_ellipses.columns):
            self.le_video_sync_df.insert(loc=len(self.le_video_sync_df.columns), column=column, value=None)
        self.re_video_sync_df = self.synced_videos_validated.drop(labels=['Arena_TTL', 'L_eye_TTL'], axis=1)
        for column in list(self.re_ellipses.columns):
            self.re_video_sync_df.insert(loc=len(self.re_video_sync_df.columns), column=column, value=None)
        print('populating le_video_sync_df')
        for row in tqdm(self.le_video_sync_df.index):
            frame = self.le_video_sync_df.L_eye_TTL[row]
            self.le_video_sync_df.loc[row, 'center_x'] = self.le_ellipses.iloc[frame]['center_x']
            self.le_video_sync_df.loc[row, 'center_y'] = self.le_ellipses.iloc[frame]['center_y']
            self.le_video_sync_df.loc[row, 'width'] = self.le_ellipses.width[frame]
            self.le_video_sync_df.loc[row, 'height'] = self.le_ellipses.height[frame]
            self.le_video_sync_df.loc[row, 'phi'] = self.le_ellipses.phi[frame]
            self.le_video_sync_df.loc[row, 'ellipse_size'] = self.le_ellipses.ellipse_size[frame]
            # le_video_sync_df.at[row, 'rostral_edge'] = le_ellipses.rostral_edge[frame]
            # le_video_sync_df.at[row, 'caudal_edge'] = le_ellipses.caudal_edge[frame]
        print('populating re_video_sync_df')
        for row in tqdm(self.re_video_sync_df.index):
            frame = self.re_video_sync_df.R_eye_TTL[row]
            self.re_video_sync_df.loc[row, 'center_x'] = self.re_ellipses.iloc[frame]['center_x']
            self.re_video_sync_df.loc[row, 'center_y'] = self.re_ellipses.iloc[frame]['center_y']
            self.re_video_sync_df.loc[row, 'width'] = self.re_ellipses.width[frame]
            self.re_video_sync_df.loc[row, 'height'] = self.re_ellipses.height[frame]
            self.re_video_sync_df.loc[row, 'phi'] = self.re_ellipses.phi[frame]
            self.re_video_sync_df.loc[row, 'ellipse_size'] = self.re_ellipses.ellipse_size[frame]
            # re_video_sync_df.at[row, 'rostral_edge'] = re_ellipses.rostral_edge[frame]
            # re_video_sync_df.at[row, 'caudal_edge'] = re_ellipses.caudal_edge[frame]