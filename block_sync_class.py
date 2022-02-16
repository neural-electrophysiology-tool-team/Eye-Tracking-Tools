import glob
import math
import os
import pathlib
import subprocess as sp

import cv2
import numpy as np
import open_ephys.analysis as OEA
import pandas as pd
import scipy.stats as stats
from bokeh.models import HoverTool
from bokeh.palettes import Category20c
from bokeh.plotting import figure, show
from ellipse import LsqEllipse
from lxml import etree
from scipy.signal import medfilt
from tqdm import tqdm

'''
This script defines the BlockSync class which takes all of the relevant data for a given trial and can be utilized
to produce a synchronized dataframe for all video sources to be used for further analysis
'''

"""
Algorithm:
1. Make sure the following files are in existence:
    1. Open ephys parsed events
    2. an arbitrary anchor signal for the timeframe which includes all video sources + electrophysiology
    3. 
     
"""


class BlockSync:
    """
    This class designed to allow parsing and synchronization of the different files acquired in a given experimental
    block. The class expects a certain file system paradigm:
     - Data will be arranged into block folders under date folders under animal folders,
     where each block contains the next structure:
     Animal_call
          ||
          Date(dd_mm_yyyy) >> block_x
                        ||
                Arena_videos -> reptilearn output
                eye_videos -> LE/RE -> video_folder -> video.h264 + .mp4, DLC analysis file.csv, timestamps.csv
                oe_files ->  open ephys output
                analysis -> empty

    """

    def __init__(self, animal_call, experiment_date, block_num, path_to_animal_folder):
        """
            defines the relevant block for analysis

            Parameters
            ----------
            animal_call :  str
                the name of the animal folder

            experiment_date :  str
                the date of the experiment in DD_MM_YYYY format, if None - will assume no date paradigm

            block_num, :  str
                block number to analyze

            path_to_animal_folder :  str
                path to the folder where animal_call folder is located

        """
        self.animal_call = animal_call
        self.experiment_date = experiment_date
        self.block_num = block_num
        self.path_to_animal_folder = path_to_animal_folder
        if experiment_date is not None:
            self.block_path = pathlib.Path(
                rf'{self.path_to_animal_folder}\{self.animal_call}\{self.experiment_date}\block_{self.block_num}')
        else:
            self.block_path = pathlib.Path(
                rf'{self.path_to_animal_folder}\{self.animal_call}\block_{self.block_num}')
        print(f'instantiated block number {self.block_num} at Path: {self.block_path}')
        try:
            self.exp_date_time = os.listdir(fr'{self.block_path}\oe_files')[0]
        except IndexError:
            print(f'block number {self.block_num} does not have open_ephys files')

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
        if (self.block_path / 'analysis' / 'arena_brightness.csv').exists():
            self.arena_brightness_df = pd.read_csv(self.block_path / 'analysis' / 'arena_brightness.csv')
        else:
            self.arena_brightness_df = None
        self.channeldict = {
            5: 'L_eye_TTL',
            6: 'Arena_TTL',
            7: 'Logical ON/OFF',
            8: 'R_eye_TTL'
        }
        p = self.block_path / 'oe_files'
        dirname = os.listdir(p)
        self.oe_dirname = [i for i in dirname if (p / i).is_dir()][0]
        p = self.block_path / 'oe_files' / self.oe_dirname
        dirname = os.listdir(p)
        self.rec_node_dirname = [i for i in dirname if (p / i).is_dir()][0]
        self.oe_path = self.block_path / 'oe_files' / self.oe_dirname / self.rec_node_dirname
        self.settings_xml = self.oe_path / 'settings.xml'
        self.sample_rate = None
        self.sample_rate = None
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
        self.euclidean_speed_per_frame = None
        self.movement_df = None
        self.no_movement_frames = None
        self.left_eye_pupil_speed = None
        self.right_eye_pupil_speed = None
        self.saccade_dict = None
        self.first_oe_timestamp = None

    def __str__(self):
        return str(f'{self.animal_call}, block {self.block_num}, on {self.exp_date_time}')

    def __repr__(self):
        return str(
            f'BlockSync object for animal {self.animal_call} with \n'
            f'block_num {self.block_num} at date {self.exp_date_time}')

    def get_sample_rate(self):
        """
        This is a utility function that gets the sample rate for the block through the settings.xml file under the
        EDITOR branch of the xml
        :return:
        """
        xml_tree = etree.parse(str(self.settings_xml))
        xml_root = xml_tree.getroot()
        for child in xml_root.iter():
            if child.tag == 'EDITOR':
                try:
                    sample_rate = int(float(child.attrib['SampleRateString'][:4]) * 1000)
                except KeyError:
                    continue
        self.sample_rate = sample_rate
        print(f'The sample rate for block {self.block_num} is {sample_rate} Hz')

    def oe_events_to_csv(self):
        """
        This function takes the open ephys events and puts them in a csv file
        :return:
        """
        csv_export_path = self.block_path / 'oe_files' / self.oe_dirname / 'events.csv'
        if not csv_export_path.is_file():
            session = OEA.Session(str(self.oe_path))
            events_df = session.recordings[0].events
            events_df.to_csv(csv_export_path)
            print(f'open ephys events exported to csv file at {csv_export_path}')
        else:
            print('events.csv file already exists')

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
        This method converts and renames the eye tracking videos in the files tree into workable .mp4 files
        """
        eye_vid_path = self.block_path / 'eye_videos'
        print('converting videos...')
        files_to_convert = \
            [file for file in glob.glob(str(eye_vid_path) + r'\**\*.h264', recursive=True) if 'DLC' not in file]
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
        videos_to_inspect = \
            [file for file in glob.glob(str(eye_vid_path) + r'\**\*.mp4', recursive=True) if 'DLC' not in file]
        timestamps_to_inspect = \
            [file for file in glob.glob(str(eye_vid_path) + r'\**\*.csv', recursive=True) if 'DLC' not in file]
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
        self.le_videos = [vid for vid in glob.glob(str(self.block_path) + r'\eye_videos\LE\**\*.mp4') if
                          "DLC" not in vid]
        self.re_videos = [vid for vid in glob.glob(str(self.block_path) + r'\eye_videos\RE\**\*.mp4') if
                          "DLC" not in vid]

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

    def synchronize_arena_timestamps(self, return_dfs=False, export_sync_df=False, get_only_anchor_vid=False):
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

        if get_only_anchor_vid:
            return
        # construct a synchronization dataframe
        self.arena_sync_df = pd.DataFrame(data=[],
                                          columns=self.arena_vidnames,
                                          index=range(len(anchor_vid)))

        # populate the df, starting with the anchor:
        self.arena_sync_df[self.arena_sync_df.columns[anchor_ind]] = range(len(anchor_vid))
        vids_to_sync = list(self.arena_sync_df.drop(axis=1, labels=self.anchor_vid_name).columns)  # CHECK ME !!!!
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
        self.arena_brightness_df if not already available

        Parameters
        ----------
        threshold_value: float
            the threshold to use in order to concentrate on LEDs

        export: binary
            if set to true, will export a dataframe to the analysis folder inside the block directory
        """
        if self.arena_brightness_df is not None:
            print('arena brightness df already exists')
            return
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

    def validate_arena_synchronization_bdf(self):
        if self.arena_bdf is None:
            print('No arena_bdf, run the create_arena_brightness_df method')
        x_axis = range(len(self.arena_bdf.index))
        columns = self.arena_bdf.columns
        bokeh_fig = figure(title=f'Block Number {self.block_num} Arena Video Synchronization Verify',
                           x_axis_label='Frame',
                           y_axis_label='Z_Score',
                           plot_width=1500,
                           plot_height=700
                           )
        color_list = ['orange', 'purple', 'teal', 'green', 'red']
        for ind, video in enumerate(columns):
            bokeh_fig.line(x_axis, self.arena_bdf[video],
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
        channels = np.unique(df['channel'].to_numpy(copy=True))
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

    def fix_timestamps(self):
        """
        This is a utility function which corrects the timestamps in the events.csv file
        :return:
        """
        events_df = pd.read_csv(self.block_path / rf'oe_files\{self.exp_date_time}\events.csv')
        s = pd.Series(((events_df['timestamp'].values - self.first_oe_timestamp) / self.sample_rate) * 1000)
        events_df['timestamp'] = s
        events_df.to_csv(self.block_path / rf'oe_files\{self.exp_date_time}\events.csv')

    def parse_open_ephys_events(self, fix_timestamps=False):
        """
        Method for parsing the Open Ephys events.csv results

        This also defines the correct beginning and ending times of a block (in the OE reference frame)

        """

        # understand the samplerate and the first timestamp
        if self.sample_rate is None:
            self.get_sample_rate()
        session = OEA.Session(str(self.oe_path))
        self.first_oe_timestamp = session.recordings[0].continuous[0].timestamps[0]
        # correct the timestamps
        if fix_timestamps:
            self.fix_timestamps()
        # parse the events of the open-ephys recording
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
                regular_interval = stats.mode(np.diff(ts.values))[0]
        # Now, determine which camera shot its first frame last to define the block start:
        if self.ts_dict['L_eye_TTL'][0] - self.ts_dict['R_eye_TTL'][0] > 0:
            last_to_start = 'L_eye_TTL'
        else:
            last_to_start = 'R_eye_TTL'
        # determine if the arena break in ttls happened before or after the last eye to start
        ttl_breaks = np.where(np.diff(self.ts_dict['Arena_TTL'].values) > regular_interval * 10)
        print(ttl_breaks)
        arena_starts = self.ts_dict['Arena_TTL'][ttl_breaks[0][0]]
        if self.ts_dict[last_to_start][0] < arena_starts:
            self.block_starts = arena_starts
            print(f'The first block frame came from the Arena_video')
        else:
            self.block_starts = self.ts_dict[last_to_start][0]
            print(f'The first block frame came from the {last_to_start}')
        # next, determine which camera shot its last shot first:
        if self.ts_dict['L_eye_TTL'].iloc[-1] - self.ts_dict['R_eye_TTL'].iloc[-1] < 0:
            first_to_end = 'L_eye_TTL'
        else:
            first_to_end = 'R_eye_TTL'
        # determine if the arena break in ttls happened before or after the last eye to start
        arena_ends = self.ts_dict['Arena_TTL'][ttl_breaks[0][1]]
        if self.ts_dict[first_to_end].iloc[-1] < arena_ends:
            self.block_ends = self.ts_dict[first_to_end].iloc[-1]
            print(f'The last frame of the block came from {first_to_end}')
        else:
            self.block_ends = arena_ends
            print('The last frame came from the Arena Video')

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

    def synchronize_block(self, ms=True):
        """
        This method defines the synchronization dataframe for the block, where frames from the different video sources
        are aligned with an anchor signal spanning the synchronized experiment timeframe
        """
        # define the anchor signal
        if ms:
            self.anchor_signal = np.arange(self.block_starts, self.block_ends, 1000 / 60)
            self.anchor_signal = self.anchor_signal[0:len(self.anchor_signal) - 60]
        else:
            self.anchor_signal = np.arange(self.block_starts, self.block_ends, 1 / 60)
            self.anchor_signal = self.anchor_signal[0:len(self.anchor_signal)]
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
                                      column=right_eye_col, value=self.r_eye_values[self.synced_videos[
                f'{right_eye_col}'].values.astype(int)][0:len(self.anchor_signal)])

        self.eye_brightness_df.insert(loc=0,
                                      column=left_eye_col,
                                      value=self.l_eye_values[self.synced_videos[f'{left_eye_col}'].values.astype(int)][
                                            0:len(self.anchor_signal)])

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
            self.validate_arena_synchronization_bdf()
            int_flag = 0
            while int_flag == 0:
                try:
                    arena_blink_ind = int(input('please identify and enter the led blink frame:'))
                    int_flag = 1
                except ValueError:
                    print('That was not a valid number, try again...')
        else:
            arena_blink_ind = suspect_list[0]
        search_range = range(arena_blink_ind - 50, arena_blink_ind + 50)
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
        # self.arena_brightness_df.drop(columns='Unnamed: 0')
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
        if self.arena_sync_df is not None:
            self.arena_sync_df.to_csv(self.block_path / 'analysis/arena_sync_df.csv')

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
        p_arena_sync_df = self.block_path / 'analysis/arena_sync_df.csv'
        if p_arena_sync_df.exists():
            self.arena_sync_df = pd.read_csv(p_arena_sync_df,
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

    def get_image_events_frames(self, arena_col='Arena_TTL'):
        if self.arena_first_ttl_frame is None:
            self.arena_first_ttl_frame = self.synced_videos_validated[arena_col][0]
        self.stim_events = pd.read_csv(self.arena_path / 'events.csv')
        # get internal timestamps for the arena
        anchor_vid_csv = self.anchor_vid_name[:-4] + '.csv'
        self.arena_internal_timestamps = pd.read_csv(self.arena_path / anchor_vid_csv)
        # get image_on_files, image_on_times and image_off_times
        self.image_on_files = self.stim_events.query('event == "show_image"').value
        self.image_on_times = self.stim_events.query('event == "show_image"').time
        self.image_off_times = self.stim_events.query('event == "image_off"').time

        # match between stim times and arena frames
        self.stim_on_arena_frames = []
        for time in self.image_on_times.values:
            frame = self.get_closest_frame(time, self.arena_internal_timestamps) + self.arena_first_ttl_frame
            self.stim_on_arena_frames.append(frame)
        self.stim_off_arena_frames = []
        for time in self.image_off_times:
            frame = self.get_closest_frame(time, self.arena_internal_timestamps) + self.arena_first_ttl_frame
            self.stim_off_arena_frames.append(frame)

    def led_speed_analysis(self, speed_threshold=0.05, plot_graph=False):

        # import the dlc_csvs and collect the data from all videos in a dataframe
        m_csvs = list((self.block_path / 'gaze_vector').glob('*.csv'))
        csv_dict = {}
        # create the csv dict to iterate over
        for i in range(len(m_csvs)):
            csv_dict[f'csv_{i}'] = pd.read_csv(m_csvs[i], header=1)

        # iterate and sort the data from the dlc into location_data
        location_data = {}
        uncertainty_thr = 0.5
        for i in range(len(m_csvs)):
            data = csv_dict[f'csv_{i}']
            # import the dataframe and convert it to floats
            data = data.iloc[1:].apply(pd.to_numeric)
            # sort the pupil elements to x and y, with p as probability
            led_elements = np.array([x for x in data.columns if 'bodyparts' not in x])
            led_xs = data[led_elements[np.arange(0, len(led_elements), 3)]]
            led_ys = data[led_elements[np.arange(1, len(led_elements), 3)]]
            led_ps = data[led_elements[np.arange(2, len(led_elements), 3)]]
            # rename dataframes for masking with p values of bad points:
            led_ps = led_ps.rename(columns=dict(zip(led_ps.columns, led_xs.columns)))
            led_ys = led_ys.rename(columns=dict(zip(led_ys.columns, led_xs.columns)))
            good_points = led_ps > uncertainty_thr
            led_xs = led_xs[good_points]
            led_ys = led_ys[good_points]
            # Data collection into dict
            location_data[f'led_xs_{i}'] = led_xs
            location_data[f'led_ys_{i}'] = led_ys
            location_data[f'led_ps_{i}'] = led_ps

        # convert dict into movement_df for iterative processing
        dict_for_df = {}
        for i in range(len(m_csvs)):
            dfx = location_data[f'led_xs_{i}']
            dfy = location_data[f'led_ys_{i}']
            new_columns_x = dfx.columns.values + f'_x_{i}'
            new_columns_y = dfy.columns.values + f'_y_{i}'
            dict_for_df[f'led_xs_{i}'] = dfx.rename(columns=dict(zip(dfx.columns.values, list(new_columns_x))))
            dict_for_df[f'led_ys_{i}'] = dfy.rename(columns=dict(zip(dfy.columns.values, list(new_columns_y))))
        df_list = [dict_for_df[x] for x in dict_for_df.keys()]
        movement_df = pd.concat(df_list, 1)

        # mark acceptable frames as those that have at least 20 active traces
        good_rows = []
        bad_rows = []
        for row in tqdm(range(0, len(movement_df))):
            count = 0
            for column in movement_df.columns:
                if not np.isnan(movement_df[f'{column}'].iloc[row]):
                    count += 1
            if count > 20:
                good_rows.append(row)
            else:
                bad_rows.append(row)

        # derivative based analysis:
        all_diffs_x = []
        all_diffs_y = []
        for col in movement_df.columns:
            y = movement_df[col].values
            x = list(range(len(movement_df)))
            dydx = np.diff(y) / np.diff(x)
            if 'x' in str(col):
                all_diffs_x.append(dydx)
            elif 'y' in str(col):
                all_diffs_y.append(dydx)
            else:
                print('weird, try again')
        all_diffs_x = np.array(all_diffs_x)
        all_diffs_y = np.array(all_diffs_y)
        all_diffs_dict = {'y': all_diffs_y,
                          'x': all_diffs_x}

        mean_speed_dict = {
            'x': [],
            'y': []
        }
        for key in ['x', 'y']:
            for i in range(len(all_diffs_dict[key][0])):
                # find the non nan values to work with:
                non_nans = []
                for j in range(len(all_diffs_dict[key])):
                    if not np.isnan(all_diffs_dict[key][j][i]):
                        non_nans.append(j)
                if len(non_nans) != 0:
                    mean = np.sum(all_diffs_dict[key][non_nans, i]) / len(non_nans)
                    mean_speed_dict[key].append(mean)
                else:
                    mean_speed_dict[key].append(np.nan)
                    if np.isnan(mean):
                        print('try again')
                        print(i)
                        print(non_nans)
                        break
        # euclidean speed calc
        x_sqrd = np.power(np.array(mean_speed_dict['x']), np.array([2]))
        y_sqrd = np.power(np.array(mean_speed_dict['y']), np.array([2]))
        euclidean_speed = np.sqrt(x_sqrd + y_sqrd)
        self.euclidean_speed_per_frame = euclidean_speed
        self.movement_df = movement_df
        low_speed_segments = euclidean_speed < speed_threshold
        low_speed_segments = medfilt(low_speed_segments, 15)
        self.no_movement_frames = np.argwhere(low_speed_segments)[:, 0] + self.arena_first_ttl_frame - 1
        if plot_graph:
            bokeh_fig = figure(title=f'speed movement analysis {print(self)}',
                               x_axis_label='Linear Frames',
                               y_axis_label='location in frame (colors) / speed(black) - Z score',
                               plot_width=1500,
                               plot_height=700
                               )
            mean_speed = euclidean_speed
            bokeh_fig.line(
                list(range(len(mean_speed))),
                np.abs((mean_speed - np.mean(mean_speed[~np.isnan(mean_speed)])) / np.std(
                    np.array(mean_speed)[~np.isnan(mean_speed)])),
                legend_label='speed',
                line_width=0.3,
                line_color='black')
            for num, col in enumerate(movement_df.columns):
                if 'y' in col:
                    try:
                        color = Category20c[20][num]
                    except IndexError:
                        color = 'green'
                elif 'x' in col:
                    try:
                        color = Category20c[20][-num]
                    except IndexError:
                        color = 'red'
                bokeh_fig.line(movement_df[col].index,
                               np.abs((movement_df[col].values - movement_df[col].mean()) / movement_df[col].std()),
                               legend_label=col,
                               line_width=1.5,
                               line_color=color)

            bokeh_fig.vbar(x=self.no_movement_frames,
                           width=1,
                           bottom=-4,
                           top=-1,
                           alpha=0.15,
                           color='green')
            show(bokeh_fig)

    def pupil_speed_graph(self,
                          plot_r=True,
                          plot_l=True,
                          plot_l_position=True,
                          plot_r_position=True,
                          smooth_velocities=False,
                          sf=3):
        lx = self.le_video_sync_df.center_x.values
        ly = self.le_video_sync_df.center_y.values
        rx = self.re_video_sync_df.center_x.values
        ry = self.re_video_sync_df.center_y.values
        if smooth_velocities:
            diff_dict = {
                'lx': np.array([lx[x + sf] - lx[x] for x in range(len(lx) - sf)]),
                'ly': np.array([ly[x + sf] - ly[x] for x in range(len(ly) - sf)]),
                'rx': np.array([rx[x + sf] - rx[x] for x in range(len(rx) - sf)]),
                'ry': np.array([ry[x + sf] - ry[x] for x in range(len(ry) - sf)])
            }
        else:
            diff_dict = {
                'lx': np.diff(lx, prepend=1).astype(float),
                'ly': np.diff(ly, prepend=1).astype(float),
                'rx': np.diff(rx, prepend=1).astype(float),
                'ry': np.diff(ry, prepend=1).astype(float),
            }
        l_e_dist = np.sqrt((diff_dict['lx'] ** 2) + (diff_dict['ly'] ** 2))
        r_e_dist = np.sqrt((diff_dict['rx'] ** 2) + (diff_dict['ry'] ** 2))
        self.left_eye_pupil_speed = l_e_dist
        self.right_eye_pupil_speed = r_e_dist
        bokeh_fig = figure(title=f'Pupil center speed for {self.__str__()}',
                           x_axis_label='Linear Frames',
                           y_axis_label='euclidean speed',
                           plot_width=1500,
                           plot_height=700)
        x_axis = self.synced_videos_validated.Arena_TTL.values - self.arena_first_ttl_frame
        if plot_l:
            bokeh_fig.line(x_axis,
                           l_e_dist[0:len(x_axis)],
                           legend_label='Left Eye Speed',
                           line_width=1.5,
                           line_color='blue')
        if plot_r:
            bokeh_fig.line(x_axis,
                           -r_e_dist[0:len(x_axis)],
                           legend_label='Inverse Right Eye Speed',
                           line_width=1.5,
                           line_color='red')
        if plot_l_position:
            bokeh_fig.line(x_axis,
                           ((self.le_video_sync_df.center_x - self.le_video_sync_df.center_x.mean()) /
                            self.le_video_sync_df.center_x.std())[0:len(x_axis)],
                           legend_label='left_pupil_position_X',
                           line_width=1.5,
                           line_color='purple')

            bokeh_fig.line(x_axis,
                           ((self.le_video_sync_df.center_y - self.le_video_sync_df.center_y.mean()) /
                            self.le_video_sync_df.center_y.std())[0:len(x_axis)],
                           legend_label='left_pupil_position_y',
                           line_width=1.5,
                           line_color='lavender')
        if plot_r_position:
            bokeh_fig.line(x_axis,
                           ((self.re_video_sync_df.center_x - self.re_video_sync_df.center_x.mean()) /
                            self.re_video_sync_df.center_x.std())[0:len(x_axis)],
                           legend_label='right_pupil_position_X',
                           line_width=1.5,
                           line_color='firebrick')

            bokeh_fig.line(x_axis,
                           ((self.re_video_sync_df.center_y - self.re_video_sync_df.center_y.mean()) /
                            self.re_video_sync_df.center_y.std())[0:len(x_axis)],
                           legend_label='right_pupil_position_y',
                           line_width=1.5,
                           line_color='pink')

        bokeh_fig.vbar(x=self.no_movement_frames - self.arena_first_ttl_frame,
                       width=1,
                       bottom=-4,
                       top=-1,
                       alpha=0.2,
                       color='green')
        show(bokeh_fig)

    def block_plot(self, plot_saccade_locs=False, saccade_frames_r=None, saccade_frames_l=None, plot_stim_on_off=False):

        # Extract the data columns and calculate Z-score across all blocks
        le_ellipses_z = (
                                    self.le_video_sync_df.ellipse_size - self.le_video_sync_df.ellipse_size.mean()) / self.le_video_sync_df.ellipse_size.std()
        re_ellipses_z = (
                                    self.re_video_sync_df.ellipse_size - self.re_video_sync_df.ellipse_size.mean()) / self.re_video_sync_df.ellipse_size.std()
        le_x_zscores = (self.le_video_sync_df.center_x - np.mean(
            self.le_video_sync_df.center_x)) / self.le_video_sync_df.center_x.std()
        le_y_zscores = (self.le_video_sync_df.center_y - np.mean(
            self.le_video_sync_df.center_y)) / self.le_video_sync_df.center_y.std()
        re_x_zscores = (self.re_video_sync_df.center_x - np.mean(
            self.re_video_sync_df.center_x)) / self.re_video_sync_df.center_x.std()
        re_y_zscores = (self.re_video_sync_df.center_y - np.mean(
            self.re_video_sync_df.center_y)) / self.re_video_sync_df.center_y.std()
        x_axis = range(len(le_ellipses_z))
        # plot everything
        bokeh_fig = figure(title=f'Pupil combined metrics block {self.block_num}',
                           x_axis_label='Linear Frames',
                           y_axis_label=' Z scores',
                           plot_width=1500,
                           plot_height=700
                           )
        bokeh_fig.add_tools(HoverTool())
        bokeh_fig.line(x_axis, le_ellipses_z + 7, legend_label='Left Eye diameter', line_width=1.5, line_color='blue')
        bokeh_fig.line(x_axis, le_x_zscores + 14, legend_label='Left Eye x position', line_width=1, line_color='cyan')
        bokeh_fig.line(x_axis, le_y_zscores, legend_label='Left Eye y position', line_width=1, line_color='green')
        bokeh_fig.line(x_axis, re_ellipses_z + 7, legend_label='Right Eye diameter', line_width=1.5, line_color='red')
        bokeh_fig.line(x_axis, re_x_zscores + 14, legend_label='Right Eye x position', line_width=1,
                       line_color='orange')
        bokeh_fig.line(x_axis, re_y_zscores, legend_label='Right Eye y position', line_width=1, line_color='pink')
        if plot_stim_on_off:
            bokeh_fig.vbar(x=self.stim_on_arena_frames, width=1, bottom=-4, top=17, alpha=0.5, color='green')
            bokeh_fig.vbar(x=self.stim_off_arena_frames, width=1, bottom=-4, top=17, alpha=0.5, color='red')
        if plot_saccade_locs:
            bokeh_fig.vbar(x=saccade_frames_l, width=1, bottom=-4, top=-1, alpha=0.8, color='purple')
            bokeh_fig.vbar(x=saccade_frames_r, width=1, bottom=-4, top=-1, alpha=0.8, color='brown')
        show(bokeh_fig)

    def collect_saccade_events_v3(self, threshold=2):
        flag = 0
        while flag == 0:
            l_saccades = np.argwhere(self.left_eye_pupil_speed > threshold)
            l_saccades = medfilt(l_saccades[:, 0], 5)
            r_saccades = np.argwhere(self.right_eye_pupil_speed > threshold)
            r_saccades = medfilt(r_saccades[:, 0], 5)
            self.block_plot(plot_saccade_locs=True, saccade_frames_r=r_saccades, saccade_frames_l=l_saccades)
            answer = input('look at the graph - is the threshold for speed okay? y/n or abort')
            if answer == 'y':
                flag = 1
            elif answer == 'n':
                threshold = float(input('insert another threshold value to try'))
            elif answer == 'abort':
                print('giving up on the block')
                return None
            else:
                print('bad input, going around again...')

        # use detected saccades locations for saccade characterisation
        # find consequitive frames and understand how many saccades there are in the trace for each axis:
        s_dict = {}
        for eye in ['l', 'r']:
            s_start = []
            s_mid = []
            s_end = []
            state = 'start'
            if eye == 'l':
                for i in range(len(l_saccades) - 1):
                    if state == 'start':
                        s_start.append(l_saccades[i])
                        state = 'mid'
                        continue
                    if l_saccades[i] + 1 == l_saccades[i + 1]:
                        s_mid.append(l_saccades[i])
                        state = 'mid'
                    else:
                        s_end.append(l_saccades[i])
                        state = 'start'
                s_end.append(l_saccades[-1])

            if eye == 'r':
                for i in range(len(r_saccades) - 1):
                    if state == 'start':
                        s_start.append(r_saccades[i])
                        state = 'mid'
                        continue
                    if r_saccades[i] + 1 == r_saccades[i + 1]:
                        s_mid.append(r_saccades[i])
                        state = 'mid'
                    else:
                        s_end.append(r_saccades[i])
                        state = 'start'
                s_end.append(r_saccades[-1])

            s_dict[f'{eye}_start'] = s_start
            s_dict[f'{eye}_ends'] = s_end
            if len(s_end) == len(s_start):
                s_dict[f'{eye}_len'] = np.array(s_end) - np.array(s_start)
            elif len(s_end) == len(s_start) + 1:
                if s_end[-2] == s_end[-1]:
                    s_end.pop(-1)
                    s_dict[f'{eye}_len'] = np.array(s_end) - np.array(s_start)
            else:
                print(f'there is a length problem where the saccades have {len(s_end)} ends and {len(s_start)} starts')
                return None
            # choose saccades with sufficient length
            s_dict[f'{eye}_start'] = np.array(s_dict[f'{eye}_start'])[list(np.argwhere(s_dict[f'{eye}_len'] > 5)[:, 0])]
            s_dict[f'{eye}_ends'] = np.array(s_dict[f'{eye}_ends'])[list(np.argwhere(s_dict[f'{eye}_len'] > 5)[:, 0])]

        # collect all saccade x y info within a 25 frame window
        for eye in ['l', 'r']:
            s_dict[f'{eye}_y_loc'] = []
            s_dict[f'{eye}_x_loc'] = []
            for i in range(len(s_dict[f'{eye}_start'])):
                x_axis = range(int(s_dict[f'{eye}_start'][i] - 25), int(s_dict[f'{eye}_start'][i] + 25))
                if eye == 'l':
                    y_data = self.le_video_sync_df.center_y[x_axis]
                    x_data = self.le_video_sync_df.center_x[x_axis]
                    s_dict[f'{eye}_y_loc'].append(y_data)
                    s_dict[f'{eye}_x_loc'].append(x_data)
                elif eye == 'r':
                    y_data = self.re_video_sync_df.center_y[x_axis]
                    x_data = self.re_video_sync_df.center_x[x_axis]
                    s_dict[f'{eye}_y_loc'].append(y_data)
                    s_dict[f'{eye}_x_loc'].append(x_data)
        saccade_dict = {
            'l': [],
            'r': []
        }

        # compute all saccades euclidean distance
        magnitude_dict = {'l': [],
                          'r': []}
        for eye in ['l', 'r']:
            for i in range(len(s_dict[f'{eye}_start'])):
                starting_pos_x = s_dict[f'{eye}_x_loc'][i].iloc[25]
                starting_pos_y = s_dict[f'{eye}_y_loc'][i].iloc[25]
                s_frames = s_dict[f'{eye}_y_loc'][i].index.values
                r = []
                for frame in s_frames:
                    a = (s_dict[f'{eye}_y_loc'][i].loc[frame] - starting_pos_y) ** 2
                    b = (s_dict[f'{eye}_x_loc'][i].loc[frame] - starting_pos_x) ** 2
                    r.append(np.sqrt(a + b))
                r_min = np.min(r)
                r_max = np.max(r)
                magnitude_dict[f'{eye}'].append(r_max - r_min)
                r_normalized = [(x - r_min) / (r_max - r_min) for x in r]
                saccade_dict[f'{eye}'].append(r_normalized)
        self.saccade_dict = saccade_dict

        # create the saccade df and slowly fill it up
        saccade_list = []
        for eye in ['r', 'l']:
            for row in range(len(s_dict[f'{eye}_start'])):
                entry = {
                    'starts': s_dict[f'{eye}_start'][row],
                    'ends': s_dict[f'{eye}_ends'][row],
                    'magnitude': magnitude_dict[f'{eye}'][row],
                    'velocity': magnitude_dict[f'{eye}'][row] / len(self.saccade_dict[f'{eye}'][row]),
                    'head_movement': None,
                    'dynamics': self.saccade_dict[f'{eye}'][row]
                }
                saccade_list.append(entry)

        self.saccade_list = saccade_list

    def collect_saccade_events_full_dynamics(self, threshold=2):
        flag = 0
        while flag == 0:
            l_saccades = np.argwhere(self.left_eye_pupil_speed > threshold)
            l_saccades = medfilt(l_saccades[:, 0], 5)
            r_saccades = np.argwhere(self.right_eye_pupil_speed > threshold)
            r_saccades = medfilt(r_saccades[:, 0], 5)
            self.block_plot(plot_saccade_locs=True, saccade_frames_r=r_saccades, saccade_frames_l=l_saccades)
            answer = input('look at the graph - is the threshold for speed okay? y/n or abort')
            if answer == 'y':
                flag = 1
            elif answer == 'n':
                threshold = float(input('insert another threshold value to try'))
            elif answer == 'abort':
                print('giving up on the block')
                return None
            else:
                print('bad input, going around again...')

        # use detected saccades locations for saccade characterisation
        # find consequitive frames and understand how many saccades there are in the trace for each axis:
        s_dict = {}
        for eye in ['l', 'r']:
            s_start = []
            s_mid = []
            s_end = []
            state = 'start'
            if eye == 'l':
                for i in range(len(l_saccades) - 1):
                    if state == 'start':
                        s_start.append(l_saccades[i])
                        state = 'mid'
                        continue
                    if l_saccades[i] + 1 == l_saccades[i + 1]:
                        s_mid.append(l_saccades[i])
                        state = 'mid'
                    else:
                        s_end.append(l_saccades[i])
                        state = 'start'
                s_end.append(l_saccades[-1])

            if eye == 'r':
                for i in range(len(r_saccades) - 1):
                    if state == 'start':
                        s_start.append(r_saccades[i])
                        state = 'mid'
                        continue
                    if r_saccades[i] + 1 == r_saccades[i + 1]:
                        s_mid.append(r_saccades[i])
                        state = 'mid'
                    else:
                        s_end.append(r_saccades[i])
                        state = 'start'
                s_end.append(r_saccades[-1])

            s_dict[f'{eye}_start'] = s_start
            s_dict[f'{eye}_ends'] = s_end
            if len(s_end) == len(s_start):
                s_dict[f'{eye}_len'] = np.array(s_end) - np.array(s_start)
            elif len(s_end) == len(s_start) + 1:
                if s_end[-2] == s_end[-1]:
                    s_end.pop(-1)
                    s_dict[f'{eye}_len'] = np.array(s_end) - np.array(s_start)
            else:
                print(f'there is a length problem where the saccades have {len(s_end)} ends and {len(s_start)} starts')
                return None
            # choose saccades with sufficient length
            s_dict[f'{eye}_start'] = np.array(s_dict[f'{eye}_start'])[list(np.argwhere(s_dict[f'{eye}_len'] > 5)[:, 0])]
            s_dict[f'{eye}_ends'] = np.array(s_dict[f'{eye}_ends'])[list(np.argwhere(s_dict[f'{eye}_len'] > 5)[:, 0])]

        # collect all saccade x y info within a 25 frame window
        velocity_dict = {
            'l_velocity': [],
            'r_velocity': []
        }
        for eye in ['l', 'r']:
            s_dict[f'{eye}_y_loc'] = []
            s_dict[f'{eye}_x_loc'] = []
            for i in range(len(s_dict[f'{eye}_start'])):
                try:
                    x_axis = range(int(s_dict[f'{eye}_start'][i] - 25), int(s_dict[f'{eye}_start'][i] + 25))
                except KeyError:
                    continue
                if eye == 'l':
                    try:
                        y_data = self.le_video_sync_df.center_y[x_axis]
                        x_data = self.le_video_sync_df.center_x[x_axis]
                    except KeyError:
                        continue
                    s_dict[f'{eye}_y_loc'].append(y_data)
                    s_dict[f'{eye}_x_loc'].append(x_data)
                    velocity_dict[f'{eye}_velocity'].append(self.left_eye_pupil_speed[x_axis])
                elif eye == 'r':
                    try:
                        y_data = self.re_video_sync_df.center_y[x_axis]
                        x_data = self.re_video_sync_df.center_x[x_axis]
                    except KeyError:
                        continue
                    s_dict[f'{eye}_y_loc'].append(y_data)
                    s_dict[f'{eye}_x_loc'].append(x_data)
                    velocity_dict[f'{eye}_velocity'].append(self.right_eye_pupil_speed[x_axis])
        saccade_dict = {
            'l': [],
            'r': []
        }

        # compute all saccades euclidean distance
        magnitude_dict = {'l': [],
                          'r': []}
        for eye in ['l', 'r']:
            for i in range(len(s_dict[f'{eye}_start'])):
                try:
                    starting_pos_x = s_dict[f'{eye}_x_loc'][i].iloc[25]
                    starting_pos_y = s_dict[f'{eye}_y_loc'][i].iloc[25]
                    s_frames = s_dict[f'{eye}_y_loc'][i].index.values
                except IndexError:
                    continue
                r = []
                for frame in s_frames:
                    a = (s_dict[f'{eye}_y_loc'][i].loc[frame] - starting_pos_y) ** 2
                    b = (s_dict[f'{eye}_x_loc'][i].loc[frame] - starting_pos_x) ** 2
                    r.append(np.sqrt(a + b))
                # Normalize for before (b) and saccade (s)
                rs_min = np.min(r[25:])
                rs_max = np.max(r[25:])
                rb_min = np.min(r[0:25])
                rb_max = np.max(r[0:25])
                magnitude_dict[f'{eye}'].append(rs_max - rs_min)
                rs_normalized = [(x - rs_min) / (rs_max - rs_min) for x in r[25:]]
                rb_normalized = [((x - rb_min) / (rb_max - rb_min)) for x in r[0:25]]
                r_normalized = rb_normalized + rs_normalized
                saccade_dict[f'{eye}'].append(r_normalized)

        self.saccade_dict = saccade_dict

        # create the saccade df and slowly fill it up
        saccade_df = pd.DataFrame(data=None,
                                  columns=['starts', 'ends', 'magnitude', 'velocity', 'head_movements', 'r_dynamics'])
        saccade_list = []
        for eye in ['r', 'l']:
            for row in range(len(s_dict[f'{eye}_start'])):
                try:
                    entry = {
                        'starts': s_dict[f'{eye}_start'][row],
                        'ends': s_dict[f'{eye}_ends'][row],
                        'magnitude': magnitude_dict[f'{eye}'][row],
                        'velocity': velocity_dict[f'{eye}_velocity'][row],
                        'head_movement': None,
                        'r_dynamics': self.saccade_dict[f'{eye}'][row],
                        'x_dynamics': s_dict[f'{eye}_x_loc'][row],
                        'y_dynamics': s_dict[f'{eye}_y_loc'][row]
                    }
                except IndexError:
                    continue
                saccade_list.append(entry)

        self.saccade_list = saccade_list

    def create_synced_video(self, filename, r_eye_vid, l_eye_vid, arena_vid1, arena_vid2, start_time, end_time,
                            frmt='H264', overlay_frame_numbers=False):
        """
        This function takes as input the names of 4 videos and concatenates them into a unified video
        The block has to be correctly synchronized in order to produce the video
        :param filename: name of the output file to save
        :param r_eye_vid: Top Right Vid
        :param l_eye_vid: Top Left Vid
        :param arena_vid1: Bottom Right Vid
        :param arena_vid2: Bottom Left Vid
        :param start_time: in seconds
        :param end_time: in seconds
        :param format: H264 by default
        :param overlay_frame_numbers:   If true prints the frame numbers used for each frame on the video,
                                        defaults to false
        :return:
        """
        re_cap = cv2.VideoCapture(r_eye_vid)
        le_cap = cv2.VideoCapture(l_eye_vid)
        ar1_cap = cv2.VideoCapture(arena_vid1)
        ar2_cap = cv2.VideoCapture(arena_vid2)
        timeseries = self.synced_videos_validated.query('Time>@start_time & Time<@end_time')
        timeseries.Arena_TTL = timeseries.Arena_TTL - self.arena_first_ttl_frame
        p_arena_frame = int(timeseries.iloc[1]['Arena_TTL'])
        p_l_eye_frame = int(timeseries.iloc[1]['L_eye_TTL'])
        p_r_eye_frame = int(timeseries.iloc[1]['R_eye_TTL'])
        anchor = 0
        fourcc = cv2.VideoWriter_fourcc(*frmt)
        out = cv2.VideoWriter(str(self.block_path / str(filename + '.mp4')), fourcc, 60.0, (640 * 2, 480 * 2))
        try:
            while ar1_cap.isOpened():
                arena_frame = int(timeseries.iloc[anchor]['Arena_TTL'])
                r_eye_frame = int(timeseries.iloc[anchor]['R_eye_TTL'])
                l_eye_frame = int(timeseries.iloc[anchor]['L_eye_TTL'])

                # Arena Frame 1
                if arena_frame != p_arena_frame + 1:
                    ar1_cap.set(1, arena_frame)
                ar1_ret, ar1_frame = ar1_cap.read()
                ar1_frame = cv2.cvtColor(ar1_frame, cv2.COLOR_BGR2GRAY)
                ar1_frame = cv2.resize(ar1_frame, (640, 480))

                # Arena Frame 2
                if arena_frame != p_arena_frame + 1:
                    ar2_cap.set(1, arena_frame)
                ar2_ret, ar2_frame = ar2_cap.read()
                ar2_frame = cv2.cvtColor(ar2_frame, cv2.COLOR_BGR2GRAY)
                ar2_frame = cv2.resize(ar2_frame, (640, 480))
                p_arena_frame = arena_frame

                # Left Eye Frame
                if l_eye_frame != p_l_eye_frame + 1:
                    le_cap.set(1, l_eye_frame)
                le_ret, le_frame = le_cap.read()
                le_frame = cv2.cvtColor(le_frame, cv2.COLOR_BGR2GRAY)
                le_frame = cv2.resize(le_frame, (640, 480))
                le_frame = cv2.flip(le_frame, 0)
                p_l_eye_frame = l_eye_frame

                # Right Eye Frame
                if r_eye_frame != p_r_eye_frame + 1:
                    re_cap.set(1, r_eye_frame)
                re_ret, re_frame = re_cap.read()
                re_frame = cv2.cvtColor(re_frame, cv2.COLOR_BGR2GRAY)
                re_frame = cv2.resize(re_frame, (640, 480))
                re_frame = cv2.flip(re_frame, 0)
                p_r_eye_frame = r_eye_frame

                eye_concat = np.hstack((le_frame, re_frame))
                ar_concat = np.hstack((ar1_frame, ar2_frame))
                vconcat = np.vstack((eye_concat, ar_concat))
                if overlay_frame_numbers:
                    framescounter = f'anchor={anchor}, R = {r_eye_frame} L= {l_eye_frame}, A = {arena_frame}'
                    cv2.putText(vconcat, framescounter, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                                cv2.LINE_AA)
                out.write(vconcat)
                anchor += 1
                print(f'writing video frame {anchor} out of {len(timeseries)} ', end='\r', flush=True)
                if anchor > len(timeseries) - 1:
                    break
        except Exception:
            print(f'Encountered a problem with frame {anchor}, stopping concatenation')
        finally:
            ar1_cap.release()
            ar2_cap.release()
            le_cap.release()
            re_cap.release()
            out.release()
            cv2.destroyAllWindows()
            print('\n')
            print('Process Finished')
