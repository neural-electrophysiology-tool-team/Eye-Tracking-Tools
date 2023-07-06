import glob
import math
import os
import pathlib
import subprocess as sp

import cv2
import numpy as np
import open_ephys.analysis as oea
import pandas as pd
import scipy.stats as stats
from bokeh.io import output as b_output
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
from ellipse import LsqEllipse
from lxml import etree
from scipy import signal
from tqdm import tqdm

'''
This script defines the BlockSync class which takes all of the relevant data for a given trial and can be utilized
to produce a synchronized dataframe for all video sources to be used for further analysis
'''


# noinspection SpellCheckingInspection
class BlockSync:
    """
    This class designed to allow parsing and synchronization of the different files acquired in a given experimental
    block. The class expects a certain file system paradigm:
     - Data will be arranged into block folders under date folders under animal folders,
     where each block contains the next structure:
     Animal_call
          ||
          Date(yyyy_mm_dd) >> block_x
                        ||
                Arena_videos -> reptilearn output
                eye_videos -> LE/RE -> video_folder -> video.h264 + .mp4, DLC analysis file.csv, timestamps.csv
                oe_files ->  open ephys output
                analysis -> empty

    """

    def __init__(self, animal_call, experiment_date, block_num, path_to_animal_folder, channeldict=None, regev=False):
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

            channeldict :  dict
                a dictionary binding the I/O board inputs to specific channel names
                (should ALWAYS correspond with the default naming scheme)


        """
        self.animal_call = animal_call
        self.experiment_date = experiment_date
        self.block_num = block_num
        self.path_to_animal_folder = pathlib.Path(path_to_animal_folder)
        if experiment_date is not None:
            self.block_path = pathlib.Path(
                self.path_to_animal_folder) / self.animal_call / self.experiment_date / ('block_' + self.block_num)
        else:
            self.block_path = pathlib.Path(
                self.path_to_animal_folder) / self.animal_call / ('block_' + self.block_num)
        print(f'instantiated block number {self.block_num} at Path: {self.block_path}')
        try:
            dir_to_check = self.block_path / "oe_files"
            self.exp_date_time = os.listdir(dir_to_check)[0]
        except IndexError:
            print(f'block number {self.block_num} does not have open_ephys files')

        if regev:
            self.arena_path = self.block_path / 'arena_videos' / 'videos'
        else:
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
        self.analysis_path = self.block_path / 'analysis'
        self.l_e_path = self.block_path / 'eye_videos' / 'LE'
        try:
            self.l_e_path = self.l_e_path / os.listdir(self.l_e_path)[0]
        except IndexError:
            print('No left eye videos to work with')
        self.r_e_path = self.block_path / 'eye_videos' / 'RE'
        try:
            self.r_e_path = self.r_e_path / os.listdir(self.r_e_path)[0]
        except IndexError:
            print('No right eye videos to work with')
        if (self.analysis_path / 'arena_brightness.csv').exists():
            self.arena_brightness_df = pd.read_csv(self.analysis_path / 'arena_brightness.csv')
            if 'Unnamed: 0' in self.arena_brightness_df.columns:
                self.arena_brightness_df = self.arena_brightness_df.drop(axis=1, labels='Unnamed: 0')
        else:
            self.arena_brightness_df = None
        if channeldict is None:
            self.channeldict = {
                4: 'LED_driver',
                5: 'L_eye_TTL',
                6: 'Arena_TTL',
                7: 'Logical ON/OFF',
                8: 'R_eye_TTL'
            }
        else:
            self.channeldict = channeldict
        p = self.block_path / 'oe_files'
        dirname = os.listdir(p)
        try:
            self.oe_dirname = [i for i in dirname if (p / i).is_dir()][0]
            p = self.block_path / 'oe_files' / self.oe_dirname
            dirname = os.listdir(p)
        except IndexError:
            print('No open ephys files here!!!!')
        try:
            self.rec_node_dirname = [i for i in dirname if (p / i).is_dir()][0]
            self.oe_path = self.block_path / 'oe_files' / self.oe_dirname / self.rec_node_dirname
            self.settings_xml = self.oe_path / 'settings.xml'
            self.sample_rate = self.get_sample_rate()
        except IndexError:
            print('No open ephys record node here!!!')
        self.oe_events = None
        self.ts_dict = None
        self.block_starts = None
        self.block_ends = None
        self.block_length = None
        self.blocksync_df = None
        self.synced_videos = None
        self.accuracy_report = None
        self.anchor_signal = None
        self.le_frame_val_list = None
        self.re_frame_val_list = None
        self.eye_brightness_df = None
        self.l_eye_values = None
        self.r_eye_values = None
        self.arena_vid_first_t = None
        self.arena_vid_last_t = None
        self.r_vid_first_t = None
        self.r_vid_last_t = None
        self.l_vid_first_t = None
        self.l_vid_last_t = None
        self.synced_videos_validated = None
        self.le_csv = None
        self.re_csv = None
        self.le_ellipses = None
        self.re_ellipses = None
        self.euclidean_speed_per_frame = None
        self.movement_df = None
        self.no_movement_frames = None
        self.saccade_dict = None
        self.eye_diff_list = None
        self.le_df = None
        self.re_df = None
        self.lag_direction = None
        self.l_e_speed = None
        self.r_e_speed = None
        self.ms_axis = None
        self.r_saccades = None
        self.l_saccades = None
        self.manual_sync_df = None
        self.r_saccades_chunked = None
        self.l_saccades_chunked = None
        self.L_pix_size = None
        self.R_pix_size = None
        self.eye_diff_mode = None

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
        sample_rate = None
        try:
            xml_tree = etree.parse(str(self.settings_xml))
            xml_root = xml_tree.getroot()
            for child in xml_root.iter():
                if child.tag == 'EDITOR':
                    try:
                        sample_rate = int(float(child.attrib['SampleRateString'][:4]) * 1000)
                    except KeyError:
                        continue
            if sample_rate is not None:
                print(f'Found the sample rate for block {self.block_num} in the xml file, it is {sample_rate} Hz')
            else:
                print(f'could not find the sample rate for block_{self.block_num} in the xml file, '
                      f'looking for it in the first recording...')
                sample_rate = self.get_sample_rate_cont()
        except OSError:
            print('could not find the sample rate in the xml file due to error, will '
                  'look in the cont file of the first recording...')
            sample_rate = self.get_sample_rate_cont()

        finally:
            if sample_rate is not None:
                return sample_rate
            else:
                print('faild to find the sample, rate - please enter it manually')
                sample_rate = input('sample_rate = ?')
                return sample_rate

    def get_sample_rate_cont(self):
        """
        This is a function that determines the sample rate of a block via the first .continuous file in the oe_folder
        :return: sample rate if found one
        """
        file_name = sorted([i for i in os.listdir(self.oe_path) if '.continuous' in i])[0]
        file_path = self.oe_path / file_name
        f = open(file_path, 'rb')
        b = f.readlines(1024)
        sample_rate = None
        for i in b:
            if 'sampleRate' in str(i):
                # print(str(i)[9])
                start_position = str(i).find('=') + 2
                sample_rate = int((str(i)[start_position:-4]))
        f.close()
        if sample_rate is not None:
            print(f'found the sample rate, it is {sample_rate}')
            return sample_rate
        else:
            print('could not find the sample rate')
            return None

    def oe_events_to_csv(self):
        """
        This method takes the open ephys events and puts them in a csv file

        """
        csv_export_path = self.block_path / 'oe_files' / self.oe_dirname / 'events.csv'
        if not csv_export_path.is_file():
            session = oea.Session(str(self.oe_path))
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
        print('handling arena files')
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
        ONLY WORKS ON WINDOWS MACHINES WITH MP4BOX INSTALLED AS A COMMAND LINE MODULE
        """
        print('handling eye video files')
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
    def oe_events_parser(open_ephys_csv_path, channel_names, arena_channel_name='Arena_TTL', export_path=None):
        """

        :param open_ephys_csv_path: The path to an open ephys analysis tools exported csv
        :param channel_names: a dictionary of the form -
                        { 1 : 'channel name' (L_eye_camera)
                          2 : 'channel name' (Arena_TTL)
                          etc..
                        }
        :param export_path: default None, if a path is specified a csv file will be saved
        :param arena_channel_name: the name in channel names which correponds with the arena TTLs
        :returns open_ephys_events: a pandas DataFrame object where each column has the ON events of one channel
                                    and has a title from channel_names
        :returns open_ephys_off_events: same but for the OFF states (only important for the logical start-stop signal)

        """

        # Infer the active channels:
        df = pd.read_csv(open_ephys_csv_path)
        channels = np.unique(df['channel'].to_numpy(copy=True))
        df_onstate = df[df['state'] == 1]  # cut the df to represent only rising edges
        ls = []
        for chan in channels:  # extract a pandas series of the ON stats timestamps for each channel
            sname = channel_names[chan]
            s = pd.Series(df_onstate['timestamp'][df_onstate['channel'] == chan], name=sname)
            # If this is the arena channel we need to collect the first and last frames which correspond with
            # the video itsef (as TTLs are always being transmitted and a pause is expected before the video starts
            if sname == arena_channel_name:
                diff_series = np.diff(s)
                diff_mode = stats.mode(diff_series)[0][0]
                arena_start_stop = np.where(diff_series > 10 * diff_mode)[0]
                if len(arena_start_stop) != 2:
                    raise ValueError(f'there is some kind of problem because there should be 2 breaks in the arena TTLs'
                          f'and there are {len(arena_start_stop)}')

                else:
                    print(f'the arena TTLs are signaling start and stop positions at {arena_start_stop}')
                    arena_start_timestamp = s.iloc[arena_start_stop[0] + 1]
                    print(f'arena first frame timestamp: {arena_start_timestamp}')
                    arena_end_timestamp = s.iloc[arena_start_stop[1]]
                    print(f'arena end frame timestamp: {arena_end_timestamp}')
            else:
                print(f'{sname} was not identified as {arena_channel_name}')
            # create a counter for every rising edge - these should match video frames
            s_counter = pd.Series(data=np.arange(len(s), dtype='Int32'), index=s.index.values, name=sname+'_frame')
            ls.append(s)
            ls.append(s_counter)

        # concatenate all channels into a dataframe with open-ephys compatible timestamps
        open_ephys_events = pd.concat(ls, axis=1)
        # use arena start_stop to clean TTLs counted before video starts and after video ends
        open_ephys_events[f'{arena_channel_name}_frame'] = open_ephys_events[f'{arena_channel_name}_frame'] - (
                    arena_start_stop[0] + 1)
        open_ephys_events[f'{arena_channel_name}_frame'][open_ephys_events[f'{arena_channel_name}_frame'] < 0] = np.nan
        open_ephys_events[f'{arena_channel_name}_frame'][
            open_ephys_events[f'{arena_channel_name}'] > arena_end_timestamp] = np.nan

        if export_path is not None:
            if export_path not in os.listdir(str(open_ephys_csv_path).split('events.csv')[0][:-1]):
                open_ephys_events.to_csv(export_path)

        return open_ephys_events, arena_start_timestamp, arena_end_timestamp

    def parse_open_ephys_events(self):
        """
        Gets the sample rate from the settings.xml file
        Creates the parsed_events.csv file
        finds the first and last frame timestamps for each video source

        """
        print('running parse_open_ephys_events...')
        # First, create the events.csv file:
        self.oe_events_to_csv()
        # understand the samplerate and the first timestamp
        # if self.sample_rate is None:
        #     self.get_sample_rate()
        session = oea.Session(str(self.oe_path))
        # self.first_oe_timestamp = session.recordings[0].continuous[0].timestamps[0]
        # parse the events of the open-ephys recording

        ex_path = self.block_path / rf'oe_files' / self.exp_date_time / 'parsed_events.csv'
        self.oe_events, self.arena_vid_first_t, self.arena_vid_last_t = self.oe_events_parser(
            self.block_path / rf'oe_files' / self.exp_date_time / 'events.csv',
            self.channeldict,
            export_path=ex_path)
        print(f'created {ex_path}')
        self.l_vid_first_t = self.oe_events['R_eye_TTL'].loc[self.oe_events['R_eye_TTL_frame'].idxmin()]
        self.l_vid_last_t = self.oe_events['R_eye_TTL'].loc[self.oe_events['R_eye_TTL_frame'].idxmax()]
        self.r_vid_first_t = self.oe_events['L_eye_TTL'].loc[self.oe_events['L_eye_TTL_frame'].idxmin()]
        self.r_vid_last_t = self.oe_events['L_eye_TTL'].loc[self.oe_events['L_eye_TTL_frame'].idxmax()]

    # def get_first_last_frame_times(self, frame_col):
    #     s = ~pd.isna(self.oe_events[frame_col])
    #     df_loc = s[s == True].index[-1]
    #     self.oe_events['L_eye_TTL'].loc[df_loc]

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

    def synchronize_block(self, export=True):
        """
        This method builds a synced_videos dataframe
        1. The arena video is used as anchor
        2. The different anchor timestamps are aligned with the closest frames of the other sources
        """
        # check if there is an exported version of the blocksync_df:
        if pathlib.Path(self.analysis_path / 'blocksync_df.csv').exists():
            self.blocksync_df = pd.read_csv(pathlib.Path(self.analysis_path / 'blocksync_df.csv'))
            print('blocksync_df loaded from analysis folder')
            return self.blocksync_df
        else:
            print('creating blocksync_df')
        # define block_starts + block_ends
        start_time = max([self.arena_vid_first_t, self.r_vid_first_t, self.l_vid_first_t])
        end_time = min([self.arena_vid_last_t, self.r_vid_last_t, self.l_vid_last_t])

        # create a loop that goes over the series of arena timestamps between start and end of block:
        arena_tf = self.oe_events.query('@start_time < Arena_TTL < @end_time')[['Arena_TTL', 'Arena_TTL_frame']]
        r_eye_tf = self.oe_events.query('@start_time < Arena_TTL < @end_time or Arena_TTL != Arena_TTL')[
            ['R_eye_TTL', 'R_eye_TTL_frame']]
        r_eye_tf = r_eye_tf[np.invert(np.isnan(r_eye_tf.R_eye_TTL.values))]  # this removes nan values
        l_eye_tf = self.oe_events.query('@start_time < Arena_TTL < @end_time or Arena_TTL != Arena_TTL')[
            ['L_eye_TTL', 'L_eye_TTL_frame']]
        l_eye_tf = l_eye_tf[np.invert(np.isnan(l_eye_tf.L_eye_TTL.values))]  # this removes nan values
        # create a dataframe for the synchronization
        self.blocksync_df = pd.DataFrame(columns=['Arena_frame', 'L_eye_frame', 'R_eye_frame'],
                                         index=arena_tf.Arena_TTL)
        for i, t in enumerate(tqdm(arena_tf.Arena_TTL)):
            arena_frame = arena_tf.Arena_TTL_frame.iloc[i]
            l_eye_frame = l_eye_tf['L_eye_TTL_frame'].iloc[self.get_closest_frame(t, l_eye_tf['L_eye_TTL'])]
            r_eye_frame = r_eye_tf['R_eye_TTL_frame'].iloc[self.get_closest_frame(t, r_eye_tf['R_eye_TTL'])]
            self.blocksync_df.loc[t] = [arena_frame, l_eye_frame, r_eye_frame]
        print('created blocksync_df')
        if export:
            self.blocksync_df.to_csv(self.analysis_path / 'blocksync_df.csv')
            print(f'exported blocksync_df to {self.analysis_path}/ blocksync_df.csv')

    def produce_drift_report(self):
        """
        Method to get an accuracy report for the blocksync_df created previously
        :return:
        """
        if self.blocksync_df is None:
            print('no blocksync created - please create it with the synchronize_block() method')
        # first, we create the column_map dict:
        l_key = [i for i in self.blocksync_df.columns if 'L' in i]
        r_key = [i for i in self.blocksync_df.columns if 'R' in i]
        l_values = [i for i in self.oe_events.columns if 'L_e' in i]
        r_values = [i for i in self.oe_events.columns if 'R_e' in i]
        l_values.append('L_eye_slip')
        r_values.append('R_eye_slip')
        column_map = {
            l_key[0]: l_values,
            r_key[0]: r_values
        }
        # now an acc_report df
        acc_report = self.blocksync_df.copy(deep=True)
        for col in ['L_eye_frame', 'R_eye_frame']:
            acc_report.insert(loc=0, column=column_map[col][2], value=np.nan)
            print(f'working on {col}...')
            for i, t in tqdm(enumerate(self.blocksync_df.index)):
                eye_frame = self.blocksync_df[col].iloc[i]
                frame_col = column_map[col][1]
                ttl_col = column_map[col][0]
                eye_loc = self.oe_events.query(f"{frame_col} == {eye_frame}").index[0]
                eye_timestamp = self.oe_events.loc[eye_loc, ttl_col]
                large = max(eye_timestamp, t)
                small = min(eye_timestamp, t)
                abs_diff = abs(large - small)
                acc_report.at[t, column_map[col][2]] = abs_diff
        return acc_report

    @staticmethod
    def video_mean_brightness(vid_path, threshold_value):
        """
        This method goes through a video and calculates the mean brightness value for each frame

        Parameters
        ----------
        :param vid_path: Pathlib.Path
            path to the video to be analyzed
        :param threshold_value: int
            before averaging, a threshold is applied (this helps find the LEDs)

        Returns:
        ----------

        :return: frame_val: np.array
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
            frame_val = BlockSync.video_mean_brightness(str(vid), threshold_value)
            frame_val_list.append(frame_val)
        print(f'done, frame_val_list contains {len(frame_val_list)} objects', flush=True)

        return frame_val_list

    def synchronize_arena_timestamps(self, return_dfs=False, export_sync_df=True, get_only_anchor_vid=False):
        """
        This function reads the different arena timestamps files, chooses the longest as an anchor and fits
        frames corresponding with the closest timestamp to the anchor.
        It creates self.arena_sync_df and self.anchor_vid_name
        """
        if (self.analysis_path / 'arena_synchronization.csv').exists():
            print('arena_sync_df already exists, loading from file...')
            self.arena_sync_df = pd.read_csv(self.analysis_path / 'arena_synchronization.csv')
            if 'Unnamed: 0' in self.arena_sync_df.columns:
                self.arena_sync_df = self.arena_sync_df.drop(axis=1, labels='Unnamed: 0')
            return
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

        # now, check if the arena_synchronization df is already calculated:
        if pathlib.Path(self.analysis_path / 'arena_synchronization.csv').exists():
            print('arena_synchronization.csv was already created, loading it...')
            self.arena_sync_df = pd.read_csv(pathlib.Path(self.analysis_path / 'arena_synchronization.csv'))
        else:
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
            self.arena_sync_df.to_csv(self.analysis_path / 'arena_synchronization.csv')
            print(f'created arena_synchronization.csv in the block analysis folder')

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

        elif self.arena_sync_df is None:
            print('no arena synchronization step performed - running it now...')
            self.synchronize_arena_timestamps()
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
            self.arena_brightness_df.to_csv(self.block_path / 'analysis' / 'arena_brightness.csv')

    def validate_arena_synchronization(self, drop=None):
        if self.arena_brightness_df is None:
            print('No arena_brightness_df, run the create_arena_brightness_df method')
        x_axis = self.arena_brightness_df.index.values
        if drop is not None:
            columns = [c for c in self.arena_brightness_df.columns if drop not in c]
        else:
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

    def create_eye_brightness_df(self, threshold_value=30, export=True):
        """
        This method creates the l/r_eye_values lists, which represent the illumination level of eye video frames
        :param export: if true will export the df to csv
        :param threshold_value: The threshold value to use as mask before calculating brightness
        :return:
        """

        # first, check if the analysis folder contains the eye brightnes df:
        if pathlib.Path(self.analysis_path / 'eye_brightness_df.csv').exists():
            self.eye_brightness_df = pd.read_csv(pathlib.Path(self.analysis_path / 'eye_brightness_df.csv'))
            print('eye_brightness_df loaded from analysis folder')
            return self.eye_brightness_df

        if self.eye_brightness_df is None:
            if self.le_frame_val_list is None:
                self.le_frame_val_list = self.produce_frame_val_list(self.le_videos, threshold_value)
            if self.re_frame_val_list is None:
                self.re_frame_val_list = self.produce_frame_val_list(self.re_videos, threshold_value)

            self.l_eye_values = stats.zscore(self.le_frame_val_list[0][1])
            self.r_eye_values = stats.zscore(self.re_frame_val_list[0][1])

            df = self.blocksync_df.merge(
                right=pd.DataFrame(self.l_eye_values, columns=['L_values']).reset_index(),
                how='left',
                left_on='L_eye_frame',
                right_on='index')

            df = df.merge(
                right=pd.DataFrame(self.r_eye_values, columns=['R_values']).reset_index(),
                how='left',
                left_on='R_eye_frame',
                right_on='index')
            df = df.drop(labels=[i for i in df.columns if 'index' in i], axis=1)
            df.index = self.blocksync_df.index
            self.eye_brightness_df = df
            if export:
                self.eye_brightness_df.to_csv(self.analysis_path / 'eye_brightness_df.csv')
                print(rf'creating {self.analysis_path}/eye_brightness_df.csv')
        else:
            print('eye_brightness_df already exists')

    @staticmethod
    def blink_rising_edges_detector(b_series, f_series, threshold):
        """
        This function finds the rising edge of each blinking event in a list of frames' brightness values
        :param threshold:
        :param b_series: value of one brightness column from the eye_brightness_df object
        :param f_series: the frame numbers for the b_series (should be taken from the same DataFrame)
        :return: a list of indexes along the series which correspond with rising edges immediately after blinking events
        """
        # create the b_series object with indexes from the synchronized dataframe:
        b_series = pd.Series(data=b_series, index=f_series)
        # find events where the threshold is crossed and return their indexes:
        blink_indexes = b_series[b_series < threshold].index
        # now reduce them to the first index in each cluster:
        rising_edges = []
        for i, f in enumerate(blink_indexes):
            try:
                if f + 1 == blink_indexes[i + 1]:
                    # print(f'{f} is before {blink_indexes[i+1]} so I continue')
                    continue
                else:
                    rising_edges.append(f + 1)
                    # print(f'found a rising edge on frame {f+1} with a brightness value of {b_series[f+1]}')
            except IndexError:
                print(f'index error on position {i} out of {len(blink_indexes)}')
        return rising_edges

    @staticmethod
    def find_min_dist(n, ls):
        """
        finds the organ from l with the minimal absolute distance to n
        :param n: number
        :param ls: list of numbers
        :return: the number from l which has the smallest absolute distance to n
        """
        n_arr = np.array([n] * len(ls))
        ls = np.array(ls)
        diff_arr = n_arr - ls
        lowest_dist_ind = np.argmin(abs(diff_arr))
        return ls[lowest_dist_ind]

    def get_eyes_diff_list(self, threshold):
        r_rising = self.blink_rising_edges_detector(self.eye_brightness_df['R_values'].values,
                                                    self.eye_brightness_df['R_eye_frame'], threshold=threshold)
        l_rising = self.blink_rising_edges_detector(self.eye_brightness_df['L_values'].values,
                                                    self.eye_brightness_df['L_eye_frame'], threshold=threshold)
        rising_d = {
            'right': r_rising,
            'left': l_rising
        }
        if len(rising_d['right']) > len(rising_d['left']):
            k_shorter = 'left'
            k_longer = 'right'
        else:
            k_shorter = 'right'
            k_longer = 'left'

        sub_list = []
        for n in rising_d[k_shorter]:
            sub_list.append(self.find_min_dist(n, rising_d[k_longer]))

        self.eye_diff_list = rising_d[k_shorter] - np.array(sub_list)
        self.eye_diff_mode = stats.mode(self.eye_diff_list)[0][0]

        # determine lag directionality
        if k_shorter == 'right':
            if self.eye_diff_mode < 0:
                self.lag_direction = ['right', 'early']
            else:
                self.lag_direction = ['right', 'late']
        else:
            if self.eye_diff_mode < 0:
                self.lag_direction = ['left', 'early']
            else:
                self.lag_direction = ['left', 'late']
        print(f'The suspected lag between eye cameras is {self.eye_diff_mode} with the direction {self.lag_direction}')

    def fix_eye_synchronization(self):

        df = self.eye_brightness_df
        if self.lag_direction[0] == 'right':
            to_shift = df[['R_eye_frame', 'R_values']].copy()
            df.loc[:, ['R_eye_frame', 'R_values']] = to_shift.shift(periods=-int(self.eye_diff_mode))
        else:
            to_shift = df[['L_eye_frame', 'L_values']].copy()
            df.loc[:, ['L_eye_frame', 'L_values']] = to_shift.shift(periods=-int(self.eye_diff_mode))
        self.manual_sync_df = df
        print('created manual_sync_df attribute for the block')

    def move_eye_sync_manual(self, cols_to_move, step):

        df = self.manual_sync_df
        to_shift = df[cols_to_move].copy()
        df.loc[:, cols_to_move] = to_shift.shift(periods=step)
        self.manual_sync_df = df

    def get_blink_frames_manual(self, threshold=-35):

        """This is a utility function which detects rising edges for manual synchronization of eyes and arena"""
        r_rising = self.blink_rising_edges_detector(self.manual_sync_df['R_values'].values,
                                                    self.manual_sync_df['R_eye_frame'], threshold=threshold)
        l_rising = self.blink_rising_edges_detector(self.manual_sync_df['L_values'].values,
                                                    self.manual_sync_df['L_eye_frame'], threshold=threshold)
        dict_rising = {'left': l_rising,
                       'right': r_rising}
        return dict_rising

    def full_sync_verification(self):
        """
        Run this step before "export_manual_sync_df" to view the synchronization of the arena in relation to eyes,
        if further movements are necessary use "Move_eye_sync_manual" and run again -
        only export when this step gives a synchronized plot
        """
        arena_br = self.arena_brightness_df.iloc[self.manual_sync_df['Arena_frame']]
        x_axis = self.manual_sync_df.index
        bokeh_fig = figure(title=f'self Number {self.block_num} Full Synchronization Verification',
                           x_axis_label='Frame',
                           y_axis_label='Brightness Z_Score',
                           plot_width=1500,
                           plot_height=700
                           )
        color_list = ['orange', 'purple', 'teal', 'green', 'yellow']
        for ind, video in enumerate(arena_br.columns):
            bokeh_fig.line(x_axis, arena_br[video],
                           legend_label=video,
                           line_width=1,
                           line_color=color_list[ind])

        bokeh_fig.line(x_axis, self.manual_sync_df['L_values'], legend_label='Left_eye_values', line_width=1,
                       line_color='blue')
        bokeh_fig.line(x_axis, self.manual_sync_df['R_values'], legend_label='Right_eye_values', line_width=1,
                       line_color='red')
        show(bokeh_fig)

    def export_manual_sync_df(self):
        self.manual_sync_df.to_csv(self.analysis_path / 'manual_sync_df.csv')

    def import_manual_sync_df(self):
        try:
            self.manual_sync_df = pd.read_csv(self.analysis_path / 'manual_sync_df.csv')
            if 'Unnamed: 0' in self.manual_sync_df.columns:
                self.final_sync_df = self.manual_sync_df.drop(axis=1, labels='Unnamed: 0')
            else:
                self.final_sync_df = self.manual_sync_df
            # create a joint x axis with ms timebase for later use
            self.ms_axis = (self.final_sync_df['Arena_TTL'].values -
                            self.final_sync_df['Arena_TTL'].values[0]) / (self.sample_rate / 1000)
        except FileNotFoundError:
            print('there is no manual sync file, manually sync the block')

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

    def read_dlc_data(self, threshold_to_use=0.999, export=True):
        """
        Method to read and analyze the dlc files and fit ellipses to create the le/re ellipses attributes of the block
        """
        if (self.analysis_path / 're_df.csv').exists() and (self.analysis_path / 'le_df.csv').exists():
            self.re_df = pd.read_csv(self.analysis_path / 're_df.csv')
            self.re_df['ms_axis'] = self.ms_axis
            if 'Unnamed: 0' in self.re_df.columns:
                self.re_df = self.re_df.drop(axis=1, labels='Unnamed: 0')
            self.le_df = pd.read_csv(self.analysis_path / 'le_df.csv')
            self.le_df['ms_axis'] = self.ms_axis
            if 'Unnamed: 0' in self.le_df.columns:
                self.le_df = self.le_df.drop(axis=1, labels='Unnamed: 0')
            print('eye dataframes loaded from analysis folder')
            return

        pl = [i for i in os.listdir(self.l_e_path) if 'DLC' in i and '.csv' in i][0]
        self.le_csv = pd.read_csv(self.l_e_path / pl, header=1)
        pr = [i for i in os.listdir(self.r_e_path) if 'DLC' in i and '.csv' in i][0]
        self.re_csv = pd.read_csv(self.r_e_path / pr, header=1)
        self.le_ellipses = self.eye_tracking_analysis(self.le_csv, threshold_to_use)
        self.re_ellipses = self.eye_tracking_analysis(self.re_csv, threshold_to_use)

        self.le_df = self.final_sync_df.drop(labels=['Arena_frame', 'R_eye_frame'], axis=1)
        for column in list(self.le_ellipses.columns):
            self.le_df.insert(loc=len(self.le_df.columns), column=column, value=None)
        self.re_df = self.final_sync_df.drop(labels=['Arena_frame', 'L_eye_frame'], axis=1)
        for column in list(self.re_ellipses.columns):
            self.re_df.insert(loc=len(self.re_df.columns), column=column, value=None)
        print('populating le_df')
        for row in tqdm(self.le_df.index):
            try:
                frame = self.le_df['L_eye_frame'].loc[row]
                if frame == frame:
                    frame = int(frame)
                    self.le_df.loc[row, 'center_x'] = self.le_ellipses.iloc[frame]['center_x']
                    self.le_df.loc[row, 'center_y'] = self.le_ellipses.iloc[frame]['center_y']
                    self.le_df.loc[row, 'width'] = self.le_ellipses.width[frame]
                    self.le_df.loc[row, 'height'] = self.le_ellipses.height[frame]
                    self.le_df.loc[row, 'phi'] = self.le_ellipses.phi[frame]
                    self.le_df.loc[row, 'ellipse_size'] = self.le_ellipses.ellipse_size[frame]
            except IndexError:
                print(f'Tried to match frame {row} but there is no frame with this index')
                continue
            # le_df.at[row, 'rostral_edge'] = le_ellipses.rostral_edge[frame]
            # le_df.at[row, 'caudal_edge'] = le_ellipses.caudal_edge[frame]
        print('populating re_video_sync_df')
        for row in tqdm(self.re_df.index):
            try:
                frame = self.re_df['R_eye_frame'].loc[row]
                if frame == frame:
                    frame = int(frame)
                    self.re_df.loc[row, 'center_x'] = self.re_ellipses.iloc[frame]['center_x']
                    self.re_df.loc[row, 'center_y'] = self.re_ellipses.iloc[frame]['center_y']
                    self.re_df.loc[row, 'width'] = self.re_ellipses.width[frame]
                    self.re_df.loc[row, 'height'] = self.re_ellipses.height[frame]
                    self.re_df.loc[row, 'phi'] = self.re_ellipses.phi[frame]
                    self.re_df.loc[row, 'ellipse_size'] = self.re_ellipses.ellipse_size[frame]
            except IndexError:
                print(f'Tried to match frame {frame} but there is no frame with this index')
                continue
            # re_video_sync_df.at[row, 'rostral_edge'] = re_ellipses.rostral_edge[frame]
            # re_video_sync_df.at[row, 'caudal_edge'] = re_ellipses.caudal_edge[frame]
        self.re_df['ms_axis'] = self.ms_axis
        self.le_df['ms_axis'] = self.ms_axis
        print('done')

        if export:
            print('exporting to analysis folder')
            self.re_df.to_csv(self.analysis_path / 're_df.csv')
            self.le_df.to_csv(self.analysis_path / 'le_df.csv')

    def block_eye_plot(self, export=False, ms_x_axis=True, plot_saccade_locs=False,
                       saccade_frames_l=None, saccade_frames_r=None):
        # normalize values:
        le_el_z = (self.le_df.ellipse_size - self.le_df.ellipse_size.mean()) / self.le_df.ellipse_size.std()
        le_x_z = (self.le_df.center_x - np.mean(self.le_df.center_x)) / self.le_df.center_x.std()
        le_y_z = (self.le_df.center_y - np.mean(self.le_df.center_y)) / self.le_df.center_y.std()
        re_el_z = (self.re_df.ellipse_size - self.re_df.ellipse_size.mean()) / self.re_df.ellipse_size.std()
        re_x_z = (self.re_df.center_x - np.mean(self.re_df.center_x)) / self.re_df.center_x.std()
        re_y_z = (self.re_df.center_y - np.mean(self.re_df.center_y)) / self.re_df.center_y.std()
        if ms_x_axis is False:
            x_axis = self.final_sync_df['Arena_TTL'].values

        else:
            x_axis = (self.final_sync_df['Arena_TTL'].values -
                      self.final_sync_df['Arena_TTL'].values[0]) / (self.sample_rate / 1000)
        b_fig = figure(title=f'Pupil combined metrics block {self.block_num}',
                       x_axis_label='OE Timestamps',
                       y_axis_label='Z score',
                       plot_width=1500,
                       plot_height=700)
        b_fig.add_tools(HoverTool())
        b_fig.line(x_axis, le_el_z+7, legend_label='Left Eye Diameter', line_width=1.5, line_color='blue')
        b_fig.line(x_axis, le_x_z+14, legend_label='Left Eye X Position', line_width=1, line_color='cyan')
        b_fig.line(x_axis, le_y_z, legend_label='Left Eye Y position', line_width=1, line_color='green')
        b_fig.line(x_axis, re_el_z+7, legend_label='Right Eye Diameter', line_width=1.5, line_color='red')
        b_fig.line(x_axis, re_x_z+14, legend_label='Right Eye X Position', line_width=1, line_color='orange')
        b_fig.line(x_axis, re_y_z, legend_label='Right Eye Y position', line_width=1, line_color='pink')
        if plot_saccade_locs:
            b_fig.vbar(x=saccade_frames_l, width=1, bottom=-4, top=-1,
                       alpha=0.8, color='purple', legend_label='Left saccades')
            b_fig.vbar(x=saccade_frames_r, width=1, bottom=-4, top=-1,
                       alpha=0.8, color='brown', legend_label='Right saccades')
        if export:
            b_output.output_file(filename=str(self.analysis_path / f'pupillometry_block_{self.block_num}.html'),
                                 title=f'block {self.block_num} pupillometry')
        show(b_fig)

    def pupil_speed_calc(self):

        """This function creates a per-frame-velocity vector and appends it to the r/l eye dataframes for
        saccade analysis"""

        lx = self.le_df.center_x.values
        ly = self.le_df.center_y.values
        rx = self.re_df.center_x.values
        ry = self.re_df.center_y.values
        diff_dict = {
            'lx': np.diff(lx, prepend=1).astype(float),
            'ly': np.diff(ly, prepend=1).astype(float),
            'rx': np.diff(rx, prepend=1).astype(float),
            'ry': np.diff(ry, prepend=1).astype(float),
        }
        self.l_e_speed = np.sqrt((diff_dict['lx'] ** 2) + (diff_dict['ly'] ** 2))
        self.le_df['velocity'] = self.l_e_speed
        self.r_e_speed = np.sqrt((diff_dict['rx'] ** 2) + (diff_dict['ry'] ** 2))
        self.re_df['velocity'] = self.r_e_speed

    def plot_speed_graph(self):
        b_fig = figure(title='pupil speed graphs',
                       x_axis_label='ms',
                       y_axis_label='euclidean speed',
                       plot_width=1500,
                       plot_height=700)
        x_axis = (self.final_sync_df['Arena_TTL'].values -
                  self.final_sync_df['Arena_TTL'].values[0]) / (self.sample_rate / 1000)
        b_fig.line(x_axis,
                   self.l_e_speed,
                   legend_label='Left eye speed',
                   line_width=1.5,
                   line_color='blue')
        b_fig.line(x_axis,
                   self.r_e_speed*-1,
                   legend_label='inverse right eye speed',
                   line_width=1.5,
                   line_color='red')
        show(b_fig)

    def saccade_event_analayzer(self, threshold=2, automatic=False):
        """
        This method first finds the speed of the pupil in each frame, then detects saccade events
        :param automatic: when set to true, will go with the given threshold and not prompt the user for input,
        might create a wrongly thresholded dataset - use with caution
        :param threshold: The velocity threshold value to use
        :return:
        """

        # first, collect pupil speed:
        self.pupil_speed_calc()

        # now check if the anlaysis was already performed:
        if (self.analysis_path / 'r_saccades.csv').exists() and (self.analysis_path / 'l_saccades.csv').exists():
            self.r_saccades_chunked = pd.read_csv(self.analysis_path / 'r_saccades.csv')
            self.l_saccades_chunked = pd.read_csv(self.analysis_path / 'l_saccades.csv')
            if 'Unnamed: 0' in self.r_saccades_chunked.columns:
                self.r_saccades_chunked = self.r_saccades_chunked.drop(axis=1, labels='Unnamed: 0')
            if 'Unnamed: 0' in self.l_saccades_chunked.columns:
                self.l_saccades_chunked = self.l_saccades_chunked.drop(axis=1, labels='Unnamed: 0')
            print('loaded chunked saccade data from analysis folder')
            return

        # This section is a trial & error iteration with a dialogue to determine correct thresholding values

        flag = 0
        while flag == 0:
            # This segment gets saccades according to a given threshold
            l_saccades = self.ms_axis[np.argwhere(self.l_e_speed > threshold)]
            l_saccades = signal.medfilt(l_saccades[:, 0], 5)
            r_saccades = self.ms_axis[np.argwhere(self.r_e_speed > threshold)]
            r_saccades = signal.medfilt(r_saccades[:, 0], 5)
            if automatic is False:
                # This segment plots it for inspection and prompts a different threshold when needed
                self.block_eye_plot(plot_saccade_locs=True, saccade_frames_r=r_saccades, saccade_frames_l=l_saccades)
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
            else:
                print('automatic ON, Going ahead with the baseline threshold')
                flag = 1
        self.l_saccades = l_saccades
        self.r_saccades = r_saccades

        # saccade chunker for each eye:
        eye_dict = {
            0: 'left_eye_saccades',
            1: 'right_eye_saccades'
        }
        df_dict = {}
        for i, saccade_times in enumerate([self.l_saccades, self.r_saccades]):
            # collect indeces where diff is more than one frame, these are saccade starts locations
            saccades_begin = np.argwhere(np.diff(saccade_times) > 20)
            # NOTICE THE FAT FINGER 20, it stems from the sample rate
            # Verify that the mask will begin with a start event:
            if 0 not in saccades_begin:
                saccades_begin = np.insert(saccades_begin, 1, 0)
            # create binary masks for start and end locations
            saccade_start = np.zeros(len(saccade_times[1:]))
            saccade_start[saccades_begin] = 1
            saccade_ends = np.zeros(len(saccade_times[1:]))
            saccade_ends[saccades_begin - 1] = 1
            # create a dataframe for time masking
            df = pd.DataFrame({
                'saccade_times': saccade_times[1:],
                'diff_list': np.diff(saccade_times),
                'saccade_start': saccade_start,
                'saccade_end': saccade_ends
            })
            # use masks to get times for start / end of saccades
            saccade_start_times = df['saccade_times'][df['saccade_start'] == 1]
            saccade_end_times = df['saccade_times'][df['saccade_end'] == 1]

            df_dict[eye_dict[i]] = pd.DataFrame({
                'saccade_start': saccade_start_times,
                'saccade_end': saccade_end_times
            })
        # collect start times and lengths for each saccade
        tight_dict = {}
        for eye in ['left_eye_saccades', 'right_eye_saccades']:
            start = np.array(df_dict[eye]['saccade_start'].dropna())
            end = np.array(df_dict[eye]['saccade_end'].dropna())
            lengths = (end - start) // 17.05
            if eye == 'left_eye_saccades':
                start_conditions = self.le_df.iloc[self.le_df['ms_axis'].isin(start).values]
                end_conditions = self.le_df.iloc[self.le_df['ms_axis'].isin(end).values]
            elif eye == 'right_eye_saccades':
                start_conditions = self.re_df.iloc[self.re_df['ms_axis'].isin(start).values]
                end_conditions = self.re_df.iloc[self.re_df['ms_axis'].isin(end).values]

            euclidean_distance = np.sqrt(
                (start_conditions['center_x'].values - end_conditions['center_x'].values) ** 2 +
                (start_conditions['center_y'].values - end_conditions['center_y'].values) ** 2)
            tight_dict[eye] = pd.DataFrame({
                'saccade_start_ms': start,
                'saccade_length_frames': lengths,
                'saccade_magnitude': euclidean_distance
            })

        self.r_saccades_chunked = tight_dict['right_eye_saccades']
        self.l_saccades_chunked = tight_dict['left_eye_saccades']
        self.r_saccades_chunked.to_csv(self.analysis_path / 'r_saccades.csv')
        self.l_saccades_chunked.to_csv(self.analysis_path / 'l_saccades.csv')

    def calibrate_pixel_size(self, known_dist, overwrite=False):
        """
        This function takes in a known distance in mm and returns a calculation of the pixel size in each video 
        according to an ROI of given known distance in the L/R frames
        :param block: BlockSync object of a trial with eye videos
        :param known_dist: The distance to use for calibration measured in mm
        :param overwrite: If True will run the method even if the output df already exists
        :return: L and R values for pixel real-world size
        """
        # first check if this calibration already exists for the block:
        if not overwrite:
            if (self.analysis_path / 'LR_pix_size.csv').exists():
                internal_df = pd.read_csv(self.analysis_path / 'LR_pix_size.csv')
                self.L_pix_size = internal_df.at[0, 'L_pix_size']
                self.R_pix_size = internal_df.at[0, 'R_pix_size']
                print("got the calibration values from the analysis folder")
                return

        # get the first frames of both eyes as reference images
        # define the eye VideoCaptures
        rcap = cv2.VideoCapture(self.re_videos[0])
        lcap = cv2.VideoCapture(self.le_videos[0])

        # get the second frames:
        lcap.set(1, 1)
        lret, lframe = lcap.read()
        rcap.set(1, 1)
        rret, rframe = rcap.read()
        if rret and lret:
            Rroi = cv2.selectROI("select the area of the known measurement through the diagonal of the ROI", rframe)
            Lroi = cv2.selectROI("select the area of the known measurement through the diagonal of the ROI", lframe)
        else:
            print('some trouble with the video retrieval, check paths and try again')
        R_dist = np.sqrt(Rroi[2] ** 2 + Rroi[3] ** 2)
        L_dist = np.sqrt(Lroi[2] ** 2 + Lroi[3] ** 2)

        self.L_pix_size = known_dist / L_dist
        self.R_pix_size = known_dist / R_dist

        cv2.destroyAllWindows()

        # save these values to a dataframe for re-initializing the block:
        internal_df = pd.DataFrame(columns=['L_pix_size','R_pix_size'])
        internal_df.at[0, 'L_pix_size'] = self.L_pix_size
        internal_df.at[0, 'R_pix_size'] = self.R_pix_size
        internal_df.to_csv(self.analysis_path / 'LR_pix_size.csv', index=False)
        print(f'exported to {self.analysis_path / "LR_pix_size.csv"}')

