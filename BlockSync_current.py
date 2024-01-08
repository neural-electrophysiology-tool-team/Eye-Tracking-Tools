import glob
import h5py
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
import pickle
from OERecording import OERecording
from scipy.signal import welch, fftconvolve
from scipy.stats import zscore as scipy_zscore
from scipy.signal import find_peaks as scipy_find_peaks
from matplotlib import pyplot as plt

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
        print(f'instantiated block number {self.block_num} at Path: {self.block_path}, new OE version')
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
                1: 'Arena_TTL',
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
            oe_metadata_file_path = [i for i in self.oe_path.iterdir() if 'OE_metaData' in str(i)][0]
            if oe_metadata_file_path.is_file():
                self.oe_metadata_file_path = oe_metadata_file_path
                # try:
                self.oe_rec = OERecording(self.oe_metadata_file_path)
                print('created the .oe_rec attribute as an open ephys recording obj with get_data functionality')
                # except Exception:
                #     print('OERecording file could not be constructed')
        except IndexError:
            print('No open ephys record node here!!!')
        self.oe_events = None
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
        self.final_sync_df = None
        self.r_saccades_chunked = None
        self.l_saccades_chunked = None
        self.L_pix_size = None
        self.R_pix_size = None
        self.eye_diff_mode = None
        self.zeroth_sample_number = None
        self.get_zeroth_sample_number()
        self.saccade_dict = None
        self.synced_saccades_dict = None
        self.non_synced_saccades_dict = None
        self.non_synced_saccades_df = None
        self.synced_saccades_df = None
        self.led_blink_frames_l = None
        self.led_blink_frames_r = None
        self.le_jitter_dict = None
        self.re_jitter_dict = None

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

    def oe_events_to_csv(self, align_to_zero=True):
        """
        This method takes the open ephys events and puts them in a csv file, if align ot zero is true will align first
        acquired sample with sample # 0 (the native OpenEphys timestamps are aligned to oe clock 0, which is almost
        always prior to acquisition start)

        """
        # helper functions:
        def subtract_number_from_column(df, subtraction_number, column_names):
            """
            This function deals with aligning open-ephys events such
            that sample #0t is given to the first aquired sample
            """
            # Create a copy of the DataFrame to avoid modifying the original
            sub_df = df.copy()

            # Iterate over the column names in the list
            for column_name in column_names:
                # Check if the column exists in the DataFrame
                if column_name in sub_df.columns:
                    # Get the indices where the column value is not NaN
                    indices = sub_df.index[~pd.isna(sub_df[column_name])]

                    # Subtract the subtraction number from the selected indices
                    sub_df.loc[indices, column_name] -= subtraction_number

            return sub_df

        csv_export_path = self.block_path / 'oe_files' / self.oe_dirname / 'events.csv'
        if not csv_export_path.is_file():
            session = oea.Session(str(self.oe_path.parent))
            events_df = session.recordnodes[0].recordings[0].events
            if align_to_zero:
                print(f'aligning to zero with {self.zeroth_sample_number}')
                subtracted_df = subtract_number_from_column(events_df,
                                                            int(self.zeroth_sample_number),
                                                            ['sample_number'])
                subtracted_df.to_csv(csv_export_path)
                print(f'open ephys events aligned to zero & exported to csv file at {csv_export_path}')
            else:
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
        if len(self.arena_timestamps) == 0:
            try:
                self.arena_timestamps = \
                    [x for x in [y for y in (self.arena_path / 'frames_timestamps').iterdir()] if x.suffix == '.csv']
            except FileNotFoundError:
                print('no arena timestamps folder found')
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
        files_to_convert = [
            str(file) for file in eye_vid_path.rglob('*.h264') if 'DLC' not in str(file)
        ]
        converted_files = [str(file) for file in eye_vid_path.rglob('*.mp4') if 'DLC' not in str(file)]
        print(f'converting files: {files_to_convert} \n avoiding conversion on files: {converted_files}')
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
            [str(file) for file in eye_vid_path.rglob('*.mp4') if 'DLC' not in str(file)]
        timestamps_to_inspect = \
            [str(file) for file in eye_vid_path.rglob('*.csv') if 'timestamps.csv' in str(file)]
        if len(videos_to_inspect) == len(timestamps_to_inspect):
            for vid in range(len(videos_to_inspect)):
                timestamps = pd.read_csv(timestamps_to_inspect[vid])
                num_reported = timestamps.shape[0]
                cap = cv2.VideoCapture(videos_to_inspect[vid])
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f'The video named {os.path.split(videos_to_inspect[vid])[1]} has reported {num_reported} frames '
                      f'and has {length} frames, it has dropped {num_reported - length} frames')
                cap.release()
        else:
            print(f'something wrong with the inspection, numbers of files does not match:')
            print(f'videos_to_inspect = {videos_to_inspect}')
            print(f'timestamps_to_inspect = {timestamps_to_inspect}')

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

    def get_eye_brightness_vectors(self, threshold_value=30, export=True):
        """
        This is a utility function that generates the eye brightness vectors for later synchronization
        This step should be performed by a long looper over all data before synchronization
        :param threshold_value: The threshold value to use as mask before claculating brightness
        :param export: if true will export the vectors into two .csv files
        :return: /
        """
        print(f'getting eye brigtness values for block {self.block_num}...')
        if self.le_videos is None:
            self.le_videos = [vid for vid in glob.glob(str(self.block_path) + r'\eye_videos\LE\**\*.mp4') if
                              "DLC" not in vid]
        if self.re_videos is None:
            self.re_videos = [vid for vid in glob.glob(str(self.block_path) + r'\eye_videos\RE\**\*.mp4') if
                              "DLC" not in vid]
        p = self.analysis_path / 'eye_brightness_values_dict.pkl'
        if p.is_file():
            print('found a file!')
            with open(self.analysis_path / 'eye_brightness_values_dict.pkl', 'rb') as file:
                eye_brightness_dict = pickle.load(file)
                if self.le_frame_val_list is None:
                    self.le_frame_val_list = eye_brightness_dict['left_eye']
                if self.re_frame_val_list is None:
                    self.re_frame_val_list = eye_brightness_dict['right_eye']
        else:
            print()
            answer = input('no eye brigtness file exists, want to make it? (no / any other answer)')
            if answer == 'no':
                return
            self.le_frame_val_list = self.produce_frame_val_list(self.le_videos, threshold_value=threshold_value)
            self.re_frame_val_list = self.produce_frame_val_list(self.re_videos, threshold_value=threshold_value)
            if export:
                export_path = p
                frame_val_dict = {

                    'left_eye': self.le_frame_val_list,
                    'right_eye': self.re_frame_val_list
                }

                with open(export_path, 'wb') as file:
                    pickle.dump(frame_val_dict, file)

    def load_eye_brightness_vectors(self, threshold_value=30, export=True):
        """
        This is a utility function that generates the eye brightness vectors for later synchronization
        This step should be performed by a long looper over all data before synchronization
        :param threshold_value: The threshold value to use as mask before claculating brightness
        :param export: if true will export the vectors into two .csv files
        :return: /
        """
        with open(self.analysis_path / 'eye_brightness_values_dict.pkl', 'rb') as file:
            eye_brightness_dict = pickle.load(file)

        if self.le_videos is None:
            self.le_videos = [vid for vid in glob.glob(str(self.block_path) + r'\eye_videos\LE\**\*.mp4') if
                              "DLC" not in vid]
        if self.re_videos is None:
            self.re_videos = [vid for vid in glob.glob(str(self.block_path) + r'\eye_videos\RE\**\*.mp4') if
                              "DLC" not in vid]
        if self.le_frame_val_list is None:
            self.le_frame_val_list = eye_brightness_dict['left_eye']
        if self.re_frame_val_list is None:
            self.re_frame_val_list = eye_brightness_dict['right_eye']

        if export:
            export_path = self.analysis_path / 'eye_brightness_values_dict.pkl'
            frame_val_dict = {

                'left_eye': self.le_frame_val_list,
                'right_eye': self.re_frame_val_list
            }

            with open(export_path, 'wb') as file:
                pickle.dump(frame_val_dict, file)

    @staticmethod
    def oe_events_parser(open_ephys_csv_path, channel_names, arena_channel_name='Arena_TTL', export_path=None,
                         auto_break_selection=False):
        """

        :param open_ephys_csv_path: The path to an open ephys analysis tools exported csv
        :param channel_names: a dictionary of the form -
                        { 1 : 'channel name' (L_eye_camera)
                          2 : 'channel name' (Arena_TTL)
                          etc...}
        :param export_path: default None, if a path is specified a csv file will be saved
        :param arena_channel_name: the name in channel names which correponds with the arena TTLs
        :param auto_break_selection: When True, automatically selects the default ttl breaks to use as start/stop frames
        :returns open_ephys_events: a pandas DataFrame object where each column has the ON events of one channel
                                    and has a title from channel_names
        :returns open_ephys_off_events: same but for the OFF states (only important for the logical start-stop signal)

        """

        # Infer the active channels:

    # infer the active channels:
        df = pd.read_csv(open_ephys_csv_path)
        channels = np.unique(df['line'].to_numpy(copy=True))
        df_onstate = df[df['state'] == 1]  # cut the df to represent only rising edges
        ls = []
        for chan in channels:  # extract a pandas series of the ON stats timestamps for each channel
            if chan in channel_names.keys():
                sname = channel_names[chan]
                s = pd.Series(df_onstate['sample_number'][df_onstate['line'] == chan], name=sname)
                # If this is the arena channel we need to collect the first and last frames which correspond with
                # the video itsef (as TTLs are always being transmitted and a pause is expected before the video starts
                if sname == arena_channel_name:
                    diff_series = np.diff(s)
                    diff_mode = stats.mode(diff_series)[0][0]
                    arena_start_stop = np.where(diff_series > 10 * diff_mode)[0]
                    if len(arena_start_stop) != 2:
                        if auto_break_selection is not False:
                            start_ind = arena_start_stop[0]
                            end_ind = arena_start_stop[1]
                        else:
                            start_ind = input(f'there is some kind of problem because '
                                              f'there should be 2 breaks in the arena TTLs'
                                              f'and there are {len(arena_start_stop)}, those indices are: '
                                              f'{[s.iloc[i] for i in arena_start_stop]}... '
                                              f'choose the index to use as startpoint:')
                            end_ind = input('choose the index to use as endpoint:')
                        arena_start_timestamp = s.iloc[arena_start_stop[int(start_ind)] + 1]
                        print(f'arena first frame timestamp: {arena_start_timestamp}')
                        arena_end_timestamp = s.iloc[arena_start_stop[int(end_ind)]]
                        print(f'arena end frame timestamp: {arena_end_timestamp}')
                    else:
                        print(f'the arena TTLs are signaling start and stop positions at {arena_start_stop}')
                        arena_start_timestamp = s.iloc[arena_start_stop[0] + 1]
                        print(f'arena first frame timestamp: {arena_start_timestamp}')
                        arena_end_timestamp = s.iloc[arena_start_stop[1]]
                        print(f'arena end frame timestamp: {arena_end_timestamp}')
                else:
                    print(f'{sname} was not identified as {arena_channel_name}')
                # create a counter for every rising edge - these should match video frames
                s_counter = pd.Series(data=np.arange(len(s), dtype='Int32'),
                                      index=s.index.values,
                                      name=sname + '_frame')
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

    def parse_open_ephys_events(self, align_to_zero=True, auto_break_selection=False, arena_channel_name='Arena_TTL'):
        """
        Gets the sample rate from the settings.xml file
        Creates the parsed_events.csv file
        finds the first and last frame timestamps for each video source

        """

        print('running parse_open_ephys_events...')
        if (self.oe_path.parent / 'parsed_events.csv').is_file():
            print(f'block {self.block_num} has a parsed events file, reading...')
            self.oe_events = pd.read_csv(str((self.oe_path.parent / 'parsed_events.csv')), index_col=0)
            self.arena_vid_first_t = \
                self.oe_events[self.oe_events[str(arena_channel_name + '_frame')] == 0]['Arena_TTL'].values[0]

            last_frame = np.nanmax(self.oe_events[str(arena_channel_name + '_frame')].values)
            self.arena_vid_last_t = \
                self.oe_events[self.oe_events[str(arena_channel_name + '_frame')] == last_frame]['Arena_TTL'].values[0]
        else:
            # First, create the events.csv file:
            self.oe_events_to_csv(align_to_zero=align_to_zero)

            # Now work on the parsed_events file and expoert it
            ex_path = self.block_path / rf'oe_files' / self.exp_date_time / 'parsed_events.csv'
            self.oe_events, self.arena_vid_first_t, self.arena_vid_last_t = self.oe_events_parser(
                self.block_path / rf'oe_files' / self.exp_date_time / 'events.csv',
                self.channeldict,
                export_path=ex_path, auto_break_selection=auto_break_selection)
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
            self.blocksync_df = pd.read_csv(pathlib.Path(self.analysis_path / 'blocksync_df.csv'), engine='python')
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
            _ = df_list.pop(anchor_ind)
            df_to_sync = df_list
            # iterate over rows and videos to find the corresponding frames
            print('Synchronizing the different arena videos')
            if '0' in anchor_vid.columns:
                anchor_vid.rename(columns={'0': 'timestamp'}, inplace=True)
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
            self.eye_brightness_df = pd.read_csv(pathlib.Path(self.analysis_path / 'eye_brightness_df.csv'),
                                                 index_col=0)
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
        This function finds the rising edge of each blinking event in a list of frames' brightness values, but uses
        the differential instead of the absolute values for a clearer picture
        :param threshold:
        :param b_series: value of one brightness column from the eye_brightness_df object
        :param f_series: the frame numbers for the b_series (should be taken from the same DataFrame)
        :return: a list of indexes along the series which correspond with rising edges immediately after blinking events
        """
        # create the b_series object with indexes from the synchronized dataframe:
        b_series = pd.Series(data=b_series, index=f_series)
        # find events where the threshold is crossed and return their indexes:
        target_indices = np.insert(np.diff(b_series) > threshold, 0, 0)

        blink_indexes = b_series[target_indices].index

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

    def full_sync_verification(self, ms_axis=True, with_arena=True):
        """
        Run this step before "export_manual_sync_df" to view the synchronization of the arena in relation to eyes,
        if further movements are necessary use "Move_eye_sync_manual" and run again -
        only export when this step gives a synchronized plot
        """
        if ms_axis:
            x_axis = self.manual_sync_df['Arena_TTL'] / 20
            x_axis_label = 'Milliseconds'
        else:
            x_axis = self.manual_sync_df.index
            x_axis_label = 'Frame count (from arena first frame)'
        bokeh_fig = figure(title=f'self Number {self.block_num} Full Synchronization Verification',
                           x_axis_label='Frame',
                           y_axis_label='Brightness Z_Score',
                           plot_width=1500,
                           plot_height=700
                           )
        color_list = ['orange', 'purple', 'teal', 'green', 'yellow']
        if with_arena:
            arena_br = self.arena_brightness_df.iloc[self.manual_sync_df['Arena_frame']]
            for ind, video in enumerate(arena_br.columns):
                bokeh_fig.line(x_axis, arena_br[video],
                               legend_label=video,
                               line_width=1,
                               line_color=color_list[ind])
        else:
            bokeh_fig.line(x_axis, self.manual_sync_df['L_values'], legend_label='Left_eye_values', line_width=1,
                           line_color='blue')
            bokeh_fig.line(x_axis, self.manual_sync_df['R_values'], legend_label='Right_eye_values', line_width=1,
                           line_color='red')
        show(bokeh_fig)

    def export_manual_sync_df(self):
        self.manual_sync_df.to_csv(self.analysis_path / 'manual_sync_df.csv')

    def import_manual_sync_df(self, align_zero=True):
        try:
            self.manual_sync_df = pd.read_csv(self.analysis_path / 'manual_sync_df.csv')
            if 'Unnamed: 0' in self.manual_sync_df.columns:
                self.final_sync_df = self.manual_sync_df.drop(axis=1, labels='Unnamed: 0')
            else:
                self.final_sync_df = self.manual_sync_df
            # create a joint x-axis with ms timebase for later use
            if align_zero:
                self.ms_axis = self.final_sync_df['Arena_TTL'].values / (self.sample_rate / 1000)
            else:
                self.ms_axis = (self.final_sync_df['Arena_TTL'].values -
                                self.final_sync_df['Arena_TTL'].values[0]) / (self.sample_rate / 1000)
        except FileNotFoundError:
            print(f'there is no manual sync file for block {self.block_num}, manually sync the block')

    @staticmethod
    def interpolate_nan(data):
        """
        Interpolate NaN values in the input data.

        Parameters:
        - data (numpy array): The input data array.

        Returns:
        - numpy array: The data with NaN values interpolated.
        """
        nan_indices = np.isnan(data)
        not_nan_indices = ~nan_indices
        data[nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(not_nan_indices),
                                      data[not_nan_indices])
        return data

    def high_pass_frequency_filter(self, data, sampling_rate, cutoff_frequency, order=4):
        """
        Apply a high-pass Butterworth filter to the input data.

        Parameters:
        - data (numpy array): The input data array.
        - sampling_rate (float): The sampling rate of the input data.
        - cutoff_frequency (float): The cutoff frequency for the high-pass filter.
        - order (int): The order of the Butterworth filter (default is 4).

        Returns:
        - numpy array: The high-pass filtered data array.
        """
        # Interpolate NaN values in the input data
        data = self.interpolate_nan(data)

        # Normalize the cutoff frequency
        normalized_cutoff = cutoff_frequency / (0.5 * sampling_rate)

        # Design a high-pass Butterworth filter in second-order sections (SOS)
        sos = signal.butter(order, normalized_cutoff, btype='high', analog=False, output='sos')

        # Apply the SOS filter to the data
        filtered_data = signal.sosfilt(sos, data)

        return filtered_data

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

        # sort the pupil elements to dfs: x and y, with p as probability
        pupil_elements = np.array([x for x in data.columns if 'Pupil' in x])

        # get X coords
        pupil_xs_before_flip = data[pupil_elements[np.arange(0, len(pupil_elements), 3)]]

        # flip the data around the midpoint of the x-axis (shooting the eye through a camera flips right and left)
        pupil_xs = 320 * 2 - pupil_xs_before_flip

        # get Y coords (no need to flip as opencv conventions already start with origin at top left of frame
        # and so, positive Y is maintained as up in a flipped image as we have)
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
            # first, take all the values, and concatenate them into an X array
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

        print(f'\n ellipses calculation complete')
        return ellipse_df

    def read_dlc_data(self, threshold_to_use=0.999, export=True):
        """
        Method to read and analyze the dlc files and fit ellipses to create the le/re ellipses attributes of the block
        """
        # if the dataframes already exist, read them
        if (self.analysis_path / 're_df.csv').exists() and (self.analysis_path / 'le_df.csv').exists():
            self.re_df = pd.read_csv(self.analysis_path / 're_df.csv', index_col=0)
            self.le_df = pd.read_csv(self.analysis_path / 'le_df.csv', index_col=0)
            # append ms_axis to df
            self.re_df['ms_axis'] = self.re_df['Arena_TTL'] / 20
            self.le_df['ms_axis'] = self.le_df['Arena_TTL'] / 20
            print('eye dataframes loaded from analysis folder')
            return

        # find the dlc files
        pl = [i for i in os.listdir(self.l_e_path) if 'DLC' in i and '.csv' in i][0]
        self.le_csv = pd.read_csv(self.l_e_path / pl, header=1)
        pr = [i for i in os.listdir(self.r_e_path) if 'DLC' in i and '.csv' in i][0]
        self.re_csv = pd.read_csv(self.r_e_path / pr, header=1)

        # perform eye tracking analysis for each eye frame
        self.le_ellipses = self.eye_tracking_analysis(self.le_csv, threshold_to_use)
        self.re_ellipses = self.eye_tracking_analysis(self.re_csv, threshold_to_use)

        # get the frame-timestamp relationship for each video
        self.le_df = self.final_sync_df.drop(labels=['Arena_frame', 'R_eye_frame'], axis=1)
        self.re_df = self.final_sync_df.drop(labels=['Arena_frame', 'L_eye_frame'], axis=1)

        # use frame numbers as the hooks to merge data and frame-timestamp relationships
        self.le_df = self.le_df.merge(self.le_ellipses, left_on='L_eye_frame', right_index=True, how='left')
        self.re_df = self.re_df.merge(self.re_ellipses, left_on='R_eye_frame', right_index=True, how='left')
        self.re_df['ms_axis'] = self.re_df['Arena_TTL'] / 20
        self.le_df['ms_axis'] = self.le_df['Arena_TTL'] / 20
        print('created le / re dataframes')

        if export:
            print('exporting to analysis folder')
            self.re_df.to_csv(self.analysis_path / 're_df.csv')
            self.le_df.to_csv(self.analysis_path / 'le_df.csv')

    # jitter detection algorithm starts here:

    # The following functions deal with robustly removing lights-out frames from the video jitter analysis, could
    # expand these indices to bad video frame removal later:

    @staticmethod
    def rolling_window_z_scores(data, roll_w_size=120):
        """
        Detect threshold-crossing data points in a 1D data vector using a rolling window approach.

        Parameters:
        - data (numpy array): 1D data vector with values.
        - roll_w_size (int): Size, in samples, of the rolling window.

        Returns:
        - numpy array: A 1D array where each element is the relative z-score of the original value in its window
        """
        result = []
        len_data = len(data)

        for i in tqdm(range(0, len_data - roll_w_size + 1, roll_w_size)):
            window_data = data[i:i + roll_w_size]

            std_value = np.std(window_data)
            zscores = scipy_zscore(window_data)

            # threshold_crossing_indices = np.where(window_data < std_value*threshold)[0]
            if i == 0:
                result = zscores
            else:
                result = np.concatenate([result, zscores])

        # Handle remaining elements after the last complete rolling window
        last_window_start = len_data - roll_w_size
        last_window_data = data[last_window_start:]

        std_value_last = np.std(last_window_data)
        zscores_last = scipy_zscore(last_window_data)
        result = np.concatenate([result, zscores_last])

        return result

    @staticmethod
    def bokeh_plotter(data_vector,
                      plot_name='default',
                      x_axis='X',
                      y_axis='Y',
                      peaks=None):
        """Generates an interactive Bokeh plot for the given data vector.
        Args:
            data_vector (list or array): The data to be plotted.
            plot_name (str, optional): The title of the plot. Defaults to 'default'.
            x_axis (str, optional): The label for the x-axis. Defaults to 'X'.
            y_axis (str, optional): The label for the y-axis. Defaults to 'Y'.
            peaks (list or array, optional): Indices of peaks to highlight on the plot. Defaults to None.
        """
        fig = figure(title=f'bokeh explorer: {plot_name}',
                     x_axis_label=x_axis,
                     y_axis_label=y_axis,
                     plot_width=1500,
                     plot_height=700)
        fig.line(range(len(data_vector)), data_vector)
        if peaks is not None:
            fig.circle(peaks, data_vector[peaks], size=10, color='red')
        show(fig)

    def collect_lights_out_events(self, data, roll_w_size=1500, plot=False, plot_title='peak detector output'):
        """Identifies potential lights-out events from the given data.

        Args:
            data (list or array): The data containing light measurements.
            roll_w_size (int, optional): The window size for rolling z-score calculation. Defaults to 1500.
            plot (binary): when True, plots the output and detection results
            plot_title (str): plot title for differentiation
        Returns:
            list: Indices of the identified potential lights-out events.
        """

        print(f'data length is {len(data)}')
        # use a function to get relative z-scores and deal with changes in ambient light
        z_score_data = self.rolling_window_z_scores(data, roll_w_size=roll_w_size)
        z_score_data = z_score_data[:len(data)]
        print(f'z_score length is {len(z_score_data)}')
        # detect peaks based on the scipy algorithm
        peak_indices, _ = scipy_find_peaks(-1 * z_score_data, width=1, distance=3000)

        # expand the peaks to include the dimming and re-lighting frames
        expanded_indices = np.sort(np.array([peak_indices - 2,
                                             peak_indices - 1,
                                             peak_indices,
                                             peak_indices + 1,
                                             peak_indices + 2]).flatten())

        if plot:
            self.bokeh_plotter(z_score_data,
                               plot_name=plot_title,
                               x_axis='Frame',
                               y_axis='brightness Z score',
                               peaks=expanded_indices)

        return expanded_indices

    def find_led_blink_frames(self, plot=False):
        r_vals = self.re_frame_val_list[0][1]
        l_vals = self.le_frame_val_list[0][1]

        print('collecting left-eye data')
        l_peaks = self.collect_lights_out_events(data=l_vals,
                                                 plot=plot,
                                                 plot_title='Left eye peak detection output')
        print("collecting right eye data")
        r_peaks = self.collect_lights_out_events(data=r_vals,
                                                 plot=plot,
                                                 plot_title='right eye peak detection output')
        self.led_blink_frames_l = l_peaks
        self.led_blink_frames_r = r_peaks

    @staticmethod
    def euclidean_distance(coord1, coord2):
        """
        Compute the Euclidean distance between two sets of (x, y) coordinates.

        Parameters:
        - coord1: Tuple or array-like, representing the first set of coordinates (x1, y1).
        - coord2: Tuple or array-like, representing the second set of coordinates (x2, y2).

        Returns:
        - distance: Euclidean distance between coord1 and coord2.
        """
        coord1 = np.array(coord1)
        coord2 = np.array(coord2)

        # Calculate the Euclidean distance
        distance = np.sqrt(np.sum((coord1 - coord2) ** 2))

        return distance

    @staticmethod
    def normxcorr2(template, image, mode="full"):
        """
        Computes the normalized cross-correlation between a template and an image.

        Parameters:
        - template (numpy.ndarray): The template array for cross-correlation.
        - image (numpy.ndarray): The image array on which cross-correlation is performed.
        - mode (str, optional): The mode parameter for the cross-correlation operation.
          Default is "full", indicating that the output is the full discrete linear cross-correlation of the inputs.

        Returns:
        - numpy.ndarray: The normalized cross-correlation result between the template and the image.

        Normalized cross-correlation is a measure of similarity between the template and sub-regions of the image.
        The function first normalizes the input arrays by subtracting their means and performs cross-correlation
        using fast Fourier transform (FFT) for efficiency. The result is then normalized by the standard deviations
        of the template and the image.

        Note: The function handles cases where divisions by zero or very close to zero might occur, setting the
        corresponding elements in the output to zero.
        """
        template = template - np.mean(template)
        image = image - np.mean(image)

        a1 = np.ones(template.shape)
        # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
        ar = np.flipud(np.fliplr(template))
        out = fftconvolve(image, ar.conj(), mode=mode)

        image = fftconvolve(np.square(image), a1, mode=mode) - \
                np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

        # Remove small machine precision errors after subtraction
        image[np.where(image < 0)] = 0

        template = np.sum(np.square(template))
        out = out / np.sqrt(image * template)

        # Remove any divisions by 0 or very close to 0
        out[np.where(np.logical_not(np.isfinite(out)))] = 0

        return out

    @staticmethod
    def get_roi_for_correlation(video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file is opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        # prompt user for ROI
        ret, frame = cap.read()
        roi = list(cv2.selectROI(frame))

        # make sure the ROI has an odd number of pixels
        if roi[2] % 2 == 0:
            roi[2] += 1
        if roi[3] % 2 == 0:
            roi[3] += 1

        cv2.destroyAllWindows()

        return roi

    def compute_cross_correlation(self, video_path, roi, correlate_with_first_frame=True):

        # sort roi coords
        x, y, w, h = tuple(roi)

        # Initialize variables
        ref_correlation_ind_xy = None
        top_correlation_values = []
        top_correlation_xy = []
        top_correlation_dist = []
        x_displacement = []
        y_displacement = []
        first_frame = None

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Read the first frame
        ret, prev_frame = cap.read()

        # Convert to grayscale and extract ROI
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_roi = prev_gray[y:y + h, x:x + w]

        # Read video frames and compute cross-correlation over time
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(num_frames), desc="Computing Cross-Correlation", unit="frame"):
            # for _ in tqdm.tqdm(range(1)):
            ret, frame = cap.read()

            if not ret:
                break
            # Convert to grayscale and extract ROI
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_frame = gray_frame[y:y + h, x:x + w]
            if first_frame is None:
                first_frame = roi_frame
            correlation = self.normxcorr2(prev_roi, roi_frame)
            if correlate_with_first_frame:
                correlation = self.normxcorr2(first_frame, roi_frame)

            curr_max_coords = np.where(correlation == np.max(correlation))
            x_cur, y_cur = curr_max_coords[0][0], curr_max_coords[1][0]

            if ref_correlation_ind_xy is None:
                ref_correlation_ind_xy = [x_cur, y_cur]
                x_ref, y_ref = ref_correlation_ind_xy[0], ref_correlation_ind_xy[1]

            current_distance = self.euclidean_distance((x_cur, y_cur), (x_ref, y_ref))
            x_displacement.append(x_ref - x_cur)
            y_displacement.append(y_ref - y_cur)
            top_correlation_dist.append(current_distance)
            top_correlation_xy.append([x_cur, y_cur])
            top_correlation_values.append(np.max(correlation))

            # Update the previous frame and ROI
            prev_roi = roi_frame

        # Release the video capture object
        cap.release()

        # export the results as a dictionary
        result_dict = {
            'top_correlation_values': top_correlation_values,
            'top_correlation_dist': top_correlation_dist,
            'top_correlation_xy': top_correlation_xy,
            'y_displacement': y_displacement,
            'x_displacement': x_displacement
        }
        result_dict = self.sort_jitter_dict(result_dict)
        return result_dict

    @staticmethod
    def sort_jitter_dict(jitter_dict):
        """an internal method to sort out some mass"""
        curr_data = jitter_dict
        top_corr_x = np.array(curr_data['top_correlation_xy'])[:, 0]
        top_corr_y = np.array(curr_data['top_correlation_xy'])[:, 1]
        curr_data['top_correlation_x'] = top_corr_x
        curr_data['top_correlation_y'] = top_corr_y
        del curr_data['top_correlation_xy']
        return curr_data

    def get_jitter_reports(self,
                           export=False,
                           overwrite=False,
                           remove_led_blinks=True,
                           sort_on_loading=True):

        if (self.analysis_path / 'jitter_report_dict.pkl').exists() and overwrite is False:
            with open(self.analysis_path / 'jitter_report_dict.pkl', 'rb') as file:
                jitter_report_dict = pickle.load(file)
                if self.re_jitter_dict is None:
                    if sort_on_loading:
                        self.re_jitter_dict = self.sort_jitter_dict(jitter_report_dict['right_eye'])
                    else:
                        self.re_jitter_dict = jitter_report_dict['right_eye']
                if self.le_jitter_dict is None:
                    if sort_on_loading:
                        self.le_jitter_dict = self.sort_jitter_dict(jitter_report_dict['left_eye'])
                    else:
                        self.le_jitter_dict = jitter_report_dict['left_eye']
                print('jitter report loaded from analysis folder')
        else:
            # get ROI for each eye video
            left_eye_roi = self.get_roi_for_correlation(self.le_videos[0])
            right_eye_roi = self.get_roi_for_correlation(self.re_videos[0])

            # run the algorithm
            self.re_jitter_dict = self.compute_cross_correlation(self.re_videos[0], right_eye_roi)
            self.le_jitter_dict = self.compute_cross_correlation(self.le_videos[0], left_eye_roi)

        if remove_led_blinks:
            print('removing LED blink events...')
            self.find_led_blink_frames(plot=True)
            frames_to_remove_l = self.led_blink_frames_l
            frames_to_remove_r = self.led_blink_frames_r

            r_df = pd.DataFrame.from_dict(self.re_jitter_dict)
            r_df.iloc[frames_to_remove_r] = np.nan
            r_df.interpolate(inplace=True)
            l_df = pd.DataFrame.from_dict(self.le_jitter_dict)
            l_df.loc[frames_to_remove_l] = np.nan
            l_df.interpolate(inplace=True)

            self.le_jitter_dict = l_df.to_dict(orient='list')
            self.re_jitter_dict = r_df.to_dict(orient='list')

        if export:
            export_path = self.analysis_path / 'jitter_report_dict.pkl'
            jitter_report_dict = {

                'left_eye': self.le_jitter_dict,
                'right_eye': self.re_jitter_dict
            }

            with open(export_path, 'wb') as file:
                pickle.dump(jitter_report_dict, file)
            print(f'results saved to {export_path}')

        print('Jitter report computed - check out re/le_jitter_dict attributes')

    @staticmethod
    def plot_jitter_vectors(jitter_dict,
                            fig_suptitle=None,
                            num_ticks=None,
                            export_path=False):
        top_correlation_values = jitter_dict['top_correlation_values']
        top_correlation_dist = jitter_dict['top_correlation_dist']
        top_correlation_x = jitter_dict['top_correlation_x']
        top_correlation_y = jitter_dict['top_correlation_y']

        fig, axs = plt.subplots(3, 1, figsize=(20, 7), sharex=True, dpi=300, constrained_layout=False)
        fig.suptitle(fig_suptitle)
        x_axis = np.arange(len(top_correlation_values)) // 60
        if num_ticks is not None:
            x_ticker = np.round(np.linspace(x_axis[0], x_axis[-1], num_ticks))
        axs[0].plot(x_axis, top_correlation_values)
        axs[0].set_title('top correlation values')
        axs[0].set_ylabel('Corr score')
        if num_ticks is not None:
            axs[0].set_xticks(x_ticker)
        axs[0].grid(True, linestyle='dotted')
        axs[1].plot(x_axis, top_correlation_dist)
        axs[1].set_title('top correlation euclidean distance')
        axs[1].set_ylabel('distance [pixels]')
        if num_ticks is not None:
            axs[1].set_xticks(x_ticker)
        axs[1].grid(True, linestyle='dotted')
        _ = axs[2].plot(x_axis, top_correlation_x, label='X coordinate')
        _ = axs[2].plot(x_axis, top_correlation_y, label='Y coordinate')
        axs[2].set_title('XY coordinates of top correlation values')
        axs[2].set_ylabel('top corr coordinates')
        axs[2].set_xlabel('Seconds')
        axs[2].legend()
        if num_ticks is not None:
            axs[2].set_xticks(x_ticker)
        axs[2].grid(True, linestyle='dotted')
        if export_path is not False:
            fig.savefig(export_path)
        return fig

    # TODO: !!!!write the jitter correction algorithm here!!!!
    def correct_jitter(self):
        """
        This function should correct the le/re dataframes such that for every frame both the x and y coordinates
        are shifted such that: corrected_x = reference_x - current_x
        where the distances are those from the jitter report (which also computes drift)
        :return:
        """
        print('method not implemented')
        return

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
            b_fig = figure(title=f'Pupil combined metrics block {self.block_num}',
                           x_axis_label='OE Timestamps',
                           y_axis_label='Z score',
                           plot_width=1500,
                           plot_height=700)
        else:
            x_axis = self.final_sync_df['Arena_TTL'].values / (self.sample_rate / 1000)
            b_fig = figure(title=f'Pupil combined metrics block {self.block_num}',
                           x_axis_label='[Milliseconds]',
                           y_axis_label='[Z score]',
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

        """This function creates a per-frame-velocity vector and
        appends it to the r/l eye dataframes for saccade analysis"""

        lx = self.le_df.center_x.fillna(np.nan).values
        ly = self.le_df.center_y.fillna(np.nan).values
        rx = self.re_df.center_x.fillna(np.nan).values
        ry = self.re_df.center_y.fillna(np.nan).values
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

    def saccade_event_analayzer(self, threshold=2, automatic=False, overwrite=False):
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
        if (self.analysis_path / 'r_saccades.csv').exists() and (self.analysis_path / 'l_saccades.csv').exists()\
                and (overwrite is False):
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

            # This segment deals with getting the euclidean distance without trying to take the sqrt of 0:
            dist_squared = ((start_conditions['center_x'].values - end_conditions['center_x'].values) ** 2) + \
                           ((start_conditions['center_y'].values - end_conditions['center_y'].values) ** 2)
            sqrt_values = np.sqrt(dist_squared[dist_squared != 0].astype(float))
            euclidean_distance = np.zeros_like(dist_squared)
            euclidean_distance[dist_squared != 0] = sqrt_values
            # euclidean_distance = np.where(dist_squared != 0, np.sqrt(dist_squared[dist_squared != 0]), 0)
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

    def get_zeroth_sample_number(self):
        """
        Open-ephys recordings write events from an imaginary zeroth timestamp a few seconds before sample recording
        actually starts. This creates a lag between the file's internal timestamps as saved in the timestamps stream of
        different continuous channels and the count-based timestamps paradigm of the matlab code from Mark.
        To correct this, I need to take the first sample number of this recording and subtract it from all
        timestamps-based synchronization (primarily events) so that everything can live on a ms timebase counted from
        sample#0
        :return:
        """

        print(f'retrieving zertoh sample number for block {self.block_num}')

        # first, try and get the sample_num from a pre-performed step:
        if self.oe_rec is not None:
            # if access to the recording metadata exists calculate the zeroth lag with the sample_rate
            self.zeroth_sample_number = int(self.oe_rec.globalStartTime_ms*(self.sample_rate / 1000))

        elif (self.analysis_path / 'zeroth_sample_num.csv').exists():
            df = pd.read_csv(self.analysis_path / 'zeroth_sample_num.csv')
            self.zeroth_sample_number = df['zeroth_sample_num'][0]
            print('read zeroth sample number from .csv file')
            del df
            return

        else:
            print('Never been done, opening OE data recording the long way to get it...')
            # open the OE datafile to get the number of the first recorded sample:
            session = oea.Session(str(self.oe_path.parent))
            zeroth_sample_num = [session.recordnodes[0].recordings[0].continuous[0].sample_numbers[0]]
            self.zeroth_sample_number = zeroth_sample_num[0]

            # get rid of the RAM overhead
            del session
        print('got it!')


    @staticmethod
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        """
        return np.isnan(y), lambda z: z.nonzero()[0]

    def parse_dataset_to_df(self, saccade_dict):

        df = pd.DataFrame(
            columns=['datetime', 'block', 'eye', 'timestamps', 'fs', 'pxx', 'samples', 'x_coords', 'y_coords',
                     'vid_inds', 'x_speed', 'y_speed', 'magnitude', 'dx', 'dy', 'direction', 'accel'])

        index_counter = 0

        block = saccade_dict  # in a certain block
        for e in block.keys():
            eye = block[e]  # in one of the eyes
            for row in range(len(eye['samples'])):  # for each saccade
                df.at[index_counter, 'datetime'] = self.exp_date_time
                df.at[index_counter, 'block'] = self.block_num
                df.at[index_counter, 'eye'] = e
                for col in eye.keys():  # for each columm
                    v = eye[col][row]  # get value of location
                    df.at[index_counter, col] = v


                print(index_counter, end='\r', flush=True)
                index_counter += 1

        print(f'done, dataframe contains {index_counter} saccades')
        return df

    def saccade_dict_creation(self, sampling_window_ms, ep_channel_number,
                              automate_saccade_detection=True, overwrite_saccade_data=False):
        """
        This function cascades over the steps that end in two dataframe objects, one for synced saccades and the other
        non-synced saccades. This function requires completion of all previous analysis steps (in particular, lizMov.mat
        should be produced using Mark's code)
        :param sampling_window_ms: the time window for each saccade (half before half after the saccade)
        :param ep_channel_number: a channel to get neural data from, limited to 1 for now (you can use get_data to
        draw additional channels based on timestamps from the dataframe later
        :return:
        """
        # collect accelerometer data
        # path definition
        p = self.oe_path / 'analysis'
        analysis_list = os.listdir(p)
        correct_analysis = [i for i in analysis_list if self.animal_call in i][0]
        p = p / str(correct_analysis)
        mat_path = p / 'lizMov.mat'
        print(f'path to mat file is {mat_path}')
        # read mat file
        mat_data = h5py.File(str(mat_path), 'r')
        mat_dict = {'t_mov_ms': mat_data['t_mov_ms'][:],
                    'movAll': mat_data['movAll'][:]}

        acc_df = pd.DataFrame(data=np.array([mat_dict['t_mov_ms'][:, 0], mat_dict['movAll'][:, 0]]).T,
                              columns=['t_mov_ms', 'movAll'])
        mat_data.close()

        self.saccade_event_analayzer(automatic=automate_saccade_detection,
                                     overwrite=overwrite_saccade_data,
                                     threshold=2)

        # create the top-level block dict object
        self.saccade_dict = {
            'L': {},
            'R': {}
        }

        # create and populate the internal dictionaries (for each eye)
        for i, e in enumerate(['L', 'R']):
            # get the correct saccades_chunked object and eye_df
            saccades_chunked = [self.l_saccades_chunked, self.r_saccades_chunked][i]
            eye_df = [self.le_df, self.re_df][i]
            saccades = saccades_chunked[saccades_chunked.saccade_length_frames > 0]
            saccade_times = np.sort(saccades.saccade_start_ms.values)
            saccade_lengths = saccades.saccade_length_frames.values
            ep_channel_numbers = [ep_channel_number]
            pre_saccade_ts = saccade_times - (sampling_window_ms / 2)  #

            # get the data of the relevant saccade time windows:
            print(f"getting data with block_number {self.block_num}: \n"
                  f"There are {len(pre_saccade_ts)} saccade events: \n"
                  f"pre_saccade_ts = {pre_saccade_ts}"
                  f"")
            ep_data, ep_timestamps = self.oe_rec.get_data(ep_channel_numbers,
                                                          pre_saccade_ts,
                                                          sampling_window_ms,
                                                          convert_to_mv=True)  # data [n_channels, n_windows, nSamples]

            # start populating the dictionary
            self.saccade_dict[e] = {
                "timestamps": [],
                "fs": [],
                "pxx": [],
                "samples": [],
                "x_coords": [],
                "y_coords": [],
                "vid_inds": [],
                "accel": [],
                "length": []
            }

            # go saccade by saccade
            for sac_i, j in enumerate(range(len(pre_saccade_ts))):
                # get specific saccade samples:
                saccade_samples = ep_data[0, j, :]  # [n_channels, n_windows, nSamples]
                # get the spectral profile for the segment
                fs, pxx = welch(saccade_samples, self.sample_rate, nperseg=16384, return_onesided=True)

                j0 = pre_saccade_ts[j]
                j1 = pre_saccade_ts[j] + sampling_window_ms
                s_df = eye_df.query("ms_axis >= @j0 and ms_axis <= @j1")
                x_coords = s_df['center_x'].values
                y_coords = s_df['center_y'].values
                vid_inds = np.array(s_df.Arena_TTL.values - s_df.Arena_TTL.values[0], dtype='int32')

                # deal with missing datapoints in saccades:
                interpolated_coords = []
                bad_saccade = False
                for y in [x_coords, y_coords]:
                    nan_count = np.sum(np.isnan(y.astype(float)))
                    if nan_count > 0:
                        if nan_count < len(y) / 2:
                            # print(f'saccade at ind {i} has {nan_count} nans, interpolating...')
                            # find nan values in the vector
                            nans, z = self.nan_helper(y.astype(float))
                            # interpolate using the helper lambda function
                            y[nans] = np.interp(z(nans), z(~nans), y[~nans].astype(float))
                            # replace the interpolated values for the saccade
                            interpolated_coords.append(y)
                        else:
                            print(f'too many nans at ind {j}, ({np.sum(np.isnan(y))}) - cannot interpolate properly',
                                  end='\r', flush=True)
                            bad_saccade = True
                    else:
                        interpolated_coords.append(y)

                # get accelerometer data for the ms_based section:
                # get_ms_segment
                ms_segment = s_df['ms_axis']
                s0 = ms_segment.iloc[0]
                s1 = ms_segment.iloc[-1]
                mov_mag = np.sum(acc_df.query('t_mov_ms > @s0 and t_mov_ms < @s1').movAll.values)

                # remove bad saccades
                if bad_saccade:
                    continue
                # append OK saccades
                else:
                    self.saccade_dict[e]['timestamps'].append(pre_saccade_ts[j])
                    self.saccade_dict[e]['x_coords'].append(interpolated_coords[0])
                    self.saccade_dict[e]['y_coords'].append(interpolated_coords[1])
                    self.saccade_dict[e]['vid_inds'].append(vid_inds)
                    self.saccade_dict[e]['fs'].append(fs)
                    self.saccade_dict[e]['pxx'].append(pxx)
                    self.saccade_dict[e]['samples'].append(saccade_samples)
                    self.saccade_dict[e]['accel'].append(mov_mag)
                    self.saccade_dict[e]['length'].append(saccade_lengths[sac_i])
        print(f'block {self.block_num} saccade dict done')
        self.sort_synced_saccades()

    def saccade_dict_creation_3d(self, sampling_window_ms, ep_channel_number,
                                 automate_saccade_detection=True, overwrite_saccade_data=False):
        """
        This function cascades over the steps that end in two dataframe objects, one for synced saccades and the other
        non-synced saccades. This function requires completion of all previous analysis steps (in particular, lizMov.mat
        should be produced using Mark's code) - This is the 3D projection demo version - still under construction
        :param overwrite_saccade_data: When True, will overwrite existing saccade detections
        :param automate_saccade_detection: When True, will not prompt for a visual inspection of saccade detections
        :param sampling_window_ms: the time window for each saccade (half before half after the saccade)
        :param ep_channel_number: a channel to get neural data from, limited to 1 for now (you can use get_data to
        draw additional channels based on timestamps from the dataframe later
        :return:
        """

        # as a preliminary step - add phi and theta to the le_df dataframe
        # read the 3d projection analysis:
        le_df_3d = pd.read_csv(self.analysis_path / 'le_df_3d.csv', index_col=0)
        le_df_3d = le_df_3d[['timestamp', 'diameter', 'theta', 'phi']]
        le_df_3d.rename(columns={'timestamp': 'ms_axis'},inplace=True)
        le_df_3d.interpolate(method='linear', inplace=True)

        re_df_3d = pd.read_csv(self.analysis_path / 're_df_3d.csv', index_col=0)
        re_df_3d = re_df_3d[['timestamp', 'diameter', 'theta', 'phi']]
        re_df_3d.rename(columns={'timestamp': 'ms_axis'},inplace=True)
        re_df_3d.interpolate(method='linear', inplace=True)
        self.le_df = self.le_df.merge(le_df_3d[['ms_axis', 'theta', 'phi']],
                                      on='ms_axis', how='left',
                                      suffixes=('_og', '_3d'))
        self.re_df = self.re_df.merge(re_df_3d[['ms_axis', 'theta', 'phi']],
                                      on='ms_axis', how='left',
                                      suffixes=('', '_3d'))

        # collect accelerometer data
        # path definition
        p = self.oe_path / 'analysis'
        analysis_list = os.listdir(p)
        correct_analysis = [i for i in analysis_list if self.animal_call in i][0]
        p = p / str(correct_analysis)
        mat_path = p / 'lizMov.mat'
        print(f'path to mat file is {mat_path}')
        # read mat file
        mat_data = h5py.File(str(mat_path), 'r')
        mat_dict = {'t_mov_ms': mat_data['t_mov_ms'][:],
                    'movAll': mat_data['movAll'][:]}

        acc_df = pd.DataFrame(data=np.array([mat_dict['t_mov_ms'][:, 0], mat_dict['movAll'][:, 0]]).T,
                              columns=['t_mov_ms', 'movAll'])
        mat_data.close()

        self.saccade_event_analayzer(automatic=automate_saccade_detection,
                                     overwrite=overwrite_saccade_data,
                                     threshold=2)

        # create the top-level block dict object
        self.saccade_dict = {
            'L': {},
            'R': {}
        }

        # create and populate the internal dictionaries (for each eye)
        for i, e in enumerate(['L', 'R']):
            # get the correct saccades_chunked object and eye_df
            saccades_chunked = [self.l_saccades_chunked, self.r_saccades_chunked][i]
            eye_df = [self.le_df, self.re_df][i]
            saccades = saccades_chunked[saccades_chunked.saccade_length_frames > 0].sort_values(by='saccade_start_ms')
            saccade_times = saccades.saccade_start_ms.values
            saccade_lengths = saccades.saccade_length_frames.values
            ep_channel_numbers = [ep_channel_number]
            pre_saccade_ts = saccade_times - (sampling_window_ms / 2)  #

            # get the data of the relevant saccade time windows:
            print(f"getting data with block_number {self.block_num}: \n"
                  f"There are {len(pre_saccade_ts)} saccade events: \n"
                  f"pre_saccade_ts = {pre_saccade_ts}"
                  f"")
            ep_data, ep_timestamps = self.oe_rec.get_data(ep_channel_numbers,
                                                          pre_saccade_ts,
                                                          sampling_window_ms,
                                                          convert_to_mv=True)  # data [n_channels, n_windows, nSamples]

            # start populating the dictionary
            self.saccade_dict[e] = {
                "timestamps": [],
                "fs": [],
                "pxx": [],
                "samples": [],
                "x_coords": [],
                "y_coords": [],
                "vid_inds": [],
                "accel": [],
                "length": []
            }

            # go saccade by saccade
            for sac_i, j in enumerate(range(len(pre_saccade_ts))):
                # get specific saccade samples:
                saccade_samples = ep_data[0, j, :]  # [n_channels, n_windows, nSamples]
                # get the spectral profile for the segment
                fs, pxx = welch(saccade_samples, self.sample_rate, nperseg=16384, return_onesided=True)

                j0 = pre_saccade_ts[j]
                j1 = pre_saccade_ts[j] + sampling_window_ms
                s_df = eye_df.query("ms_axis >= @j0 and ms_axis <= @j1")
                x_coords = s_df['phi_3d'].values
                y_coords = s_df['theta'].values
                vid_inds = np.array(s_df.Arena_TTL.values - s_df.Arena_TTL.values[0], dtype='int32')

                # deal with missing datapoints in saccades:
                interpolated_coords = []
                bad_saccade = False
                for y in [x_coords, y_coords]:
                    nan_count = np.sum(np.isnan(y.astype(float)))
                    if nan_count > 0:
                        if nan_count < len(y) / 2:
                            # print(f'saccade at ind {i} has {nan_count} nans, interpolating...')
                            # find nan values in the vector
                            nans, z = self.nan_helper(y.astype(float))
                            # interpolate using the helper lambda function
                            y[nans] = np.interp(z(nans), z(~nans), y[~nans].astype(float))
                            # replace the interpolated values for the saccade
                            interpolated_coords.append(y)
                        else:
                            print(f'too many nans at ind {j}, ({np.sum(np.isnan(y))}) - cannot interpolate properly',
                                  end='\r', flush=True)
                            bad_saccade = True
                    else:
                        interpolated_coords.append(y)

                # get accelerometer data for the ms_based section:
                # get_ms_segment
                ms_segment = s_df['ms_axis']
                s0 = ms_segment.iloc[0]
                s1 = ms_segment.iloc[-1]
                mov_mag = np.sum(acc_df.query('t_mov_ms > @s0 and t_mov_ms < @s1').movAll.values)

                # remove bad saccades
                if bad_saccade:
                    continue
                # append OK saccades
                else:
                    self.saccade_dict[e]['timestamps'].append(pre_saccade_ts[j])
                    self.saccade_dict[e]['x_coords'].append(interpolated_coords[0])
                    self.saccade_dict[e]['y_coords'].append(interpolated_coords[1])
                    self.saccade_dict[e]['vid_inds'].append(vid_inds)
                    self.saccade_dict[e]['fs'].append(fs)
                    self.saccade_dict[e]['pxx'].append(pxx)
                    self.saccade_dict[e]['samples'].append(saccade_samples)
                    self.saccade_dict[e]['accel'].append(mov_mag)
                    self.saccade_dict[e]['length'].append(saccade_lengths[sac_i])
        print(f'block {self.block_num} saccade dict done')
        self.sort_synced_saccades()

    @staticmethod
    def saccade_before_after_broken(coords):
        max_ind = np.argmax(coords)
        min_ind = np.argmin(coords)
        if max_ind < min_ind:
            before = coords[max_ind]
            after = coords[min_ind]
        else:
            before = coords[min_ind]
            after = coords[max_ind]
        delta = after - before
        return before, after, delta

    def sort_synced_saccades(self):
        """
        This function takes a saccades dictionary and returns two sorted dictionaries -
        one with synced saccades, the other with non-synced saccades
        :param b_dict:
        :return:
        """

        b_dict = self.saccade_dict
        # get the two timestamps vectors
        l_times = np.array(b_dict['L']['timestamps'])
        r_times = np.array(b_dict['R']['timestamps'])

        # I want to collect the matching indices from the L and R dictionaries
        # and create a "synced saccades dict" object
        # that only has two-eyed saccades included in it...
        # first, I have to understand which rows of the dictionaries go together:
        # create a matrix of [left eye timestamp, -,left eye ind, -]
        s_mat = np.empty([len(l_times), 5])
        s_mat[:, 0] = l_times
        s_mat[:, 2] = np.arange(0, len(l_times))
        # find and fit the right eye times and indices on columns 1 and 3
        for i, lt in enumerate(s_mat[:, 0]):
            array = np.abs((r_times - lt))
            ind_min_diff = np.argmin(array)
            min_diff = array[ind_min_diff]
            rt = r_times[ind_min_diff]
            s_mat[i, 3] = ind_min_diff
            s_mat[i, 1] = rt
            s_mat[i, 4] = min_diff

        # create a dataframe for queries and testing, define a threshold and remove non sync saccades
        s_df = pd.DataFrame(s_mat, columns=['lt', 'rt', 'left_ind', 'right_ind', 'diff'])
        threshold = 1400  # 70 ms to consider a saccade simultaneous
        s_df = s_df.query('diff<@threshold')
        ind_dict = {
            'L': s_df['left_ind'].values,
            'R': s_df['right_ind'].values
        }

        # create a synced dictionary for the block:
        synced_b_dict = {
            'L': {},
            'R': {}
        }
        for e in ['L', 'R']:
            inds = ind_dict[e].astype(int)
            synced_b_dict[e] = {
                "timestamps": np.array(b_dict[e]['timestamps'])[inds],
                "fs": np.array(b_dict[e]['fs'])[inds],
                "pxx": np.array(b_dict[e]['pxx'])[inds],
                "samples": np.array(b_dict[e]['samples'])[inds],
                "x_coords": np.array(b_dict[e]['x_coords'], dtype=object)[inds],
                "y_coords": np.array(b_dict[e]['y_coords'], dtype=object)[inds],
                "vid_inds": np.array(b_dict[e]['vid_inds'], dtype=object)[inds],
                "accel": np.array(b_dict[e]['accel'])[inds],
                "length": np.array(b_dict[e]['length'], dtype=object)[inds]
            }

        non_sync_b_dict = {
            'L': {},
            'R': {}
        }
        for e in ['L', 'R']:
            inds = ind_dict[e].astype(int)
            logical = np.ones(len(b_dict[e]['timestamps'])).astype(np.bool)
            logical[inds] = 0
            non_sync_b_dict[e] = {
                "timestamps": np.array(b_dict[e]['timestamps'])[logical],
                "fs": np.array(b_dict[e]['fs'])[logical],
                "pxx": np.array(b_dict[e]['pxx'])[logical],
                "samples": np.array(b_dict[e]['samples'])[logical],
                "x_coords": np.array(b_dict[e]['x_coords'], dtype=object)[logical],
                "y_coords": np.array(b_dict[e]['y_coords'], dtype=object)[logical],
                "vid_inds": np.array(b_dict[e]['vid_inds'], dtype=object)[logical],
                "accel": np.array(b_dict[e]['accel'])[logical],
                "length": np.array(b_dict[e]['length'], dtype=object)[inds]
            }
        self.calibrate_pixel_size(10)
        self.synced_saccades_dict = self.saccade_dict_enricher(synced_b_dict)
        self.synced_saccades_df = self.parse_dataset_to_df(self.synced_saccades_dict)
        # get a calibrated dx dy
        self.synced_saccades_df['calib_dx'] = \
            self.synced_saccades_df['dx'] * \
            (self.synced_saccades_df['eye'].map({'R': self.R_pix_size, 'L': self.L_pix_size}))
        self.synced_saccades_df['calib_dy'] = \
            self.synced_saccades_df['dy'] * \
            (self.synced_saccades_df['eye'].map({'R': self.R_pix_size, 'L': self.L_pix_size}))
        
        self.non_synced_saccades_dict = self.saccade_dict_enricher(non_sync_b_dict)
        self.non_synced_saccades_df = self.parse_dataset_to_df(self.non_synced_saccades_dict)
        self.non_synced_saccades_df['calib_dy'] = \
            self.non_synced_saccades_df['dy'] * \
            (self.non_synced_saccades_df['eye'].map({'R': self.R_pix_size, 'L': self.L_pix_size}))
        self.non_synced_saccades_df['calib_dx'] = \
            self.non_synced_saccades_df['dx'] * \
            (self.non_synced_saccades_df['eye'].map({'R': self.R_pix_size, 'L': self.L_pix_size}))

    @staticmethod
    def spherical_to_polar(yaw, pitch):
        """
        This function recieves yaw and pitch in radians and returns the polar coordinates
        :param yaw: theta values in radians
        :param pitch: phi values in radians
        :return: r: magnitude, theta: the angle (counter-clockwise from the positive x-axis)
        """

        # Calculate r
        r = math.sqrt(yaw ** 2 + pitch ** 2)

        # Calculate theta
        theta_rad = math.atan2(pitch, yaw)

        # Convert theta to degrees
        theta_deg = math.degrees(theta_rad)

        # Adjust theta to the range [0, 360)
        if theta_deg < 0:
            theta_deg += 360

        return r, theta_deg

    @staticmethod
    def saccade_before_after(coords):
        before = coords[0]
        after = coords[-1]
        delta = after - before
        return before, after, delta

    def saccade_dict_enricher(self, saccade_dict):
        """
        Helper function to enrich saccade dictionary before arranging it into a dataframe
        :param saccade_dict:
        :return:
        """

        for e in ['L', 'R']:
            saccade_dict[e]['x_speed'] = []
            saccade_dict[e]['y_speed'] = []
            saccade_dict[e]['magnitude'] = []
            saccade_dict[e]['dx'] = []  # TEMP
            saccade_dict[e]['dy'] = []  # TEMP
            saccade_dict[e]['direction'] = []
            saccade_dict[e]['saccade_initiation_x'] = []
            saccade_dict[e]['saccade_initiation_y'] = []
            saccade_dict[e]['saccade_termination_x'] = []
            saccade_dict[e]['saccade_termination_y'] = []

            for s in range(len(saccade_dict[e]['timestamps'])):
                # speed:
                saccade_dict[e]['x_speed'].append(np.insert(np.diff(saccade_dict[e]['x_coords'][s]), 0, float(0)))
                saccade_dict[e]['y_speed'].append(np.insert(np.diff(saccade_dict[e]['y_coords'][s]), 0, float(0)))
                saccade_length = saccade_dict[e]['length'][s]
                # Understand directionality and magnitude:
                # understand before and after - SHOULD THINK ABOUT THIS FOR FURTHER CORRECTION LATER
                saccade_initiation_ind = int(len(saccade_dict[e]['x_coords'][s])) // 2
                saccade_end_ind = saccade_initiation_ind + int(saccade_length)
                x_before, x_after, dx = (
                    self.saccade_before_after(saccade_dict[e]['x_coords'][s][saccade_initiation_ind:saccade_end_ind+1]))
                y_before, y_after, dy = (
                    self.saccade_before_after(saccade_dict[e]['y_coords'][s][saccade_initiation_ind:saccade_end_ind+1]))

                s_mag, theta = self.spherical_to_polar(dx, dy)

                # old version, with error!!!
                """
                s_mag = np.sqrt(dx ** 2 + dy ** 2)
                
                # get direction quadrant
                if dx > 0 and dy > 0:
                    quad = 0
                elif dx < 0 < dy:
                    quad = 1
                elif dx < 0 and dy < 0:
                    quad = 2
                elif dx > 0 > dy:
                    quad = 3
                # get direction (theta calculated from quadrent border)
                degrees_in_quadrent = np.rad2deg(np.arctan(np.abs(dy) / np.abs(dx)))
                theta = degrees_in_quadrent + (quad * 90)
                """

                # collect into dict
                saccade_dict[e]['dx'].append(dx)
                saccade_dict[e]['dy'].append(dy)
                saccade_dict[e]['magnitude'].append(s_mag)
                saccade_dict[e]['direction'].append(theta)
                saccade_dict[e]['saccade_initiation_x'].append(x_before)
                saccade_dict[e]['saccade_initiation_y'].append(y_before)
                saccade_dict[e]['saccade_termination_x'].append(x_after)
                saccade_dict[e]['saccade_termination_y'].append(y_after)

        return saccade_dict


