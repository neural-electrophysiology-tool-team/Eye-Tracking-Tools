import glob
import os
import pathlib
import subprocess as sp
import cv2
import numpy as np
import open_ephys.analysis as OEA
import pandas as pd
import scipy.stats as stats
from lxml import etree
from bokeh.plotting import figure, show
from ellipse import LsqEllipse
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
        self.analysis_path = self.block_path / 'analysis'
        self.l_e_path = self.block_path / 'eye_videos' / 'LE'
        self.l_e_path = self.l_e_path / os.listdir(self.l_e_path)[0]
        self.r_e_path = self.block_path / 'eye_videos' / 'RE'
        self.r_e_path = self.r_e_path / os.listdir(self.r_e_path)[0]
        if (self.analysis_path / 'arena_brightness.csv').exists():
            self.arena_brightness_df = pd.read_csv(self.analysis_path / 'arena_brightness.csv')
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
        self.sample_rate = self.get_sample_rate()
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
        sample_rate = None
        xml_tree = etree.parse(str(self.settings_xml))
        xml_root = xml_tree.getroot()
        for child in xml_root.iter():
            if child.tag == 'EDITOR':
                try:
                    sample_rate = int(float(child.attrib['SampleRateString'][:4]) * 1000)
                except KeyError:
                    continue
        if sample_rate is not None:
            print(f'The sample rate for block {self.block_num} is {sample_rate} Hz')
        else:
            print(f'could not find the sample rate for block_{self.block_num}')
        return sample_rate

    def oe_events_to_csv(self):
        """
        This method takes the open ephys events and puts them in a csv file

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
                print(arena_start_stop)
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

    @staticmethod
    def get_frame_timeseries(df, channel):
        index_range = range(0, len(df[channel][df[channel].notna()]))
        timeseries = pd.Series(df[channel][df[channel].notna()])
        timeseries = pd.Series(timeseries.values, index=index_range, name=channel)
        return timeseries

    def parse_open_ephys_events(self):
        """
        Gets the sample rate from the settings.xml file
        Creates the parsed_events.csv file
        finds the first and last frame timestamps for each video source

        """
        # First, create the events.csv file:
        self.oe_events_to_csv()
        # understand the samplerate and the first timestamp
        # if self.sample_rate is None:
        #     self.get_sample_rate()
        session = OEA.Session(str(self.oe_path))
        self.first_oe_timestamp = session.recordings[0].continuous[0].timestamps[0]
        # parse the events of the open-ephys recording

        ex_path = self.block_path / rf'oe_files\{self.exp_date_time}\parsed_events.csv'
        self.oe_events, self.arena_vid_first_t, self.arena_vid_last_t = self.oe_events_parser(
            self.block_path / rf'oe_files\{self.exp_date_time}\events.csv',
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

    def synchronize_block(self):
        """
        This method builds a synced_videos dataframe
        1. The arena video is used as anchor
        2. The different anchor timestamps are aligned with the closest frames of the other sources
        """
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
        for i, t in enumerate(arena_tf.Arena_TTL):
            arena_frame = arena_tf.Arena_TTL_frame.iloc[i]
            l_eye_frame = l_eye_tf['L_eye_TTL_frame'].iloc[self.get_closest_frame(t, l_eye_tf['L_eye_TTL'])]
            r_eye_frame = r_eye_tf['R_eye_TTL_frame'].iloc[self.get_closest_frame(t, r_eye_tf['R_eye_TTL'])]
            self.blocksync_df.loc[t] = [arena_frame, l_eye_frame, r_eye_frame]

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

    def create_eye_brightness_df(self, threshold_value=30, export=True):
        """
        This method creates the l/r_eye_values lists, which represent the illumination level of eye video frames
        :param threshold_value: The threshold value to use as mask before calculating brightness
        :return:
        """
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
