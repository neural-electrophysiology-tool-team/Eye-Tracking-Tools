import UtilityFunctions as uf
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import scipy.stats as stats

from matplotlib.patches import Ellipse
import itertools as itr


def open_ephys_events_parser(open_ephys_csv_path, channel_names, export_path=None):
    """
    :param open_ephys_csv_path: The path to an open ephys analysis tools exported csv (using TrialReporter.ipynb)
    :param channel_names: a dictionary of the form -
                    { 1 : 'channel name' (L_eye_camera)
                      2 : 'channel name' (Arena_TTL)
                      etc..
                    }
    :param export_path: default None, if a path is specified a csv file will be saved
    :returns open_ephys_events: a pandas DataFrame object where each column has the ON events of one channel and has a title from channel_names
    """

    # Infer the active channels:
    df = pd.read_csv(open_ephys_csv_path)
    channels = df['channel'].to_numpy(copy=True)
    channels = np.unique(channels)
    df_onstate = df[df['state'] == 1]  # cut the df to represent only rising edges
    list = []
    for chan in channels:  # extract a pandas series of the ON stats timestamps for each channel
        Sname = channel_names[chan]
        s = pd.Series(df_onstate['timestamp'][df_onstate['channel'] == chan], name=Sname)
        list.append(s)
    open_ephys_events = pd.concat(list, axis=1)
    if export_path is not None:
        open_ephys_events.to_csv(export_path)
    return open_ephys_events


def get_frame_timeseries(df, channel):
    index_range = range(0, len(df[channel][df[channel].notna()]))
    timeseries = pd.Series(df[channel][df[channel].notna()])
    timeseries = pd.Series(timeseries.values, index=index_range, name=channel)

    return timeseries


def get_frame_from_time(vid_timeseries, timestamp):
    array = np.abs((vid_timeseries.to_numpy())-timestamp)
    index_of_lowest_diff = np.argmin(array)
    accuracy = abs(vid_timeseries[index_of_lowest_diff] - timestamp)
    return index_of_lowest_diff, accuracy


def TTL_timeseries_synchronization(ts_list, anchor_vid_position):
    """
    param: ts_list: list of timeseries (one for each video) representing rising edge events in the open ephys file
    param: anchor_vid_position: the video from the ts_list to use as a sync reference (all other video frames will be fitted to this video)
    returns: synchronized_ttls: np.array where each column represents a video and each row represents the best fit for the anchor video frames (timewise)
    """
    anchor_vid = ts_list[anchor_vid_position].to_numpy()
    synchronized_ttls = []
    synchronization_accuracy = []
    for frame in range(len(anchor_vid)):
        time = anchor_vid[frame]
        if frame % 50 == 0:
            print(f'frame {frame} out of {len(anchor_vid)}', end='\r', flush=True)
        sync_frame = []
        sync_accuracy = []
        for vid in ts_list:
            f, a = get_frame_from_time(vid, time)
            sync_frame.append(f)
            sync_accuracy.append(a)
        synchronized_ttls.append(sync_frame)
        synchronization_accuracy.append(sync_accuracy)
    synchronized_ttls = np.array(synchronized_ttls)
    synchronization_accuracy = np.array(synchronization_accuracy)

    return synchronized_ttls, synchronization_accuracy


def get_timeseries_list(channel_events):
    """
    :param channel_events: the dataframe extracted from the open_ephys csv file
    :return: np.array of TTLs (rising edges)
    """
    ts_list = []
    for chan in list(channel_events.columns)[1:5]:
        ts = get_frame_timeseries(channel_events, str(chan))
        ts_list.append(ts)
    return ts_list
