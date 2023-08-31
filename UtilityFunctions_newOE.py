import numpy as np
import pathlib
import math
import tqdm
from open_ephys import analysis as oea
import scipy.io
from matplotlib import pyplot as plt
from BlockSync_current import *
from OERecording import *
import scipy.io
import h5py
import re
from lxml import etree as ET
import scipy.signal as sig
import pandas as pd
from scipy.stats import kde


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


def multi_block_saccade_dict_creation_current(blocklist, sampling_window_ms, ep_channel_number):

    saccade_dict = {}
    # loop over the blocks from here:
    for block in blocklist:
        # collect accelerometer data
        # path definition
        p = block.oe_path / 'analysis'
        analysis_list = os.listdir(p)
        correct_analysis = [i for i in analysis_list if block.animal_call in i][0]
        p = p / str(correct_analysis)
        matPath = p / 'lizMov.mat'
        print(f'path to mat file is {matPath}')
        # read mat file
        mat_data = h5py.File(str(matPath), 'r')
        mat_dict = {'t_mov_ms': mat_data['t_mov_ms'][:],
                    'movAll': mat_data['movAll'][:]}

        acc_df = pd.DataFrame(data=np.array([mat_dict['t_mov_ms'][:, 0], mat_dict['movAll'][:, 0]]).T,
                              columns=['t_mov_ms', 'movAll'])
        mat_data.close()

        block.saccade_event_analayzer(automatic=True, threshold=2)

        # create the top-level block dict object
        block_dict = {
            'L': {},
            'R': {}
        }

        # create and populate the internal dictionaries (for each eye)
        for i, e in enumerate(['L', 'R']):
            # get the correct saccades_chunked object and eye_df
            saccades_chunked = [block.l_saccades_chunked, block.r_saccades_chunked][i]
            eye_df = [block.le_df, block.re_df][i]
            saccades = saccades_chunked[saccades_chunked.saccade_length_frames > 0]
            saccade_times = np.sort(saccades.saccade_start_ms.values)
            ep_channel_numbers = [ep_channel_number]
            pre_saccade_ts = saccade_times - (sampling_window_ms / 2)

            # get the data of the relevant saccade time windows:
            print(f'calling get_data with the following inputs:'
                  f'eye = {e}'
                  f'block = {block}'
                  f'pre_saccade_ts = {pre_saccade_ts} \n'
                  f'sampling_window_ms = {sampling_window_ms}')
            ep_data, ep_timestamps = block.oe_rec.get_data(ep_channel_numbers, pre_saccade_ts, sampling_window_ms,
                                                           convert_to_mv=True)  # [n_channels, n_windows, nSamples]

            # start populating the dictionary
            block_dict[e] = {
                "timestamps": [],
                "fs": [],
                "pxx": [],
                "samples": [],
                "x_coords": [],
                "y_coords": [],
                "vid_inds": [],
                "accel": []
            }

            # go saccade by saccade
            for j in range(len(pre_saccade_ts)):
                # get specific saccade samples:
                saccade_samples = ep_data[0, j, :]  # [n_channels, n_windows, nSamples]
                # get the spectral profile for the segment
                fs, pxx = sig.welch(saccade_samples, block.sample_rate,nperseg=16384,return_onesided=True)

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
                    if nan_count > 0 :
                        if nan_count < len(y)/2:
                            # print(f'saccade at ind {i} has {nan_count} nans, interpolating...')
                            # find nan values in the vector
                            nans, z = nan_helper(y.astype(float))
                            # interpolate using the helper lambda function
                            y[nans] = np.interp(z(nans),z(~nans),y[~nans].astype(float))
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
                    block_dict[e]['timestamps'].append(pre_saccade_ts[j])
                    block_dict[e]['x_coords'].append(interpolated_coords[0])
                    block_dict[e]['y_coords'].append(interpolated_coords[1])
                    block_dict[e]['vid_inds'].append(vid_inds)
                    block_dict[e]['fs'].append(fs)
                    block_dict[e]['pxx'].append(pxx)
                    block_dict[e]['samples'].append(saccade_samples)
                    block_dict[e]['accel'].append(mov_mag)
        saccade_dict[block.block_num] = block_dict
    return saccade_dict


def sort_synced_saccades(b_dict):
    """
    This function takes a saccades dictionary and returns two sorted dictionaries - one with synced saccades, the other with non-synced saccades
    :param b_dict:
    :return:
    """
    # get the two timestamps vectors
    l_times = np.array(b_dict['L']['timestamps'])
    r_times = np.array(b_dict['R']['timestamps'])

    # I want to collect the matching indices from the L and R dictionaries and create a "synced saccades dict" object
    # that only has two-eyed saccades included in it...
    # first, I have to understand which rows of the dictionaries go together:
    # create a matrix of [left eye timestamp, -,left eye ind, -]
    s_mat = np.empty([len(l_times),5])
    s_mat[:,0] = l_times
    s_mat[:,2] = np.arange(0,len(l_times))
    # find and fit the right eye times and indices on columns 1 and 3
    for i, lt in enumerate(s_mat[:,0]):
        array = np.abs((r_times - lt))
        ind_min_diff = np.argmin(array)
        min_diff = array[ind_min_diff]
        rt = r_times[ind_min_diff]
        s_mat[i,3] = ind_min_diff
        s_mat[i,1] = rt
        s_mat[i,4] = min_diff

    # create a dataframe for queries and testing, define a threshold and remove non sync saccades
    s_df = pd.DataFrame(s_mat,columns=['lt','rt','left_ind','right_ind','diff'])
    threshold = 1400 # 70 ms to consider a saccade simultaneous
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
            "x_coords": np.array(b_dict[e]['x_coords'])[inds],
            "y_coords": np.array(b_dict[e]['y_coords'])[inds],
            "vid_inds": np.array(b_dict[e]['vid_inds'])[inds],
            "accel": np.array(b_dict[e]['accel'])[inds]
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
            "x_coords": np.array(b_dict[e]['x_coords'])[logical],
            "y_coords": np.array(b_dict[e]['y_coords'])[logical],
            "vid_inds": np.array(b_dict[e]['vid_inds'])[logical],
            "accel": np.array(b_dict[e]['accel'])[logical]
        }
    return synced_b_dict, non_sync_b_dict


def saccade_before_after(coords):
    max_ind = np.argmax(coords)
    min_ind = np.argmin(coords)
    if max_ind < min_ind:
        before = coords[max_ind]
        after = coords[min_ind]
    else:
        before = coords[min_ind]
        after = coords[max_ind]
    delta = after-before
    return before, after, delta


def saccade_dict_enricher(saccade_dict):
    for k in saccade_dict.keys():
        sync_saccades = saccade_dict[k]
        for e in ['L', 'R']:
            sync_saccades[e]['x_speed'] = []
            sync_saccades[e]['y_speed'] = []
            sync_saccades[e]['magnitude'] = []
            sync_saccades[e]['dx'] = []  # TEMP
            sync_saccades[e]['dy'] = []  # TEMP
            sync_saccades[e]['direction'] = []

            for s in range(len(sync_saccades[e]['timestamps'])):
                # speed:
                sync_saccades[e]['x_speed'].append(np.insert(np.diff(sync_saccades[e]['x_coords'][s]), 0, float(0)))
                sync_saccades[e]['y_speed'].append(np.insert(np.diff(sync_saccades[e]['y_coords'][s]), 0, float(0)))

                # Understand directionality and magnitude:
                # understand before and after
                x_before, x_after, dx = saccade_before_after(sync_saccades[e]['x_coords'][s])
                y_before, y_after, dy = saccade_before_after(sync_saccades[e]['y_coords'][s])

                # calculate magnitude (euclidean)
                s_mag = np.sqrt(dx**2 + dy**2)

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
                degrees_in_quadrent = np.rad2deg(np.arctan(np.abs(dy)/np.abs(dx)))
                theta = degrees_in_quadrent + (quad*90)

                # collect into dict
                sync_saccades[e]['dx'].append(dx)
                sync_saccades[e]['dy'].append(dy)
                sync_saccades[e]['magnitude'].append(s_mag)
                sync_saccades[e]['direction'].append(theta)
        saccade_dict[k] = sync_saccades
    return saccade_dict


def parse_dataset_to_df(saccade_dict, blocklist):

    date_list = [block.oe_dirname[-19:] for block in blocklist]
    num_list = [block.block_num for block in blocklist]
    num_date_dict = dict(zip(num_list, date_list))
    df = pd.DataFrame(columns=['datetime', 'block', 'eye', 'timestamps', 'fs', 'pxx', 'samples', 'x_coords', 'y_coords',
                               'vid_inds', 'x_speed', 'y_speed', 'magnitude', 'dx', 'dy', 'direction', 'accel'])
    d = saccade_dict
    index_counter = 0
    for k in d.keys():
        block = d[k] # in a certain block
        for e in block.keys():
            eye = block[e]  # in one of the eyes
            for row in range(len(eye['samples'])): # for each saccade
                for col in eye.keys():  # for each columm
                    v = eye[col][row]  # get value of location
                    df.at[index_counter, 'block'] = k
                    df.at[index_counter, 'eye'] = e
                    df.at[index_counter, 'datetime'] = num_date_dict[k]
                    df.at[index_counter, col] = v
                print(index_counter, end='\r', flush=True)
                index_counter += 1
    print(f'done, dataframe contains {index_counter} saccades')
    return df


def plot_kde(ax, x, y, nbins, title, xlim=False, ylim=False, global_max=None, global_min=None):
    k = kde.gaussian_kde(np.array([x,y]).astype(np.float))

    if global_max and global_min:
        minimal_coordinate= global_min
        maximal_coordinate = global_max
    else:
        minimal_coordinate= min([x.min(),y.min()])
        maximal_coordinate = max([x.max(),y.max()])
    xi, yi = np.mgrid[minimal_coordinate:maximal_coordinate:nbins*1j, minimal_coordinate:maximal_coordinate:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.set_title(str(title))
    sp = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.jet)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_aspect('equal','box')
    return sp


def block_generator(block_numbers, experiment_path, animal, bad_blocks=[], regev=True):
    """
    creates a block_collection to iterate over with multi-block functions
    :param block_numbers: list of block numbers to loop over
    :param experiment_path: pathlib.Path instance to the experiment folder
    :param animal: string
    :param bad_blocks: blocks to ignore
    :return:
    """
    p = pathlib.Path(experiment_path) / animal
    date_folder_list = [i for i in p.iterdir() if 'block' not in str(i).lower() and i.is_dir()]
    block_collection = []
    for date_path in date_folder_list:
        date = date_path.name
        # list all the blocks in the folder:
        folder_list = [i for i in date_path.iterdir()]
        for block in folder_list:
            if 'block' in str(block):
                block_number = block.name[-3:]
                try:
                    if int(block_number) in block_numbers and int(block_number) not in bad_blocks:
                        # block definition
                        block = BlockSync(animal_call=animal,
                                          experiment_date=date,block_num=block_number,
                                          path_to_animal_folder=str(experiment_path),regev=regev)
                        block_collection.append(block)
                except ValueError:
                    continue
    return block_collection


def create_video_from_segments(segments_df, blocklist, export_path):
    """
    This creates a video of connected segments out of the segments_df dataframe and saves it in the export path
    :param segments_df: a dataframe containing the block number, start ttl , stop ttl for each segment
    :param blocklist: a list with BlockSync objects that are involved with creating the segments_df
    :param export_path:
    :return:
    """
    # some stuff for the text
    font = cv2.FONT_HERSHEY_SIMPLEX

    df = segments_df
    blockdict = {}
    for b in blocklist:
        b_num = b.block_num
        blockdict[b_num] = b

    # define the video writer
    vid_out = cv2.VideoWriter(str(export_path),
                              cv2.VideoWriter_fourcc('H', '2', '6', '4'),
                              60.0, (640*2, 480))  # set the frame size to be two vid's worth

    # for each block of the dataframe (each segment):
    for b in np.unique(df['block'].values):
        block = blockdict[b]
        b_df = df.query('block == @b')

        # define the eye VideoCaptures
        rcap = cv2.VideoCapture(block.re_videos[0])
        lcap = cv2.VideoCapture(block.le_videos[0])

        # for each row of the block df (each video segment):
        for row in tqdm(b_df.index):
            # understand the ttls
            ttl0 = b_df.at[row, 'start']
            ttl1 = b_df.at[row, 'stop']

            # figure out the frames for each eye
            seg_df = block.final_sync_df.query("Arena_TTL > @ttl0 and Arena_TTL < @ttl1")
            r_frames = seg_df['R_eye_frame']
            l_frames = seg_df['L_eye_frame']

            # initialize the time vector for the segment
            last_frame_L = 0
            last_frame_R = 0
            time_vec = (seg_df['Arena_TTL'].values - block.final_sync_df.iloc[0]['Arena_TTL']) / block.sample_rate

            # for each synced timestamp:
            for i, t in enumerate(time_vec):

                # get frame numbers from the df
                rf_num = r_frames.iloc[i]
                lf_num = l_frames.iloc[i]
                if rf_num != rf_num:
                    rf_num = last_frame_R
                if lf_num != lf_num:
                    lf_num = last_frame_L

                # read the frame (and make sure to conserve setting steps) - BOTH EYES
                if last_frame_L + 1 != lf_num:
                    lcap.set(1, int(lf_num))
                l_ret, l_f = lcap.read()
                l_f = cv2.cvtColor(l_f, cv2.COLOR_BGR2GRAY)
                l_f = cv2.flip(l_f, 0)
                l_f = cv2.resize(l_f, (640, 480))

                last_frame_L = lf_num

                if last_frame_R + 1 != rf_num:
                    rcap.set(1, int(rf_num))
                r_ret, r_f = rcap.read()
                r_f = cv2.cvtColor(r_f, cv2.COLOR_BGR2GRAY)
                r_f = cv2.flip(r_f, 0)
                r_f = cv2.resize(r_f, (640, 480))
                last_frame_R = rf_num

                if r_ret and l_ret:
                    eye_concat = np.hstack((l_f, r_f))
                    eye_concat = cv2.putText(eye_concat, f'Block {block.block_num}, Timestamp [Seconds]: {t} ',
                                             org=(10, 450), fontFace=font, fontScale=1, color=0, thickness=2)
                    vid_out.write(eye_concat)
                    cv2.imshow('frame_view', eye_concat)

                    key = cv2.waitKey(1)

                    if key == ord('q'):
                        break

        rcap.release()
        lcap.release()
    vid_out.release()
    cv2.destroyAllWindows()