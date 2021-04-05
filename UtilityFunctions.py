import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import scipy.stats as stats
from ellipse import LsqEllipse

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

def get_z_mad_values(frame_val, calc_mad=True, calc_zscores=True):
    """

    :param calc_zscores: True for z-score based calculation
    :param calc_mad: True for mean absolute deviation calculation
    :param frame_val: an array containing [indexes , values] of the video frames
    :return:
    """
    values = frame_val[1]
    indexes = frame_val[0]
    if calc_mad:
        mad = stats.median_abs_deviation(values)
    if calc_zscores:
        zscores = stats.zscore(values)
    if calc_mad and calc_zscores:
        return z, mad
    elif calc_mad and not calc_zscores:
        return mad
    elif not calc_mad and calc_zscores:
        return z

def compress_frames(led_off_frames_ndarry):
    clustered_off_frames = []
    if type(led_off_frames_ndarry) == type(np.array([])):
        led_off_frames = led_off_frames_ndarry.tolist()
    else:
        led_off_frames = led_off_frames_ndarry
    if len(led_off_frames) == 0:
        print('no frames in the list')
        return []
    clust_count = 0
    start_clust = 0
    end_clust = 0
    for c in range(len(led_off_frames)-1):
        if clust_count == 0:
            start_clust = led_off_frames[c]
            end_clust = 0
        if led_off_frames[c] == led_off_frames[c+1] - 1:
            clust_count += 1
            continue
        else:
            end_clust = led_off_frames[c]

        if start_clust == end_clust:
            clustered_off_frames.append(start_clust)
        elif end_clust > start_clust:
            clustered_off_frames.append(start_clust)
            clustered_off_frames.append(end_clust)
        clust_count = 0
    if clust_count != 0:
        clustered_off_frames.append(start_clust)
        clustered_off_frames.append(led_off_frames[c])

    clustered_off_frames.append(led_off_frames[-1])
    return clustered_off_frames

def led_off_frame_validation(vidpath, led_off_frames):
    validated_led_off = []
    cap = cv2.VideoCapture(vidpath)
    breaker = 0
    i = 0
    while breaker == 0:
        if i == len(led_off_frames):
            breaker = 1
        if breaker == 1:
            break
        f = led_off_frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        frame = cv2.putText(frame, f'frame number {str(f)}', (10, 10), 1, 5, (255, 255, 0), 3)
        cv2.imshow('validation', frame)
        key = ord('a')
        while key not in [ord('q'), ord('k'), ord('s'), ord('b')]:
            key = cv2.waitKey(0)
            if key == ord('q'):
                breaker = 1
                break
            elif key == ord('k'):
                if not f in validated_led_off:
                    validated_led_off.append(f)
                i += 1
                break
            elif key == ord('s'):
                if f in validated_led_off:
                    validated_led_off.remove(f)
                i+=1
                break
            elif key == ord('b'):
                i = i - 1
                break

    cap.release()
    cv2.destroyAllWindows()
    return validated_led_off

#def ellipse_fit(points_per_frame):

