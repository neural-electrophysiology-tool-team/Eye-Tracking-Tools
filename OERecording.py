import numpy as np
import h5py
import re


class OERecording:
    # define a helper function to get a group out of an hdf5 file (without resolving internal references!)
    """
    This is a class designed to get the metadata of an open-ephys format recording that was first analyzed by Mark's
    matlab code - I should probably make an independent function sometime in the future but for now this class reads
    the produced metadata file to allow efficient access to the recording and provides the get_data function
    """

    def group_to_dict(self, group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = self.group_to_dict(item)
            else:

                result[key] = item[()]  # Convert dataset to NumPy array and assign its value
        return result

    # a helper function to sort recording files by their correct numbering scheme
    @staticmethod
    def extract_number_from_file(filename, suffix):
        match = re.search(r'(\d+)\.' + suffix + '$', filename)
        if match:
            return int(match.group(1))
        else:
            return None

    def __init__(self, oe_metadata_file_path):
        # initialize some critical variables (currently taken from the metadata file, should become independent!)
        self.allTimeStamps = None
        self.sample_ms = None
        self.channelNumbers = None
        self.bytesPerRecCont = None
        self.recordLength = None
        self.blkCont = None
        self.blkBytesCont = None
        self.MicrovoltsPerAD = None
        self.mat_file = None
        self.globalStartTime_ms = None

        # create the metadata_dict object:
        # open the mat file:
        try:
            self.mat_file = h5py.File(str(oe_metadata_file_path), 'r')
        except Exception:
            print(f'an error occured while trying to reach {str(oe_metadata_file_path)}, check matlab output format!')
            if self.mat_file is not None:
                self.mat_file.close()

        # implement on the metaData group:
        meta_dict = self.group_to_dict(self.mat_file['metaData'])

        # resolve internal references of blkCont object:
        # get to the blkCont mat and reform it from reference instances
        blk_cont_dict = {
            'Repeat': [],
            'Types': [],
            'Str': []
        }
        blk_cont_group = self.mat_file['metaData/blkCont']
        for i in blk_cont_group['Repeat']:
            res = np.array(self.mat_file[i[0]][0])
            blk_cont_dict['Repeat'].append(res[0])

        for i in blk_cont_group['Types']:
            res = np.array((self.mat_file[i[0]]))
            str_array = np.vectorize(chr)(res).flatten()
            str_value = ''.join(str_array.flatten())
            blk_cont_dict['Types'].append(str_value)

        for i in blk_cont_group['Str']:
            res = np.array(self.mat_file[i[0]])
            str_array = np.vectorize(chr)(res).flatten()
            str_value = ''.join(str_array.flatten())
            blk_cont_dict['Str'].append(str_value)

        # close the file
        self.mat_file.close()

        # switch out the dictionary blkCont attribute
        meta_dict['blkCont'] = blk_cont_dict
        # parse a dictionary into attributes of the class
        for key, value in meta_dict.items():
            setattr(self, key, value)

        # add some stuff from Mark's class that's not in the metadata
        self.headerSizeByte = 1024
        self.fileExtension = 'continuous'
        self.eventFileExtension = 'events'
        self.signalBits = 16  # the quantization of the sampling card
        self.dataSamplesPerRecord = 1024
        self.maxTTLBit = 9
        self.oe_file_path = oe_metadata_file_path.parent

        # get the channel files but also sort them by their true numbering scheme
        self.channel_files = sorted([i.name for i in oe_metadata_file_path.parent.iterdir() if
                                     ('.continuous' in str(i)) & ('AUX' not in str(i)) & ('ADC' not in str(i))],
                                    key=lambda x: self.extract_number_from_file(x, suffix='continuous'))

        self.analog_files = sorted([i.name for i in oe_metadata_file_path.parent.iterdir() if 'ADC' in str(i)],
                                   key=lambda x: self.extract_number_from_file(x, suffix='continuous'))
        self.accel_files = sorted([i.name for i in oe_metadata_file_path.parent.iterdir() if ('AUX' in str(i))],
                                  key=lambda x: self.extract_number_from_file(x, suffix='continuous'))

    def get_data(self, channels,
                 start_time_ms,
                 window_ms,
                 convert_to_mv=True,
                 return_timestamps=True,
                 repress_output=False):
        """
        This is a translated matlab function that efficiently retrieves data from Open-Ephys format neural recordings
        :param self: an OERecording class obj. with a metadata file created by the matlab class with the same name
        :param channels: a vector of channel numbers to sample from [1XN channels]
        :param start_time_ms: a vector of window start times [1XN] in ms
        :param window_ms: a single value, the length of the sampling window from each startTime [ms_value]
        :param convert_to_mv: when True, turns the output into the mV representation of the sampled data
        :param return_timestamps: when True, the output will include sample timestamps from 0 in ms
        :param repress_output: when True, will not perform print commands
        :return: data_matrix - an array with the shape [n_channels, n_windows, nSamples] with int16 / mV values
        """
        window_samples = int(
            np.round(window_ms / self.sample_ms))  # round the time in ms to the nearest whole sample count
        n_windows = len(start_time_ms)  # get the number of start times provided
        start_time_ms = np.round(
            start_time_ms / self.sample_ms) * self.sample_ms  # round the start times to the nearest whole sample step
        window_ms = window_samples * self.sample_ms  # get the ms based length of the rounded window

        # deal with the channel numbers:
        if len(channels) == 0 or channels is None:  # if no channels were provided
            channels = self.channelNumbers

        if not all([c in self.channelNumbers for c in channels]):  # if requested channels do not exist in the file
            raise ValueError('one or more of the entered channels does not exist in the recording!')
        n_ch = len(channels)

        # initialize some variables for the data extraction:
        # waveform matrix:
        data_matrix = np.zeros(shape=(int(window_samples), n_windows, n_ch),
                               dtype=self.blkCont['Types'][3],
                               order='F')
        # List to store the record indices for waveform extraction:
        p_rec_idx = []
        # List to store the indices where reading from the file should start (one per reading window):
        read_start_indices = []
        records_per_trial_list = []
        for i in range(n_windows):
            # find the relevant record blocks in the block list:
            p_single_trial_time_stamps = np.where((self.allTimeStamps[0] >= (start_time_ms[0][i] - self.recordLength)) &
                                                  (self.allTimeStamps[0] < (start_time_ms[0][i] + window_ms)))[1]
            try:
                # this collects the indices to start reading from
                read_start_indices.append(p_single_trial_time_stamps[0])
            except IndexError:
                read_start_indices.append(p_single_trial_time_stamps)

            # Calculate time stamps in milliseconds based on sampling freq & record block length
            single_trial_time_stamps = np.round(self.allTimeStamps[0][
                                                    p_single_trial_time_stamps] / self.sample_ms) * self.sample_ms
            # Get the number of records per trial & append to a list
            records_per_trial = len(single_trial_time_stamps[0])
            records_per_trial_list.append(records_per_trial)

            # get the real time values for each sample index:
            time_idx = np.tile((np.arange(self.dataSamplesPerRecord) * self.sample_ms).reshape(-1, 1),
                               (1, records_per_trial)) + single_trial_time_stamps.reshape(1, -1)
            # Find time indices within the requested time window
            # (chunks are 1024 in size so they are usually cut for most time windows, result is as a boolean matrix)
            p_rec_idx.append((time_idx >= start_time_ms[0][i]) & (time_idx < (start_time_ms[0][i] + window_ms)))

            # Due to rounding issues, there may be an error when there is one sample too much -
            # in this case the last sample is removed
            if np.sum(p_rec_idx[i]) == window_samples + 1:
                if repress_output is not True:
                    print(f'sample removed for window #{i}')
                p_rec_idx[i][0, np.where(p_rec_idx[i][0, :] == 1)[0][0]] = False

        p_rec_idx = np.hstack(p_rec_idx)  # Concatenate record indices into a single array

        # now for the data extraction itself:
        for i in range(n_ch):  # iterate over channels
            data = np.zeros(p_rec_idx.shape, dtype=np.dtype('>i2'))  # Initialize the data array for a specific channel
            curr_rec = 0  # for this channel, initialize the record counter
            c_file = self.oe_file_path / self.channel_files[channels[i] - 1]  # get path of current channel file
            with open(c_file, 'rb') as fid:  # open the file such that it will close when left alone
                for j in range(n_windows):  # Iterate over sampling windows
                    # use seek to go to the appropriate position in the file
                    fid.seek(int(self.headerSizeByte + (read_start_indices[j] * self.bytesPerRecCont) + np.sum(
                        self.blkBytesCont[0:3])), 0)
                    # calculate the skip size, cut in half because each int16 is 2 bytes and the matlab
                    # function takes bytes as skip (which fromfile does not, uniform datatype)
                    skip_size = int(self.bytesPerRecCont[0][0] - self.blkBytesCont[3][0]) // 2
                    read_size = self.dataSamplesPerRecord
                    # calculate total element count to read:
                    total_bytes = (read_size + skip_size) * records_per_trial_list[j]
                    # read data from file in a single vector, including skip_data:
                    # (Notice datatype is non-flexible in this version of the function!!!)
                    data_plus_breaks = np.fromfile(fid, dtype=np.dtype('>i2'), count=total_bytes, sep='')
                    # reshape into an array with a column-per-record shape:
                    try:
                        data_plus_breaks = data_plus_breaks.reshape(int(records_per_trial_list[j]),
                                                                    read_size + skip_size)
                    except ValueError:
                        print('There was a problem reshaping ...', data_plus_breaks)
                        if return_timestamps:
                            return None, None
                        else:
                            return None

                    # slice the array to get rid of the skip_data at the end of each column (record):
                    clean_data = data_plus_breaks[:, : read_size]
                    # transpose and store the current_rec data:
                    data[:, curr_rec: curr_rec + records_per_trial_list[j]] = clean_data.T
                    curr_rec = curr_rec + records_per_trial_list[j]  # move forward to the next reading window
            # this loop exit closes the current channel file
            # vectorize the data from the channel and perform a boolean snipping of non-window samples:
            data_vec = data.T[p_rec_idx.T]

            # put the data in the final data_matrix waveform matrix:
            # check for end-of-recording exceedance :
            if len(data_vec) < int(window_samples) * n_windows:
                if repress_output is not True:
                    print(f'The requested data segment between {read_start_indices[j]} ms and '
                          f'{read_start_indices[j] + window_ms} ms exceeds the recording length, '
                          f'and will be 0-padded to fit the other windows')
                num_zeros = (int(window_samples) * n_windows) - len(data_vec)
                data_vec = np.pad(data_vec, (0, num_zeros), mode='constant')
            data_matrix[:, :, i] = data_vec.reshape(int(window_samples), n_windows, order='F')

        data_matrix = np.transpose(data_matrix, [2, 1, 0])

        if convert_to_mv:
            data_matrix = data_matrix * self.MicrovoltsPerAD[0]

        if return_timestamps:
            timestamps = np.tile(np.arange(window_samples) * self.sample_ms, (n_windows, 1))
            start_times = np.tile(start_time_ms.T, window_samples)
            timestamps = timestamps + start_times
            return data_matrix, timestamps
        else:
            return data_matrix

    def get_analog_data(self, channels, start_time_ms, window_ms, convert_to_mv=True, return_timestamps=True):
        """
        This is a translated matlab function that efficiently retrieves data from Open-Ephys format neural recordings
        :param self: an OERecording class obj. with a metadata file created by the matlab class with the same name
        :param channels: a vector of channel numbers to sample from [1XN channels]
        :param start_time_ms: a vector of window start times [1XN] in ms
        :param window_ms: a single value, the length of the sampling window from each startTime [ms_value]
        :param convert_to_mv: when True, turns the output into the mV representation of the sampled data
        :param return_timestamps: when True, the output will include sample timestamps from 0 in ms
        :return: data_matrix - an array with the shape [n_channels, n_windows, nSamples] with int16 / mV values
        """
        window_samples = int(
            np.round(window_ms / self.sample_ms))  # round the time in ms to the nearest whole sample count
        n_windows = len(start_time_ms)  # get the number of start times provided
        start_time_ms = np.round(
            start_time_ms / self.sample_ms) * self.sample_ms  # round the start times to the nearest whole sample step
        window_ms = window_samples * self.sample_ms  # get the ms based length of the rounded window

        # deal with the channel numbers:
        if not channels:  # if no channels were provided
            channels = self.analogChannelNumbers

        if not all([c in self.channelNumbers for c in channels]):  # if requested channels do not exist in the file
            raise ValueError('one or more of the entered channels does not exist in the recording!')
        n_ch = len(channels)

        # initialize some variables for the data extraction:
        # waveform matrix:
        data_matrix = np.zeros(shape=(int(window_samples), n_windows, n_ch),
                               dtype=self.blkCont['Types'][3],
                               order='F')
        # List to store the record indices for waveform extraction:
        p_rec_idx = []
        # List to store the indices where reading from the file should start (one per reading window):
        read_start_indices = []
        records_per_trial_list = []
        for i in range(n_windows):
            # find the relevant record blocks in the block list:
            p_single_trial_time_stamps = np.where((self.allTimeStamps[0] >= (start_time_ms[0][i] - self.recordLength)) &
                                                  (self.allTimeStamps[0] < (start_time_ms[0][i] + window_ms)))[1]
            try:
                # this collects the indices to start reading from
                read_start_indices.append(p_single_trial_time_stamps[0])
            except IndexError:
                print('Index error again - FIX ME PLEASE')
                read_start_indices.append(p_single_trial_time_stamps)
                print(p_single_trial_time_stamps)  # This is debug step 1
                print(type(p_single_trial_time_stamps))

            # Calculate time stamps in milliseconds based on sampling freq & record block length
            single_trial_time_stamps = np.round(self.allTimeStamps[0][
                                                    p_single_trial_time_stamps] / self.sample_ms) * self.sample_ms
            # Get the number of records per trial & append to a list
            records_per_trial = len(single_trial_time_stamps[0])
            records_per_trial_list.append(records_per_trial)

            # get the real time values for each sample index:
            time_idx = np.tile((np.arange(self.dataSamplesPerRecord) * self.sample_ms).reshape(-1, 1),
                               (1, records_per_trial)) + single_trial_time_stamps.reshape(1, -1)
            # Find time indices within the requested time window
            # (chunks are 1024 in size, so they are usually cut for most time windows, result is as a boolean matrix)
            p_rec_idx.append((time_idx >= start_time_ms[0][i]) & (time_idx < (start_time_ms[0][i] + window_ms)))

            # Due to rounding issues, there may be an error when there is one sample too much -
            # in this case the last sample is removed
            if np.sum(p_rec_idx[i]) == window_samples + 1:
                print(f'sample removed for window #{i}')
                p_rec_idx[i][0, np.where(p_rec_idx[i][0, :] == 1)[0][0]] = False

        p_rec_idx = np.hstack(p_rec_idx)  # Concatenate record indices into a single array

        # now for the data extraction itself:
        for i in range(n_ch):  # iterate over channels
            data = np.zeros(p_rec_idx.shape, dtype=np.dtype('>i2'))  # Initialize the data array for a specific channel
            curr_rec = 0  # for this channel, initialize the record counter
            c_file = self.oe_file_path / self.analog_files[channels[i] - 1]  # get path of current channel file
            with open(c_file, 'rb') as fid:  # open the file such that it will close when left alone
                for j in range(n_windows):  # Iterate over sampling windows
                    # use seek to go to the appropriate position in the file
                    fid.seek(int(self.headerSizeByte + (read_start_indices[j] * self.bytesPerRecCont) + np.sum(
                        self.blkBytesCont[0:3])), 0)
                    # calculate the skip size, cut in half because each int16 is 2 bytes and the matlab
                    # function takes bytes as skip (which fromfile does not, uniform datatype)
                    skip_size = int(self.bytesPerRecCont[0][0] - self.blkBytesCont[3][0]) // 2
                    read_size = self.dataSamplesPerRecord
                    # calculate total element count to read:
                    total_bytes = (read_size + skip_size) * records_per_trial_list[j]
                    # read data from file in a single vector, including skip_data:
                    # (Notice datatype is non-flexible in this version of the function!!!)
                    data_plus_breaks = np.fromfile(fid, dtype=np.dtype('>i2'), count=total_bytes, sep='')
                    # reshape into an array with a column-per-record shape:
                    data_plus_breaks = data_plus_breaks.reshape(int(records_per_trial_list[j]), read_size + skip_size)
                    # slice the array to get rid of the skip_data at the end of each column (record):
                    clean_data = data_plus_breaks[:, : read_size]
                    # transpose and store the current_rec data:
                    data[:, curr_rec: curr_rec + records_per_trial_list[j]] = clean_data.T
                    curr_rec = curr_rec + records_per_trial_list[j]  # move forward to the next reading window
            # this loop exit closes the current channel file
            # vectorize the data from the channel and perform a boolean snipping of non-window samples:
            data_vec = data.T[p_rec_idx.T]

            # put the data in the final data_matrix waveform matrix:
            # check for end-of-recording exceedance :
            if len(data_vec) < int(window_samples) * n_windows:
                print(f'The requested data segment between {read_start_indices[j]} ms and '
                      f'{read_start_indices[j] + window_ms} ms exceeds the recording length, '
                      f'and will be 0-padded to fit the other windows')
                num_zeros = (int(window_samples) * n_windows) - len(data_vec)
                data_vec = np.pad(data_vec, (0, num_zeros), mode='constant')
            data_matrix[:, :, i] = data_vec.reshape(int(window_samples), n_windows, order='F')

        data_matrix = np.transpose(data_matrix, [2, 1, 0])

        if convert_to_mv:
            data_matrix = data_matrix * self.MicrovoltsPerADAnalog[-1]

        if return_timestamps:
            timestamps = np.tile(np.arange(window_samples) * self.sample_ms, (n_windows, 1))
            start_times = np.tile(start_time_ms.T, window_samples)
            timestamps = timestamps + start_times
            return data_matrix, timestamps
        else:
            return data_matrix

    def get_accel_data(self, channels, start_time_ms, window_ms, convert_to_mv=True, return_timestamps=True,
                       direct_paths_to_files=None):
        """
        This is a translated matlab function that efficiently retrieves data from Open-Ephys format neural recordings
        :param self: an OERecording class obj. with a metadata file created by the matlab class with the same name
        :param channels: a vector of channel numbers to sample from [1XN channels]
        :param start_time_ms: a vector of window start times [1XN] in ms
        :param window_ms: a single value, the length of the sampling window from each startTime [ms_value]
        :param convert_to_mv: when True, turns the output into the mV representation of the sampled data
        :param return_timestamps: when True, the output will include sample timestamps from 0 in ms
        :return: data_matrix - an array with the shape [n_channels, n_windows, nSamples] with int16 / mV values
        """
        window_samples = int(
            np.round(window_ms / self.sample_ms))  # round the time in ms to the nearest whole sample count
        n_windows = len(start_time_ms)  # get the number of start times provided
        start_time_ms = np.round(
            start_time_ms / self.sample_ms) * self.sample_ms  # round the start times to the nearest whole sample step
        window_ms = window_samples * self.sample_ms  # get the ms based length of the rounded window

        # deal with the channel numbers:
        if not channels:  # if no channels were provided
            channels = self.analogChannelNumbers

        if not all([c in self.analogChannelNumbers for c in channels]):  # if requested channels do not exist in the file
            raise ValueError('one or more of the entered channels does not exist in the recording!')
        n_ch = len(channels)

        # initialize some variables for the data extraction:
        # waveform matrix:
        data_matrix = np.zeros(shape=(int(window_samples), n_windows, n_ch),
                               dtype=self.blkCont['Types'][3],
                               order='F')
        # List to store the record indices for waveform extraction:
        p_rec_idx = []
        # List to store the indices where reading from the file should start (one per reading window):
        read_start_indices = []
        records_per_trial_list = []
        for i in range(n_windows):
            # find the relevant record blocks in the block list:
            p_single_trial_time_stamps = np.where((self.allTimeStamps[0] >= (start_time_ms[0][i] - self.recordLength)) &
                                                  (self.allTimeStamps[0] < (start_time_ms[0][i] + window_ms)))[1]
            try:
                # this collects the indices to start reading from
                read_start_indices.append(p_single_trial_time_stamps[0])
            except IndexError:
                print('hi')
                read_start_indices.append(p_single_trial_time_stamps)

            # Calculate time stamps in milliseconds based on sampling freq & record block length
            single_trial_time_stamps = np.round(self.allTimeStamps[0][
                                                    p_single_trial_time_stamps] / self.sample_ms) * self.sample_ms
            # Get the number of records per trial & append to a list
            records_per_trial = len(single_trial_time_stamps[0])
            records_per_trial_list.append(records_per_trial)

            # get the real time values for each sample index:
            time_idx = np.tile((np.arange(self.dataSamplesPerRecord) * self.sample_ms).reshape(-1, 1),
                               (1, records_per_trial)) + single_trial_time_stamps.reshape(1, -1)
            # Find time indices within the requested time window
            # (chunks are 1024 in size so they are usually cut for most time windows, result is as a boolean matrix)
            p_rec_idx.append((time_idx >= start_time_ms[0][i]) & (time_idx < (start_time_ms[0][i] + window_ms)))

            # Due to rounding issues, there may be an error when there is one sample too much -
            # in this case the last sample is removed
            if np.sum(p_rec_idx[i]) == window_samples + 1:
                print(f'sample removed for window #{i}')
                p_rec_idx[i][0, np.where(p_rec_idx[i][0, :] == 1)[0][0]] = False

        p_rec_idx = np.hstack(p_rec_idx)  # Concatenate record indices into a single array

        # now for the data extraction itself:
        for i in range(n_ch):  # iterate over channels
            data = np.zeros(p_rec_idx.shape, dtype=np.dtype('>i2'))  # Initialize the data array for a specific channel
            curr_rec = 0  # for this channel, initialize the record counter
            if direct_paths_to_files is None:
                c_file = self.oe_file_path / self.accel_files[channels[i] - 1]  # get path of current channel file
            else:
                c_file = self.oe_file_path / direct_paths_to_files[i] # get the direct path of current channel file
            with open(c_file, 'rb') as fid:  # open the file such that it will close when left alone
                for j in range(n_windows):  # Iterate over sampling windows
                    # use seek to go to the appropriate position in the file
                    fid.seek(int(self.headerSizeByte + (read_start_indices[j] * self.bytesPerRecCont) + np.sum(
                        self.blkBytesCont[0:3])), 0)
                    # calculate the skip size, cut in half because each int16 is 2 bytes and the matlab
                    # function takes bytes as skip (which fromfile does not, uniform datatype)
                    skip_size = int(self.bytesPerRecCont[0][0] - self.blkBytesCont[3][0]) // 2
                    read_size = self.dataSamplesPerRecord
                    # calculate total element count to read:
                    total_bytes = (read_size + skip_size) * records_per_trial_list[j]
                    # read data from file in a single vector, including skip_data:
                    # (Notice datatype is non-flexible in this version of the function!!!)
                    data_plus_breaks = np.fromfile(fid, dtype=np.dtype('>i2'), count=total_bytes, sep='')
                    # reshape into an array with a column-per-record shape:
                    data_plus_breaks = data_plus_breaks.reshape(int(records_per_trial_list[j]), read_size + skip_size)
                    # slice the array to get rid of the skip_data at the end of each column (record):
                    clean_data = data_plus_breaks[:, : read_size]
                    # transpose and store the current_rec data:
                    data[:, curr_rec: curr_rec + records_per_trial_list[j]] = clean_data.T
                    curr_rec = curr_rec + records_per_trial_list[j]  # move forward to the next reading window
            # this loop exit closes the current channel file
            # vectorize the data from the channel and perform a boolean snipping of non-window samples:
            data_vec = data.T[p_rec_idx.T]

            # put the data in the final data_matrix waveform matrix:
            # check for end-of-recording exceedance :
            if len(data_vec) < int(window_samples) * n_windows:
                print(f'The requested data segment between {read_start_indices[j]} ms and '
                      f'{read_start_indices[j] + window_ms} ms exceeds the recording length, '
                      f'and will be 0-padded to fit the other windows')
                num_zeros = (int(window_samples) * n_windows) - len(data_vec)
                data_vec = np.pad(data_vec, (0, num_zeros), mode='constant')
            data_matrix[:, :, i] = data_vec.reshape(int(window_samples), n_windows, order='F')

        data_matrix = np.transpose(data_matrix, [2, 1, 0])

        if convert_to_mv:
            data_matrix = data_matrix * self.MicrovoltsPerADAnalog[-1] / 1000

        if return_timestamps:
            timestamps = np.tile(np.arange(window_samples) * self.sample_ms, (n_windows, 1))
            start_times = np.tile(start_time_ms.T, window_samples)
            timestamps = timestamps + start_times
            return data_matrix, timestamps
        else:
            return data_matrix
