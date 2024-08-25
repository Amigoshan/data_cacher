from os.path import join
import numpy as np
import time
'''
In the low level, each modality corresponds to a folder in the traj folder
This file defines the interfaces of the Modality: 
    - folder name
    - function that convert framestr to file
    - data type
    - data shape
New feature:
One folder sometimes contains more than one type of data that cannot be concatenate together, e.g. flow and flow_mask
We now return a list of numpy array, instead of one. 
'''
# TODO:


# please register new types here
TYPEDICT = dict()

def register(dst):
    def dec_register(cls):
        clsname = cls.__name__
        assert clsname not in dst, "Register error: type name {} duplicated".format(clsname)
        dst[clsname] = cls
        return cls
    return dec_register

def get_modality_type(typename):
    assert typename in TYPEDICT, "Unknow type {} for the cacher!".format(typename)
    return TYPEDICT[typename]

def repeat_function(func, func_params, repeat_times, error_msg = ""):
    try_count = 0
    res = None
    while res is None and try_count < repeat_times:
        res = func(**func_params)
        if res is None:
            time.sleep(0.2)
            try_count += 1
    assert res is not None, "Error in function {} after trying for {} times".format(error_msg, try_count)
    return res

class ModBase(object):
    def __init__(self, datashapelist):
        '''
        Note that datashepe is (h, w) 2D value for resizing the data
        self.data_shape is the shape with channel that used to initialize the buffer

        We allow the loader to return multiple numpy arrays (e.g. in the case of loading flow returns both flow and mask)
        If which case the datashape will be a list of (h, w)s

        The data_types is a list of data type 
        The data_shapes is a list of data shape
        '''
        self.name = self.__class__.__name__ # the key name in a dictionary
        self.data_types = None # needs to be filled in derived classes
        self.data_info = {} # store additional information 

        assert isinstance(datashapelist, list), "Type Error: datashape {} should be a list".format(datashapelist)
        self.data_shapes = datashapelist.copy() # needs to be filled in derived classes

        # handle the data with different frequency
        self.freq_mult = 1 # used when this modality has higher frequency, now only integer is supported, e.g. for IMU freq_mult=10
        self.drop_last = 0 # how many frames are dropped at the end of each trajectory, e.g. for optical flow, 1 frame is dropped
    
    def load_data(self, trajdir, framestr, ind_env):
        raise NotImplementedError


class FrameModBase(ModBase):
    '''
    This defines modality that is organized in frames
    such as image, depth, flow, LiDAR
    '''
    def load_data(self, trajdir, framestr, ind_env):
        '''
        ind_env: the index of the frame in the opposite direction of the frame 
                 this is used to handle some modalities miss a few frames at last
        new: the load_data function will be return a list of numpy arrays, in most cases, the lengh of the list will be one
        '''
        filenamelist = self.framestr2filename(framestr)
        if ind_env > self.drop_last:
            datalist = self.load_frame(trajdir, filenamelist)
            datalist = self.resize_data(datalist)
            datalist = self.transpose(datalist)
        else: # the frame does not exist, create a dummy frame
            datalist = []
            for datashape, datatype in zip(self.data_shapes, self.data_types):
                data = np.zeros(datashape, dtype=datatype)
                datalist.append(data)
        return datalist 

    def framestr2filename(self, framestr):
        raise NotImplementedError
    
    def resize_data(self, datalist):
        raise NotImplementedError

    def transpose(self, datalist):
        raise NotImplementedError

    def load_frame(self, trajdir, filename):
        raise NotImplementedError

class SimpleModBase(ModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder

    Note that we assume there is a one-to-one mapping between the framestr and the index of the frames in the modality
    The frame can be cropped in the datafile, but not really cropped in the data folder on the hard drive
    We will find the cooresponding frame by the framestr, instead of the frame index in the datafile

    For example, in the data folder, we have a trajectory of 100 frames. 
    It contains image folders with 100 images in each, and IMU numpy arrays that are with 100 in lengh
    We can crop the trajectory "virtually" in the datafile into two trajectories just by defining a datafile like this:
    --- datafile ---
    trajstr 30
    000000
    000001
    ...
    000029
    trajstr 40
    000050
    000051
    ...
    000089
    --- end of datafile ---
    If the second trajectory is loaded into the cache, we need to make sure the starting frame of that trajectory for IMU or motion is 50, instead of 0. 
    
    New: this allows that one class returns a list of numpy arrays, in most cases the length of the list will be one
    '''
    def __init__(self, datashapes):
        '''
        datashapes is not useful in this class because we do not resize the data
        the length of the datashapes indicates the number of numpy arrays it will return
        '''
        super().__init__(datashapes)
        self.data_types = [np.float32, ] * len(datashapes)

    def get_filename(self, trajdir):
        raise NotImplementedError

    def data_padding(self):
        '''
        Pad the entire trajectory to compensate the dropped frames
        '''
        raise NotImplementedError

    def crop_trajectory(self, data, framestrlist):
        '''
        The trajecotry can start from the middle of the trajectory
        '''
        raise NotImplementedError

    def load_data(self, trajdir, framestrlist):
        '''
        The startingframe indicates the starting frame of this modality
        It is used in the case that the trajectory is cropped into pieces and the current trajectory is not start from the first frame
        '''
        filenamelist = self.get_filename()
        datalist = []
        for k, filename in enumerate(filenamelist):
            if filename.endswith('.npy'):
                data = np.load(join(trajdir, filename)) # this assume the data is stored as np file, this can be changed in the future
            elif filename.endswith('.txt'):
                data = np.loadtxt(join(trajdir, filename))
            else:
                assert False, "File format is not supported {}".format(filename)
            # crop the trajectory based on the starting and ending frames
            padding = self.data_padding(k)
            if padding is not None:
                data = np.concatenate((data, padding), axis=0)
            data = self.crop_trajectory(data, framestrlist)
            assert len(data) == len(framestrlist) * self.freq_mult, "Error Loading {}, data len {}, framestr len {}".format(self.name, len(data), len(framestrlist))
            datalist.append(data)
        return datalist
