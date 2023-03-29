from os.path import join
import numpy as np

'''
In the low level, each modality corresponds to a folder in the traj folder
This file defines the interfaces of the Modality: 
    - folder name
    - function that convert framestr to file
    - data type
    - data shape
'''
# TODO:
#       Simple Mod
#       Frequency 
#       Handle freq_mult in dropping frames

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

class ModBase(object):
    def __init__(self, datashape):
        '''
        Note that datashepe is (h, w) 2D value for resizing the data
        self.data_shape is the shape with channel that used to initialize the buffer
        '''
        self.name = self.__class__.__name__ # the key name in a dictionary
        self.data_type = None # needs to be filled in derived classes
        self.data_shape = None # needs to be filled in derived classes
        self.data_info = {} # store additional information 

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
        '''
        filename = self.framestr2filename(framestr)
        if ind_env > self.drop_last:
            data = self.load_frame(join(trajdir, filename))
        else: # the frame does not exist, create a dummy frame
            data = np.zeros(self.data_shape, dtype=self.data_type)
        data = self.resize_data(data)
        return data 

    def framestr2filename(self, framestr):
        raise NotImplementedError
    
    def resize_data(self, ):
        raise NotImplementedError

    def load_frame(self, filename):
        raise NotImplementedError

class SimpleModBase(ModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder

    Note that we assume that there is one-to-one mapping between the framestr and the index of the frames in the modality
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
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_type = np.float32

    def get_filename(self, trajdir):
        raise NotImplementedError

    def data_padding(self):
        raise NotImplementedError

    def crop_trajectory(self, data, framestrlist):
        raise NotImplementedError

    def load_data(self, trajdir, framestrlist):
        '''
        The startingframe indicates the starting frame of this modality
        It is used in the case that the trajectory is cropped into pieces and the current trajectory is not start from the first frame
        '''
        filename = self.get_filename()
        if filename.endswith('.npy'):
            data = np.load(join(trajdir, filename)) # this assume the data is stored as np file, this can be changed in the future
        elif filename.endswith('.txt'):
            data = np.loadtxt(join(trajdir, filename))
        else:
            assert False, "File format is not supported {}".format(filename)
        # crop the trajectory based on the starting and ending frames
        data = self.crop_trajectory(data, framestrlist)
        padding = self.data_padding()
        if padding is not None:
            data = np.concatenate((data, padding), axis=0)
        return data

