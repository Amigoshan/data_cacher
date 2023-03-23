import cv2
from .ModBase import SimpleModBase, FrameModBase, register, TYPEDICT
from os.path import join
import numpy as np
from .ply_io import read_ply

'''
In the low level, each modality corresponds to a folder in the traj folder
This file defines the interfaces of the Modality: 
    - folder name
    - function that convert framestr to file
    - data type
    - data shape
'''

class IMUBase(SimpleModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (3,)

        self.freq_mult = 10
        self.drop_last = 10

        self.folder_name = 'imu'

    def data_padding(self):
        return np.zeros((10,3), dtype=np.float32)

class LiDARBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape) # point dimention, e.g. 3 for tartanvo, 6 if rgb is included
        self.data_shape = (57600, 3) # 57600 is the maximun points for Velodyn 16 where there are 16x3600 points
        self.data_type = np.float32

    def load_frame(self, filename):
        if filename.endswith('npy'):
            data = np.load(filename)
        elif filename.endswith('ply'):
            data = read_ply(filename)
        else:
            assert False, "Unknow file type for LiDAR {}".format(filename)
        assert data.shape[0] <= self.data_shape[0], "The number of LiDAR points {} exeeds the maximum size {}".format(data.shape[0], self.data_shape[0])
        return data

    def resize_data(self, lidar):
        return lidar

class RGBModBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (self.h, self.w, 3)
        self.data_type = np.uint8

    def load_frame(self, filename):
        # read image
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert img is not None, "Error loading RGB {}".format(filename)
        return img

    def resize_data(self, img):
        # resize image
        (h, w, _) = img.shape
        if h != self.h or w != self.w:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
        return img

class DepthModBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (self.h, self.w)
        self.data_type = np.float32

    def load_frame(self, filename):
        depth_rgba = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert depth_rgba is not None, "Error loading depth {}".format(filename)
        depth = depth_rgba.view("<f4")
        depth = np.squeeze(depth, axis=-1)
        return depth

    def resize_data(self, depth):
        # resize image
        (h, w) = depth.shape
        if h != self.h or w != self.w:
            depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
        return depth

class FlowModBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (self.h, self.w, 2)
        self.data_type = np.float32

    def load_frame(self, filename):
        # if filename is None: 
        #     return np.zeros((10,10,2), dtype=np.float32), np.zeros((10,10), dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow16 = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert flow16 is not None, "Error loading flow {}".format(filename)
        flow32 = flow16[:,:,:2].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0
        # mask8 = flow16[:,:,2].astype(np.uint8)
        return flow32 #, mask8

    def resize_data(self, flow):
        # resize image
        (h, w, _) = flow.shape
        if h != self.h or w != self.w:
            flow = cv2.resize(flow, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
        return flow

class SegModBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (self.h, self.w)
        self.data_type = np.uint8

    def load_frame(self, filename):
        segimg = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert segimg is not None, "Error loading seg {}".format(filename)
        return segimg

    def resize_data(self, seg):
        # resize image
        (h, w) = seg.shape
        if h != self.h or w != self.w:
            seg = cv2.resize(seg, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
        return seg

@register(TYPEDICT)
class rgb_lcam_front(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_front"
        self.file_suffix = "lcam_front"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.png')

@register(TYPEDICT)
class depth_lcam_front(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_front"
        self.file_suffix = "lcam_front_depth"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.png')

@register(TYPEDICT)
class seg_lcam_front(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_front"
        self.file_suffix = "lcam_front_seg"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.png')

@register(TYPEDICT)
class flow_lcam_front(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "flow_lcam_front"
        self.file_suffix = "flow"
        self.drop_last = 1 # the flow is one frame shorter than other modalities
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        framenum = int(framestr)
        framestr2 = str(framenum + 1).zfill(6)
        return join(self.folder_name, framestr + '_' + framestr2 + '_' + self.file_suffix + '.png')

@register(TYPEDICT)
class imu_acc(IMUBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return join(self.folder_name, 'acc.npy')

@register(TYPEDICT)
class imu_gyro(IMUBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return join(self.folder_name, 'gyro.npy')

@register(TYPEDICT)
class pose_lcam_front(SimpleModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def get_filename(self):
        return 'pose_lcam_front.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class lidar(LiDARBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = 'lidar'
        self.file_suffix = 'lcam_front_lidar'

    def framestr2filename(self, framestr):
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.ply')
