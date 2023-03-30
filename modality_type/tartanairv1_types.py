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

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0]) * self.freq_mult
        endind = int(framestrlist[-1]) * self.freq_mult # IMU len = (N-1)*10, where N is the number of images
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading IMU, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind:endind]

    def data_padding(self):
        return np.zeros((10,3), dtype=np.float32)

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

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.png')

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

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.png')

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

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        framenum = int(framestr)
        framestr2 = str(framenum + 1).zfill(6)
        return join(self.folder_name, framestr + '_' + framestr2 + '_' + self.file_suffix + '.png')

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

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.png')

@register(TYPEDICT)
class image_left(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_left"
        self.file_suffix = "left"

@register(TYPEDICT)
class image_left_blur(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_left_blur_0.5"
        self.file_suffix = "left"

@register(TYPEDICT)
class image_right(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_right"
        self.file_suffix = "right"

@register(TYPEDICT)
class depth_left(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_left"
        self.file_suffix = "left_depth"

@register(TYPEDICT)
class depth_right(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_right"
        self.file_suffix = "right_depth"
    
@register(TYPEDICT)
class seg_left(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_left"
        self.file_suffix = "left_seg"

@register(TYPEDICT)
class seg_right(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_right"
        self.file_suffix = "right_seg"

@register(TYPEDICT)
class flow_left(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "flow"
        self.file_suffix = "flow"
        self.drop_last = 1 # the flow is one frame shorter than other modalities

@register(TYPEDICT)
class flow2_left(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "flow2"
        self.file_suffix = "flow"
        self.drop_last = 2 # the flow is one frame shorter than other modalities

@register(TYPEDICT)
class flow4_left(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "flow4"
        self.file_suffix = "flow"
        self.drop_last = 4 # the flow is one frame shorter than other modalities

@register(TYPEDICT)
class pose_left(SimpleModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading pose, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind:endind]

    def get_filename(self):
        return 'pose_left.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class motion_left(SimpleModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (6,)
        self.drop_last = 1

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) # motion len = N -1 , where N is the number of images
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading motion, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind:endind]

    def get_filename(self):
        return 'motion_left.npy'

    def data_padding(self):
        return np.zeros((1,6), dtype=np.float32)

@register(TYPEDICT)
class imu_acc_v1(IMUBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return join(self.folder_name, 'accel_left.npy')

@register(TYPEDICT)
class imu_gyro_v1(IMUBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return join(self.folder_name, 'gyro_left.npy')

