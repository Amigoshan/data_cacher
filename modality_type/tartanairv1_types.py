from .ModBase import register, TYPEDICT
from .tartanair_types import RGBModBase, DepthModBase, SegModBase, IMUBase, FlowModBase, MotionModBase, PoseModBase
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

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        framenum = int(framestr)
        framestr2 = str(framenum + 2).zfill(6)
        return [join(self.folder_name, framestr + '_' + framestr2 + '_' + self.file_suffix + '.png')]

@register(TYPEDICT)
class flow4_left(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "flow4"
        self.file_suffix = "flow"
        self.drop_last = 4 # the flow is one frame shorter than other modalities

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        framenum = int(framestr)
        framestr2 = str(framenum + 4).zfill(6)
        return [join(self.folder_name, framestr + '_' + framestr2 + '_' + self.file_suffix + '.png')]

@register(TYPEDICT)
class pose_left(PoseModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return ['pose_left.txt']

@register(TYPEDICT)
class motion_left(MotionModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1

    def get_filename(self):
        return ['motion_left.npy']

@register(TYPEDICT)
class motion2_left(MotionModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2

    def get_filename(self):
        return ['motion_left2.npy']

@register(TYPEDICT)
class motion3_left(MotionModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 3

    def get_filename(self):
        return ['motion_left3.npy']

@register(TYPEDICT)
class motion4_left(MotionModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4

    def get_filename(self):
        return ['motion_left4.npy']

@register(TYPEDICT)
class imu_acc_v1(IMUBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return [join(self.folder_name, 'accel_left.npy')]

@register(TYPEDICT)
class imu_gyro_v1(IMUBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return [join(self.folder_name, 'gyro_left.npy')]

