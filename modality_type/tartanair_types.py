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
        return data[startind: endind]

    def data_padding(self):
        '''
        In TartanAir, the lengh of IMU seq is (N-1)*10
        We would like the data be aligned, which means the nominal lengh should be N*10
        That's why we pad the data with 10 frames
        '''
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

# === lcam_front ===
@register(TYPEDICT)
class rgb_lcam_front(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_front"
        self.file_suffix = "lcam_front"
    
@register(TYPEDICT)
class depth_lcam_front(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_front"
        self.file_suffix = "lcam_front_depth"
    
@register(TYPEDICT)
class seg_lcam_front(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_front"
        self.file_suffix = "lcam_front_seg"

# === rcam_front ===
@register(TYPEDICT)
class rgb_rcam_front(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_rcam_front"
        self.file_suffix = "rcam_front"
    
@register(TYPEDICT)
class depth_rcam_front(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_rcam_front"
        self.file_suffix = "rcam_front_depth"
    
@register(TYPEDICT)
class seg_rcam_front(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_rcam_front"
        self.file_suffix = "rcam_front_seg" 


# === lcam_back ===
@register(TYPEDICT)
class rgb_lcam_back(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_back"
        self.file_suffix = "lcam_back"
    
@register(TYPEDICT)
class depth_lcam_back(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_back"
        self.file_suffix = "lcam_back_depth"
    
@register(TYPEDICT)
class seg_lcam_back(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_back"
        self.file_suffix = "lcam_back_seg"

# === rcam_back ===
@register(TYPEDICT)
class rgb_rcam_back(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_rcam_back"
        self.file_suffix = "rcam_back"
    
@register(TYPEDICT)
class depth_rcam_back(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_rcam_back"
        self.file_suffix = "rcam_back_depth"
    
@register(TYPEDICT)
class seg_rcam_back(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_rcam_back"
        self.file_suffix = "rcam_back_seg" 

# === lcam_left ===
@register(TYPEDICT)
class rgb_lcam_left(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_left"
        self.file_suffix = "lcam_left"
    
@register(TYPEDICT)
class depth_lcam_left(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_left"
        self.file_suffix = "lcam_left_depth"
    
@register(TYPEDICT)
class seg_lcam_left(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_left"
        self.file_suffix = "lcam_left_seg"

# === rcam_left ===
@register(TYPEDICT)
class rgb_rcam_left(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_rcam_left"
        self.file_suffix = "rcam_left"
    
@register(TYPEDICT)
class depth_rcam_left(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_rcam_left"
        self.file_suffix = "rcam_left_depth"
    
@register(TYPEDICT)
class seg_rcam_left(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_rcam_left"
        self.file_suffix = "rcam_left_seg" 

# === lcam_right ===
@register(TYPEDICT)
class rgb_lcam_right(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_right"
        self.file_suffix = "lcam_right"
    
@register(TYPEDICT)
class depth_lcam_right(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_right"
        self.file_suffix = "lcam_right_depth"
    
@register(TYPEDICT)
class seg_lcam_right(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_right"
        self.file_suffix = "lcam_right_seg"

# === rcam_right ===
@register(TYPEDICT)
class rgb_rcam_right(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_rcam_right"
        self.file_suffix = "rcam_right"
    
@register(TYPEDICT)
class depth_rcam_right(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_rcam_right"
        self.file_suffix = "rcam_right_depth"
    
@register(TYPEDICT)
class seg_rcam_right(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_rcam_right"
        self.file_suffix = "rcam_right_seg" 

# === lcam_top ===
@register(TYPEDICT)
class rgb_lcam_top(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_top"
        self.file_suffix = "lcam_top"
    
@register(TYPEDICT)
class depth_lcam_top(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_top"
        self.file_suffix = "lcam_top_depth"
    
@register(TYPEDICT)
class seg_lcam_top(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_top"
        self.file_suffix = "lcam_top_seg"

# === rcam_top ===
@register(TYPEDICT)
class rgb_rcam_top(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_rcam_top"
        self.file_suffix = "rcam_top"
    
@register(TYPEDICT)
class depth_rcam_top(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_rcam_top"
        self.file_suffix = "rcam_top_depth"
    
@register(TYPEDICT)
class seg_rcam_top(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_rcam_top"
        self.file_suffix = "rcam_top_seg" 

# === lcam_bottom ===
@register(TYPEDICT)
class rgb_lcam_bottom(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_bottom"
        self.file_suffix = "lcam_bottom"
    
@register(TYPEDICT)
class depth_lcam_bottom(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_bottom"
        self.file_suffix = "lcam_bottom_depth"
    
@register(TYPEDICT)
class seg_lcam_bottom(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_bottom"
        self.file_suffix = "lcam_bottom_seg"

# === rcam_bottom ===
@register(TYPEDICT)
class rgb_rcam_bottom(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_rcam_bottom"
        self.file_suffix = "rcam_bottom"
    
@register(TYPEDICT)
class depth_rcam_bottom(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_rcam_bottom"
        self.file_suffix = "rcam_bottom_depth"
    
@register(TYPEDICT)
class seg_rcam_bottom(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_rcam_bottom"
        self.file_suffix = "rcam_bottom_seg" 

# ==== FLOW ====
@register(TYPEDICT)
class flow_lcam_front(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "flow_lcam_front"
        self.file_suffix = "flow"
        self.drop_last = 1 # the flow is one frame shorter than other modalities
    
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
    The pose of the left front camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading pose, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]

    def get_filename(self):
        return 'pose_lcam_front.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_lcam_right(SimpleModBase):
    '''
    The pose of the left right-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)


    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading pose, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]
        
    def get_filename(self):
        return 'pose_lcam_right.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_lcam_back(SimpleModBase):
    '''
    The pose of the left back-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)


    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading pose, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]
        
    def get_filename(self):
        return 'pose_lcam_back.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_lcam_left(SimpleModBase):
    '''
    The pose of the left left-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)


    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading pose, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]
        
    def get_filename(self):
        return 'pose_lcam_left.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_lcam_top(SimpleModBase):
    '''
    The pose of the left top-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)


    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading pose, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]
        
    def get_filename(self):
        return 'pose_lcam_top.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_lcam_bottom(SimpleModBase):
    '''
    The pose of the left bottom-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)


    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading pose, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]
        
    def get_filename(self):
        return 'pose_lcam_bottom.txt'

    def data_padding(self):
        return None


@register(TYPEDICT)
class pose_rcam_front(SimpleModBase):
    '''
    The pose of the left front camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def get_filename(self):
        return 'pose_rcam_front.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_rcam_right(SimpleModBase):
    '''
    The pose of the left right-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def get_filename(self):
        return 'pose_rcam_right.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_rcam_back(SimpleModBase):
    '''
    The pose of the left back-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def get_filename(self):
        return 'pose_rcam_back.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_rcam_left(SimpleModBase):
    '''
    The pose of the left left-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def get_filename(self):
        return 'pose_rcam_left.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_rcam_top(SimpleModBase):
    '''
    The pose of the left top-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def get_filename(self):
        return 'pose_rcam_top.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class pose_rcam_bottom(SimpleModBase):
    '''
    The pose of the left bottom-facing camera.
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (7,)

    def get_filename(self):
        return 'pose_rcam_bottom.txt'

    def data_padding(self):
        return None

@register(TYPEDICT)
class motion_lcam_front(SimpleModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (6,)

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) # motion len = N -1 , where N is the number of images
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading motion, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]

    def get_filename(self):
        return 'motion_lcam_front.npy'

    def data_padding(self):
        return np.zeros((1,6), dtype=np.float32)

@register(TYPEDICT)
class lidar(LiDARBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = 'lidar'
        self.file_suffix = 'lcam_front_lidar'

    def framestr2filename(self, framestr):
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.ply')
