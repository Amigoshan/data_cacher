import cv2
from .ModBase import SimpleModBase, FrameModBase, register, TYPEDICT, repeat_function
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
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.data_shapes = [(3,)]

        self.freq_mult = 10
        self.drop_last = 10

        self.folder_name = 'imu'

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0]) * self.freq_mult
        endind = (int(framestrlist[-1]) + 1) * self.freq_mult # IMU len = (N-1)*10, where N is the number of images
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading IMU, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]

    def data_padding(self, k):
        '''
        In TartanAir, the lengh of IMU seq is (N-1)*10
        We would like the data be aligned, which means the nominal lengh should be N*10
        That's why we pad the data with 10 frames
        '''
        return np.zeros((self.freq_mult, self.data_shapes[k][0]), dtype=np.float32)

class LiDARBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist) # point dimention, e.g. 3 for tartanvo, 6 if rgb is included
        self.data_shapes = [(57600, 3)] # 57600 is the maximun points for Velodyn 16 where there are 16x3600 points
        self.data_shape = self.data_shapes[0]
        self.data_types = [np.float32]

    def load_frame(self, trajdir, filenamelist):
        lidarlist = []
        for filename in filenamelist:
            if filename.endswith('npy'):
                data = np.load(join(trajdir,filename))
            elif filename.endswith('ply'):
                data = read_ply(join(trajdir,filename))
            else:
                assert False, "Unknow file type for LiDAR {}".format(filename)
            assert data.shape[0] <= self.data_shape[0], "The number of LiDAR points {} exeeds the maximum size {}".format(data.shape[0], self.data_shape[0])
            lidarlist.append(data)
        return lidarlist

    def transpose(self, lidarlist):
        return lidarlist

    def resize_data(self, lidarlist):
        return lidarlist

class EventsBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist) # point dimention, e.g. 3 for tartanvo, 6 if rgb is included
        lenlist = len(datashapelist)
        self.data_types = []
        for k in range(lenlist):
            self.data_types.append(np.float32)

        self.folder_name = "" # to be filled in the derived class
        self.file_suffix = "" # to be filled in the derived class

    def load_frame(self, trajdir, filenamelist):
        eventtensorlist = []
        for filename in filenamelist:
            if filename.endswith('.npz'):
                eventtensor = np.load(join(trajdir,filename))['event_tensor']
            else:
                raise NotImplementedError
            eventtensorlist.append(eventtensor)
        return eventtensorlist
    
    def transpose(self, events):
        return events

    def resize_data(self, events):
        return events

class RGBModBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        listlen = len(datashapelist) # this is usually one
        self.data_types = []
        for k in range(listlen):
            self.data_shapes[k] = (3,) + tuple(self.data_shapes[k] )
            self.data_types.append(np.uint8)

        self.folder_name = "" # to be filled in the derived class
        self.file_suffix = "" # to be filled in the derived class

    def load_frame(self, trajdir, filenamelist):
        # read image
        imglist = []
        for filename in filenamelist:
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.ppm'):
                img = repeat_function(cv2.imread, {'filename': join(trajdir,filename), 'flags': cv2.IMREAD_UNCHANGED}, 
                                        repeat_times=10, error_msg="loading RGB " + filename)
                if img.shape[2] == 4: # flying things returns 4 channels RGBA
                    img = img[:,:,:3]
            elif filename.endswith('.npy'):
                img = np.load(join(trajdir,filename))
            else:
                raise NotImplementedError
            imglist.append(img)
        return imglist

    def resize_data(self, imglist):
        # resize image
        for k, img in enumerate(imglist):
            h, w = img.shape[0], img.shape[1]
            target_h, target_w = self.data_shapes[k][1], self.data_shapes[k][2]
            if h != target_h or w != target_w:
                imglist[k] = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR )
        return imglist

    def transpose(self, imglist):
        reslist = []
        for img in imglist:
            reslist.append(img.transpose(2,0,1))
        return reslist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, framestr + file_suffix + '.png')]

class DepthModBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        listlen = len(datashapelist) # this is usually one
        self.data_types = []
        for k in range(listlen):
            self.data_types.append(np.float32)

        self.folder_name = "" # to be filled in the derived class
        self.file_suffix = "" # to be filled in the derived class

    def load_frame(self, trajdir, filenamelist):
        depthlist = []
        for filename in filenamelist:
            depth_rgba = repeat_function(cv2.imread, {'filename': join(trajdir,filename), 'flags': cv2.IMREAD_UNCHANGED}, 
                                    repeat_times=10, error_msg="loading depth " + filename)
            depth = depth_rgba.view("<f4")
            depth = np.squeeze(depth, axis=-1)
            depthlist.append(depth)
        return depthlist

    def resize_data(self, depthlist):
        # resize depth
        for k, depth in enumerate(depthlist):
            (h, w) = depth.shape
            target_h, target_w = self.data_shapes[k]
            if h != target_h or w != target_w:
                depthlist[k] = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR )
        return depthlist

    def transpose(self, depthlist):
        return depthlist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, framestr + file_suffix + '.png')]

class FlowModBase(FrameModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        # we assume that the flow might return flow or (flow, mask)
        # we also assume that the flow will always be returned, the mask is optional
        self.listlen = len(datashapes) # this is usually one
        self.data_shapes[0] = (2,) + tuple(self.data_shapes[0]) # add one dim to the 
        if self.listlen == 1:
            self.data_types = [np.float32] # for flow and mask
        else:
            self.data_types = [np.float32, np.uint8] # for flow and mask

        self.folder_name = "" # to be filled in the derived class
        self.file_suffix = "" # to be filled in the derived class

    def load_frame(self, trajdir, filenamelist):
        # if filename is None: 
        #     return np.zeros((10,10,2), dtype=np.float32), np.zeros((10,10), dtype=np.uint8) # return an arbitrary shape because it will be resized later
        if filenamelist[0].endswith('.png'):
            flow16 = repeat_function(cv2.imread, {'filename': join(trajdir, filenamelist[0]), 'flags': cv2.IMREAD_UNCHANGED}, 
                                    repeat_times=10, error_msg="loading depth " + filenamelist[0])
            flow32 = flow16[:,:,:2].astype(np.float32)
            flow32 = (flow32 - 32768) / 64.0
        elif filenamelist[0].endswith('.npy'):
            flow32 = np.load(join(trajdir, filenamelist[0]))
        else:
            raise NotImplementedError

        if self.listlen == 1:
            return [flow32]

        mask8 = flow16[:,:,2].astype(np.uint8)
        return [flow32, mask8]

    def resize_data(self, flowmasklist):
        # resize image
        flow = flowmasklist[0]
        target_h, target_w = self.data_shapes[0][1], self.data_shapes[0][2]
        (h, w, _) = flow.shape
        if h != target_h or w != target_w:
            scale_w, scale_h = float(target_w) / w, float(target_h) / h
            flow = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR )
            flow[:,:,0] = flow[:,:,0] * scale_w
            flow[:,:,1] = flow[:,:,1] * scale_h
        if self.listlen == 1: 
            return [flow]
        
        mask = flowmasklist[1]
        target_h, target_w = self.data_shapes[1]
        (h, w) = mask.shape
        if h != target_h or w != target_w:
            scale_w, scale_h = float(target_w) / w, float(target_h) / h
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST )
        return [flow, mask]

    def transpose(self, flowlist):
        reslist = []
        for img in flowlist:
            if len(img.shape) == 3:
                reslist.append(img.transpose(2,0,1))
            else:
                reslist.append(img)
        return reslist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        framenum = int(framestr)
        framestr2 = str(framenum + 1).zfill(6)
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, framestr + '_' + framestr2 + file_suffix + '.png')]

class SegModBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        listlen = len(datashapelist)
        self.data_types = []
        for k in range(listlen):
            self.data_types.append(np.uint8)

        self.folder_name = "" # to be filled in the derived class
        self.file_suffix = "" # to be filled in the derived class

    def load_frame(self, trajdir, filenamelist):
        segglist = []
        for filename in filenamelist:
            segimg = repeat_function(cv2.imread, {'filename': join(trajdir,filename), 'flags': cv2.IMREAD_UNCHANGED}, 
                                    repeat_times=10, error_msg="loading depth " + filename)
            segglist.append(segimg)

        return segglist

    def resize_data(self, seglist):
        # resize image
        for k, seg in enumerate(seglist):
            (h, w) = seg.shape
            target_h, target_w = self.data_shapes[k]
            if h != target_h or w != target_w:
                seglist[k] = cv2.resize(seg, (target_w, target_h), interpolation=cv2.INTER_NEAREST )
        return seglist

    def transpose(self, seglist):
        return seglist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, framestr + file_suffix + '.png')]

class PoseModBase(SimpleModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shapes = [(7,)]

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading pose, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]

    def data_padding(self, k):
        return None

class MotionModBase(SimpleModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shapes = [(6,)]

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1 # motion len = N -1 , where N is the number of images
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading motion, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]

    def data_padding(self, k):
        return np.zeros((self.drop_last, self.data_shapes[k][0]), dtype=np.float32)

# === lcam_front ===
@register(TYPEDICT)
class image_lcam_front(RGBModBase):
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
class image_rcam_front(RGBModBase):
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
class image_lcam_back(RGBModBase):
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
class image_rcam_back(RGBModBase):
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
class image_lcam_left(RGBModBase):
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
class image_rcam_left(RGBModBase):
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
class image_lcam_right(RGBModBase):
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
class image_rcam_right(RGBModBase):
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
class image_lcam_top(RGBModBase):
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
class image_rcam_top(RGBModBase):
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
class image_lcam_bottom(RGBModBase):
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
class image_rcam_bottom(RGBModBase):
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

# ==== lcam equirect ====
@register(TYPEDICT)
class image_lcam_equirect(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_equirect"
        self.file_suffix = "lcam_equirect_image"

@register(TYPEDICT)
class depth_lcam_equirect(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_equirect"
        self.file_suffix = "lcam_equirect_depth"
    
@register(TYPEDICT)
class seg_lcam_equirect(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_equirect"
        self.file_suffix = "lcam_equirect_seg" 

# ==== rcam equirect ====
@register(TYPEDICT)
class image_rcam_equirect(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_rcam_equirect"
        self.file_suffix = "rcam_equirect_image"

@register(TYPEDICT)
class depth_rcam_equirect(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_rcam_equirect"
        self.file_suffix = "rcam_equirect_depth"

# ==== lcam fish ====
@register(TYPEDICT)
class image_lcam_fish(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_lcam_fish"
        self.file_suffix = "lcam_fish_image"

@register(TYPEDICT)
class depth_lcam_fish(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_lcam_fish"
        self.file_suffix = "lcam_fish_depth"
    
@register(TYPEDICT)
class seg_lcam_fish(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_lcam_fish"
        self.file_suffix = "lcam_fish_seg" 

# ==== rcam fish ====
@register(TYPEDICT)
class image_rcam_fish(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_rcam_fish"
        self.file_suffix = "rcam_fish_image"

@register(TYPEDICT)
class depth_rcam_fish(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_rcam_fish"
        self.file_suffix = "rcam_fish_depth"

@register(TYPEDICT)
class seg_rcam_fish(SegModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "seg_rcam_fish"
        self.file_suffix = "rcam_fish_seg" 

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
        return [join(self.folder_name, 'acc.npy')]

@register(TYPEDICT)
class imu_gyro(IMUBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return [join(self.folder_name, 'gyro.npy')]

@register(TYPEDICT)
class imu(IMUBase):
    '''
    This combines imu_acc and imu_gyro
    '''
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.data_shapes = [(6,)]

        self.freq_mult = 10
        self.drop_last = 10

        self.folder_name = 'imu'

        self.imu_acc_loader = imu_acc(datashapelist)
        self.imu_gyro_loader = imu_gyro(datashapelist)

    def load_data(self, trajdir, framestrlist):
        '''
        The startingframe indicates the starting frame of this modality
        It is used in the case that the trajectory is cropped into pieces and the current trajectory is not start from the first frame
        '''
        datalist_acc = self.imu_acc_loader.load_data(trajdir, framestrlist)
        datalist_gyro = self.imu_gyro_loader.load_data(trajdir, framestrlist)

        datalist = [np.concatenate((acc, gyro), axis=-1) for acc, gyro in zip(datalist_acc, datalist_gyro)]
        return datalist

@register(TYPEDICT)
class pose_lcam_front(PoseModBase):
    '''
    The pose of the left front camera.
    '''
    def get_filename(self):
        return ['pose_lcam_front.txt']

@register(TYPEDICT)
class pose_lcam_right(PoseModBase):
    '''
    The pose of the left right-facing camera.
    '''
    def get_filename(self):
        return ['pose_lcam_right.txt']

@register(TYPEDICT)
class pose_lcam_back(PoseModBase):
    '''
    The pose of the left back-facing camera.
    '''
    def get_filename(self):
        return ['pose_lcam_back.txt']

@register(TYPEDICT)
class pose_lcam_left(PoseModBase):
    '''
    The pose of the left left-facing camera.
    '''
    def get_filename(self):
        return ['pose_lcam_left.txt']

@register(TYPEDICT)
class pose_lcam_top(PoseModBase):
    '''
    The pose of the left top-facing camera.
    '''
    def get_filename(self):
        return ['pose_lcam_top.txt']

@register(TYPEDICT)
class pose_lcam_bottom(PoseModBase):
    '''
    The pose of the left bottom-facing camera.
    '''
    def get_filename(self):
        return ['pose_lcam_bottom.txt']

@register(TYPEDICT)
class pose_rcam_front(PoseModBase):
    '''
    The pose of the right front-facing camera.
    '''
    def get_filename(self):
        return ['pose_rcam_front.txt']

@register(TYPEDICT)
class pose_rcam_right(PoseModBase):
    '''
    The pose of the right right-facing camera.
    '''
    def get_filename(self):
        return ['pose_rcam_right.txt']

@register(TYPEDICT)
class pose_rcam_back(PoseModBase):
    '''
    The pose of the right back-facing camera.
    '''
    def get_filename(self):
        return ['pose_rcam_back.txt']

@register(TYPEDICT)
class pose_rcam_left(PoseModBase):
    '''
    The pose of the right left-facing camera.
    '''
    def get_filename(self):
        return ['pose_rcam_left.txt']

@register(TYPEDICT)
class pose_rcam_top(PoseModBase):
    '''
    The pose of the right top-facing camera.
    '''
    def get_filename(self):
        return ['pose_rcam_top.txt']

@register(TYPEDICT)
class pose_rcam_bottom(PoseModBase):
    '''
    The pose of the right bottom-facing camera.
    '''
    def get_filename(self):
        return ['pose_rcam_bottom.txt']


@register(TYPEDICT)
class motion_lcam_front(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_front.npy']

@register(TYPEDICT)
class motion2_lcam_front(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_front2.npy']

class motion4_lcam_front(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_front4.npy']

@register(TYPEDICT)
class motion_lcam_back(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_back.npy']

@register(TYPEDICT)
class motion2_lcam_back(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_back2.npy']

class motion4_lcam_back(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_back4.npy']

@register(TYPEDICT)
class motion_lcam_left(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_left.npy']

@register(TYPEDICT)
class motion2_lcam_left(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_left2.npy']

class motion4_lcam_left(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_left4.npy']

@register(TYPEDICT)
class motion_lcam_right(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_right.npy']

@register(TYPEDICT)
class motion2_lcam_right(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_right2.npy']

class motion4_lcam_right(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_right4.npy']

@register(TYPEDICT)
class motion_lcam_up(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_up.npy']

@register(TYPEDICT)
class motion2_lcam_up(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_up2.npy']

class motion4_lcam_up(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_up4.npy']

@register(TYPEDICT)
class motion_lcam_bottom(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_bottom.npy']

@register(TYPEDICT)
class motion2_lcam_bottom(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_bottom2.npy']

class motion4_lcam_bottom(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_lcam_bottom4.npy']

@register(TYPEDICT)
class motion_rcam_front(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_front.npy']

@register(TYPEDICT)
class motion2_rcam_front(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_front2.npy']

class motion4_rcam_front(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_front4.npy']

@register(TYPEDICT)
class motion_rcam_back(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_back.npy']

@register(TYPEDICT)
class motion2_rcam_back(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_back2.npy']

class motion4_rcam_back(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_back4.npy']

@register(TYPEDICT)
class motion_rcam_left(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_left.npy']

@register(TYPEDICT)
class motion2_rcam_left(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_left2.npy']

class motion4_rcam_left(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_left4.npy']

@register(TYPEDICT)
class motion_rcam_right(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_right.npy']

@register(TYPEDICT)
class motion2_rcam_right(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_right2.npy']

class motion4_rcam_right(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_right4.npy']

@register(TYPEDICT)
class motion_rcam_up(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_up.npy']

@register(TYPEDICT)
class motion2_rcam_up(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_up2.npy']

class motion4_rcam_up(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_up4.npy']

@register(TYPEDICT)
class motion_rcam_bottom(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_bottom.npy']

@register(TYPEDICT)
class motion2_rcam_bottom(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_bottom2.npy']

class motion4_rcam_bottom(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion_rcam_bottom4.npy']

@register(TYPEDICT)
class lidar(LiDARBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = 'lidar'
        self.file_suffix = 'lcam_front_lidar'

    def framestr2filename(self, framestr):
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, framestr + file_suffix + '.ply')]

@register(TYPEDICT)
class event_cam(EventsBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = 'events'
        self.sub_folder = 'event_tensors'
        self.file_suffix = 'event_tensor'
        self.drop_last = 1

    def framestr2filename(self, framestr):
        framenum = int(framestr)
        framestr2 = str(framenum + 1).zfill(6)
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, join(self.sub_folder, framestr + '_' + framestr2 + file_suffix + '.npz'))]
