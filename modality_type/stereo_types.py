import cv2
from .ModBase import FrameModBase, register, TYPEDICT

from os.path import join
import numpy as np

import re
 
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('ascii'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

class RGBModBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (3, self.h, self.w)
        self.data_type = np.uint8

    def load_frame(self, filename):
        # read image
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert img is not None, "Error loading RGB {}".format(filename)
        return img

    def transpose(self, img):
        return img.transpose(2,0,1)

    def resize_data(self, img):
        # resize image
        (h, w, _) = img.shape
        if h != self.h or w != self.w:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
        return img

# class DepthModBase(FrameModBase):
#     def __init__(self, datashape):
#         super().__init__(datashape)
#         self.data_shape = (self.h, self.w)
#         self.data_type = np.float32

#     def load_frame(self, filename):
#         dispImg, _ = readPFM(filename)
#         assert dispImg is not None, "Error loading depth {}".format(filename)
#         return dispImg

#     def transpose(self, disp):
#         return disp

#     def resize_data(self, depth):
#         # resize image
#         (h, w) = depth.shape
#         if h != self.h or w != self.w:
#             depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
#         return depth

@register(TYPEDICT)
class sceneflow_left(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "left"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '.png')

@register(TYPEDICT)
class sceneflow_right(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "right"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '.png')

@register(TYPEDICT)
class sceneflow_disp(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (self.h, self.w)
        self.data_type = np.float32
        self.folder_name = "left"

    def load_frame(self, filename):
        filename = filename.replace('frames_cleanpass', 'disparity').replace('frames_finalpass', 'disparity') # hard code
        dispImg, _ = readPFM(filename)
        assert dispImg is not None, "Error loading depth {}".format(filename)
        return dispImg

    def resize_data(self, depth):
        # resize image
        (h, w) = depth.shape
        if h != self.h or w != self.w:
            depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST )
        return depth

    def transpose(self, disp):
        return disp

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '.pfm') 

@register(TYPEDICT)
class kitti_left(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "colored_0"
        self.file_suffix = "10"
        self.drop_last = 1 # the flow is one frame shorter than other modalities
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.png')

@register(TYPEDICT)
class kitti_right(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "colored_1"
        self.file_suffix = "10"
        self.drop_last = 1 # the flow is one frame shorter than other modalities
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_' + self.file_suffix + '.png')

@register(TYPEDICT)
class kitti_disp(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (self.h, self.w)
        self.folder_name = "disp_occ"
        self.file_suffix = "10"
        self.drop_last = 1 # the flow is one frame shorter than other modalities


    def load_frame(self, filename):
        dispImg = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert dispImg is not None, "Error loading depth {}".format(filename)
        return dispImg

    def resize_data(self, depth):
        # resize image
        (h, w) = depth.shape
        if h != self.h or w != self.w:
            depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST )
        return depth

    def transpose(self, disp):
        return disp

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '_10.png')    

    # def framestr2filename(self, framestr):
    #     '''
    #     This is very dataset specific
    #     Basically it handles how each dataset naming the frames and organizing the data
    #     '''
    #     dispImg = cv2.imread(framestr)
    #     # print(dispImg.shape)
    #     imgh, imgw, _ = dispImg.shape
    #     imgx = (imgh-370)//2
    #     imgy = (imgw - 1224)//2
    #     dispImg = dispImg[imgx:imgx+370, imgy:imgy+1224, 0].astype(np.float32)
    #     return dispImg

