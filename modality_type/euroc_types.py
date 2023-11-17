import cv2
import numpy as np
from .ModBase import SimpleModBase, FrameModBase, register, TYPEDICT
from os.path import join

class RGBModBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (3, self.h, self.w)
        self.data_type = np.uint8

    def load_frame(self, filename):
        # read image
        img = cv2.imread(filename)
        assert img is not None, "Error loading RGB {}".format(filename)
        return img

    def resize_data(self, img):
        # resize image
        (h, w, _) = img.shape
        if h != self.h or w != self.w:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
        return img

    def transpose(self, img):
        return img.transpose(2,0,1)

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '.png')

class DispModBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (self.h, self.w)
        self.data_type = np.float32

    def load_frame(self, filename):
        # read image
        disp_rgba = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert disp_rgba is not None, "Error loading disparity {}".format(filename)
        disp = disp_rgba.view("<f4")
        disp = np.squeeze(disp, axis=-1)
        return disp


    def resize_data(self, disparity):
        # resize image
        (h, w) = disparity.shape
        if h != self.h or w != self.w:
            disparity = cv2.resize(disparity, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
            scale_w = float(self.w) / w
            disparity[:,:] = disparity[:,:] * scale_w
        return disparity

    def transpose(self, disparity):
        return disparity

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return join(self.folder_name, framestr + '.png')

class FlowModBase(FrameModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        (self.h, self.w) = datashape # (h, w)
        self.data_shape = (2, self.h, self.w)
        self.data_type = np.float32
        self.drop_last = 1

    def load_frame(self, filename):
        # if filename is None: 
        #     return np.zeros((10,10,2), dtype=np.float32), np.zeros((10,10), dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow16 = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert flow16 is not None, "Error loading flow {}".format(filename)
        flow32 = flow16[:,:,:2].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0
        # flow32[:,:,2] = flow32[:,:,2] / 65535.
        return flow32

    def resize_data(self, flow):
        # resize image
        (h, w, _) = flow.shape
        if h != self.h or w != self.w:
            scale_w, scale_h = float(self.w) / w, float(self.h) / h
            flow = cv2.resize(flow, (self.w, self.h), interpolation=cv2.INTER_LINEAR )
            flow[:,:,0] = flow[:,:,0] * scale_w
            flow[:,:,1] = flow[:,:,1] * scale_h
        return flow

    def transpose(self, flow):
        return flow.transpose(2,0,1)

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        framenum = int(framestr)
        framestr2 = str(framenum + 1).zfill(6)
        return join(self.folder_name, framestr + '_' + framestr2 + '_flow.png')

class MotionModBase(SimpleModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.data_shape = (6,)

    def crop_trajectory(self, data, framestrlist):
        startind = int(framestrlist[0])
        endind = int(framestrlist[-1]) + 1 # motion len = N -1 , where N is the number of images
        datalen = data.shape[0]
        assert startind < datalen and endind <= datalen, "Error in loading motion, startind {}, endind {}, datalen {}".format(startind, endind, datalen)
        return data[startind: endind]

    def data_padding(self):
        return np.zeros((self.drop_last, 6), dtype=np.float32)

@register(TYPEDICT)
class euroc_lmotion(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return 'motion.txt'

@register(TYPEDICT)
class euroc_lmotion2(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return 'motion2.npy'

@register(TYPEDICT)
class euroc_lmotion3(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 3 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return 'motion3.npy'

@register(TYPEDICT)
class euroc_lmotion4(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return 'motion4.npy'


@register(TYPEDICT)
class euroc_lcam(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "cam0/data2"

@register(TYPEDICT)
class euroc_rcam(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "cam1/data2"

@register(TYPEDICT)
class euroc_ldisp(DispModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "cam0/disp_hsm"

@register(TYPEDICT)
class euroc_lflow(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "cam0/flow"

