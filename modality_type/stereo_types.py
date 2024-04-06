import cv2
from .ModBase import FrameModBase, register, TYPEDICT
from .tartanair_types import RGBModBase, repeat_function
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

@register(TYPEDICT)
class sceneflow_left(RGBModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.folder_name = "left"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.png')]

    def load_frame(self, trajdir, filenamelist):
        # read image
        imglist = []
        for filename in filenamelist:
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.ppm'):
                img = repeat_function(cv2.imread, {'filename': join(trajdir,filename)}, 
                                        repeat_times=10, error_msg="loading RGB " + filename)
                if img.shape[2] == 4: # flying things returns 4 channels RGBA
                    img = img[:,:,:3]
            elif filename.endswith('.npy'):
                img = np.load(join(trajdir,filename))
            else:
                raise NotImplementedError
            imglist.append(img)
        return imglist

@register(TYPEDICT)
class sceneflow_right(sceneflow_left):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.folder_name = "right"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.png')]

@register(TYPEDICT)
class sceneflow_disp(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.data_types = [np.float32]
        self.folder_name = "left"

    def load_frame(self, trajdir, filenamelist):
        displist = []
        trajdir = trajdir.replace('frames_cleanpass', 'disparity').replace('frames_finalpass', 'disparity') # hard code
        for filename in filenamelist:
            filename = join(trajdir, filename)
            dispImg, _ = readPFM(filename)
            assert dispImg is not None, "Error loading depth {}".format(filename)
            displist.append(dispImg.copy())
        return displist

    def resize_data(self, displist):
        # resize image
        for k, disp in enumerate(displist):
            (h, w) = disp.shape
            target_h, target_w = self.data_shapes[k]
            if h != target_h or w != target_w:
                displist[k] = cv2.resize(disp, (target_w, target_h), interpolation=cv2.INTER_NEAREST )
                scale_w = float(target_w) / w
                disp = disp * scale_w
        return displist

    def transpose(self, displist):
        return displist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.pfm') ]

@register(TYPEDICT)
class kitti_left(RGBModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.folder_name = "colored_0"
        self.file_suffix = "10"
    
    def load_frame(self, trajdir, filenamelist):
        imglist = []
        for filename in filenamelist:
            img = cv2.imread(join(trajdir, filename), cv2.IMREAD_UNCHANGED)
            assert img is not None, "Error loading image {}".format(filename)
            # the kitti stereo image comes in different sizes
            # clip the data, because the cacher buffer cannot deal with difference sizes
            # imgh, imgw, _ = img.shape
            # imgx = (imgh-370)//2
            # imgy = (imgw - 1224)//2
            # img = img[imgx:imgx+370, imgy:imgy+1224,:]
            imglist.append(img)
        return imglist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '_' + self.file_suffix + '.png')]

@register(TYPEDICT)
class kitti_right(kitti_left):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.folder_name = "colored_1"
        self.file_suffix = "10"

@register(TYPEDICT)
class kitti_disp(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.folder_name = "disp_occ"
        self.file_suffix = "10"
        self.data_types = [np.float32]


    def load_frame(self, trajdir, filenamelist):
        displist = []
        for filename in filenamelist:
            dispImg = cv2.imread(join(trajdir, filename))
            assert dispImg is not None, "Error loading depth {}".format(filename)
            # imgh, imgw, _ = dispImg.shape
            # imgx = (imgh-370)//2
            # imgy = (imgw - 1224)//2
            # dispImg = dispImg[imgx:imgx+370, imgy:imgy+1224, 0].astype(np.float32)
            displist.append(dispImg[:,:,0].astype(np.float32))
        return displist

    def resize_data(self, displist):
        # resize image
        for k, disp in enumerate(displist):
            (h, w) = disp.shape
            target_h, target_w = self.data_shapes[k]
            if h != target_h or w != target_w:
                displist[k] = cv2.resize(disp, (target_w, target_h), interpolation=cv2.INTER_NEAREST )
                scale_w = float(target_w) / w
                disp = disp * scale_w
        return displist

    def transpose(self, displist):
        return displist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '_10.png')]

if __name__=="__main__":
    from utils import visdepth
    import cv2
    for frameid in range(1,100):
        trajfolder = '/peru/tartanvo_data/kitti/stereo/2012'
        # import ipdb;ipdb.set_trace()
        # image left
        datatype = kitti_left([(200, 400)])
        datalist0 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        # image right
        datatype = kitti_right([(200, 400)])
        datalist1 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        
        datatype = kitti_disp([(200, 400)])
        datalist2 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        depthvis = visdepth(datalist2[0])

        disp = cv2.hconcat((datalist0[0].transpose(1,2,0), datalist1[0].transpose(1,2,0), depthvis))
        cv2.imshow('img', disp)
        cv2.waitKey(0)

    # for frameid in range(6,16):
    #     trajfolder = '/home/amigo/tmp/data/sceneflow/frames_cleanpass/TEST/A/0000'
    #     # import ipdb;ipdb.set_trace()
    #     datatype = sceneflow_left([(200, 400)])
    #     datalist0 = datatype.load_data(trajfolder, str(frameid).zfill(4), 100)
    #     # image right
    #     datatype = sceneflow_right([(200, 400)])
    #     datalist1 = datatype.load_data(trajfolder, str(frameid).zfill(4), 100)
        
    #     datatype = sceneflow_disp([(200, 400)])
    #     datalist2 = datatype.load_data(trajfolder, str(frameid).zfill(4), 100)
    #     depthvis = visdepth(datalist2[0])

    #     disp = cv2.hconcat((datalist0[0].transpose(1,2,0), datalist1[0].transpose(1,2,0), depthvis))
    #     cv2.imshow('img', disp)
    #     cv2.waitKey(0)

    import ipdb;ipdb.set_trace()