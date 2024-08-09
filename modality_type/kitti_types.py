from .ModBase import register, TYPEDICT, repeat_function
from .tartanair_types import RGBModBase, FlowModBase, DepthModBase, MotionModBase
from os.path import join
import cv2
import numpy as np

@register(TYPEDICT)
class kitti_lmotion(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion.npy']

@register(TYPEDICT)
class kitti_lmotion2(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 2 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return 'motion2.npy'

@register(TYPEDICT)
class kitti_lmotion3(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 3 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return 'motion3.npy'

@register(TYPEDICT)
class kitti_lmotion4(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 4 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return 'motion4.npy'

@register(TYPEDICT)
class kitti_lcam(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_left"
        self.file_suffix = ''

@register(TYPEDICT)
class kitti_rcam(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_right"
        self.file_suffix = ''

@register(TYPEDICT)
class kitti_ldisp(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "disp"

    def resize_data(self, displist):
        # resize disparity
        for k, disp in enumerate(displist):
            (h, w) = disp.shape
            target_h, target_w = self.data_shapes[k]
            if h != target_h or w != target_w:
                displist[k] = cv2.resize(disp, (target_w, target_h), interpolation=cv2.INTER_LINEAR )
                displist[k] = displist[k] * (float(target_w)/w)
        return displist

@register(TYPEDICT)
class kitti_lflow(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "flow"
        self.file_suffix = 'flow'
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        framenum = int(framestr)
        framestr2 = str(framenum + 1).zfill(6)
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, framestr + '_' + framestr2 + file_suffix + '.npy')]

@register(TYPEDICT)
class kitti_gtdisp(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "disp_occ"
        self.file_suffix = '10'

    def load_frame(self, trajdir, filenamelist):
        disp = repeat_function(cv2.imread, {'filename': join(trajdir, filenamelist[0]), 'flags': cv2.IMREAD_UNCHANGED}, 
                                repeat_times=10, error_msg="loading depth " + filenamelist[0])
        disp = disp.astype(np.float32)/256.
        mask = disp>0
        return [disp]

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, framestr + file_suffix + '.png')]

@register(TYPEDICT)
class kitti_rgb_for_flow(RGBModBase):
    '''
    flow dataset does not come in sequence
    in colored_0 folder, _10 and _11 files are the two images 
    we define in the kitti_flow.txt 394 trajecoties with lenght of 2
    '''
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "colored_0"

@register(TYPEDICT)
class kitti_gtflow(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "flow_occ"
        # self.file_suffix = '10'
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def load_frame(self, trajdir, filenamelist):
        flow16 = repeat_function(cv2.imread, {'filename': join(trajdir, filenamelist[0]), 'flags': cv2.IMREAD_UNCHANGED}, 
                                repeat_times=10, error_msg="loading depth " + filenamelist[0])
        flow32 = flow16[:,:,1:].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0
        flow32 = flow32[:,:,::-1]

        if self.listlen == 1:
            return [flow32]

        mask8 = flow16[:,:,0].astype(np.uint8)
        return [flow32.copy(), mask8]

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        file_suffix = '_' + self.file_suffix if self.file_suffix != "" else ""
        return [join(self.folder_name, framestr + file_suffix + '.png')]

if __name__=="__main__":
    trajfolder = '/peru/tartanvo_data/kitti/stereo_flow_cropped/2015'
    from utils import visdepth, visflow
    import cv2
    for frameid in range(300):
        # import ipdb;ipdb.set_trace()
        # image left
        datatype = kitti_gtflow([(320, 1216)])
        datalist = datatype.load_data(trajfolder, str(frameid).zfill(6)+'_10', 100)
        print(len(datalist), datalist[0].shape)
        vis_flow = visflow(datalist[0].transpose(1,2,0))

        datatype = kitti_rgb_for_flow([(320, 1216)])
        datalist1 = datatype.load_data(trajfolder, str(frameid).zfill(6)+'_10', 100)

        datatype = kitti_rgb_for_flow([(320, 1216)])
        datalist2 = datatype.load_data(trajfolder, str(frameid).zfill(6)+'_11', 100)

        # datatype = kitti_gtdisp([(320, 1216)])
        # datalist3 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        # print(len(datalist3), datalist3[0].shape)
        # vis_disp = visdepth(datalist3[0])
        # import ipdb;ipdb.set_trace()
        cv2.imshow('img2', cv2.vconcat((vis_flow, datalist1[0].transpose(1,2,0), datalist2[0].transpose(1,2,0))))        
        cv2.waitKey(0)

    import ipdb;ipdb.set_trace()
