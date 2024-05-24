from .ModBase import register, TYPEDICT
from .tartanair_types import DepthModBase, FlowModBase, MotionModBase
from .tartandrive_types import GreyModBase
from os.path import join
import cv2

@register(TYPEDICT)
class euroc_lmotion(MotionModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def get_filename(self):
        return ['motion.txt']

@register(TYPEDICT)
class euroc_lcam(GreyModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "cam0/data2"

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.png')]

@register(TYPEDICT)
class euroc_rcam(GreyModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "cam1/data2"

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.png')]

@register(TYPEDICT)
class euroc_ldisp(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "cam0/disp_hsm"

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
class euroc_lflow(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "cam0/flow"
        self.file_suffix = 'flow'
        self.drop_last = 1

if __name__=="__main__":
    trajfolder = '/bigdata/tartanvo_data/euroc/MH_01_easy_mav0_StereoRectified'
    from utils import visdepth, visflow
    import cv2
    for frameid in range(300):
        # import ipdb;ipdb.set_trace()
        # image left
        datatype = euroc_lcam([(480, 752)])
        datalist0 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        # image right
        datatype = euroc_rcam([(480, 752)])
        datalist1 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        # image color
        datatype = euroc_ldisp([(200, 400)])
        datalist2 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        print(len(datalist2), datalist2[0].shape)
        # rgb map
        datatype = euroc_lflow([(200, 400)])
        datalist3 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        print(len(datalist3), datalist3[0].shape)
        # height map
        datatype = euroc_lmotion([6])
        datalist4 = datatype.load_data(trajfolder, [str(frameid).zfill(6)])
        print(len(datalist4), datalist4[0].shape,  datalist4[0].dtype)

        disp = cv2.hconcat((datalist0[0], datalist1[0]))
        cv2.imshow('img', disp)
        visdisp = visdepth(datalist2[0])
        vis_flow = visflow(datalist3[0].transpose(1,2,0))
        disp2 = cv2.hconcat((visdisp, vis_flow))
        cv2.imshow('img2', disp2)        
        cv2.waitKey(0)

    import ipdb;ipdb.set_trace()
    