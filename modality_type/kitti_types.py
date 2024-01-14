from .ModBase import register, TYPEDICT
from .tartanair_types import RGBModBase, FlowModBase, DepthModBase, MotionModBase
from os.path import join

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

@register(TYPEDICT)
class kitti_rcam(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "image_right"

@register(TYPEDICT)
class kitti_ldisp(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "disp"

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
