from .ModBase import register, TYPEDICT, repeat_function
from .tartanair_types import RGBModBase, FlowModBase
from os.path import join
import numpy as np
from .stereo_types import readPFM

def read_flo_bytes(bio):
    """
    bio is an io.BytesIO object.
    """
    try:
      buffer = bio.getvalue() # python2
    except:
      buffer = bio.getbuffer() # python3

    magic = np.frombuffer( buffer, dtype=np.float32, count=1 )

    if ( 202021.25 != magic ):
        print('Matic number incorrect. Expect 202021.25, get {}. Invalid .flo file.'.format( \
            magic ))

        return None
    
    W = np.frombuffer( buffer, dtype=np.int32, count=1, offset=4 )
    H = np.frombuffer( buffer, dtype=np.int32, count=1, offset=8 )

    W = int(W)
    H = int(H)

    data = np.frombuffer( buffer, dtype=np.float32, \
        count=2*W*H, offset=12 )

    return np.resize( data, ( H, W, 2 ) )

def read_flo(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

@register(TYPEDICT)
class chair_img(RGBModBase):
    # (384, 512, 3)
    def framestr2filename(self, framestr):
        return [framestr + '.ppm']

@register(TYPEDICT)
class chair_flow(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def framestr2filename(self, framestr):
        return [framestr.split('_')[0] + '_flow.flo']

    def load_frame(self, trajdir, filenamelist):
        # if filename is None: 
        #     return np.zeros((10,10,2), dtype=np.float32), np.zeros((10,10), dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow = repeat_function(read_flo, {'fn': join(trajdir, filenamelist[0])}, 
                                repeat_times=10, error_msg="loading flow " + filenamelist[0])

        return [flow]


@register(TYPEDICT)
class sintel_img(RGBModBase):
    # (436, 1024, 3)
    def framestr2filename(self, framestr):
        return ['frame_' + framestr + '.png']

@register(TYPEDICT)
class sintel_flow(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def load_data(self, trajdir, framestr, ind_env):
        '''
        ind_env: the index of the frame in the opposite direction of the frame 
                 this is used to handle some modalities miss a few frames at last
        new: the load_data function will be return a list of numpy arrays, in most cases, the lengh of the list will be one
        '''
        filenamelist = self.framestr2filename(framestr)
        trajdir = trajdir.replace('final','flow').replace('clean','flow')
        if ind_env > self.drop_last:
            datalist = self.load_frame(trajdir, filenamelist)
            datalist = self.resize_data(datalist)
            datalist = self.transpose(datalist)
        else: # the frame does not exist, create a dummy frame
            datalist = []
            for datashape, datatype in zip(self.data_shapes, self.data_types):
                data = np.zeros(datashape, dtype=datatype)
                datalist.append(data)
        return datalist 

    def framestr2filename(self, framestr):
        return ['frame_' + framestr + '.flo']

    def load_frame(self, trajdir, filenamelist):
        flow = repeat_function(read_flo, {'fn': join(trajdir, filenamelist[0])}, 
                                repeat_times=10, error_msg="loading flow " + filenamelist[0])

        return [flow]


@register(TYPEDICT)
class flying_img(RGBModBase):
    # (540, 960, 3)
    def framestr2filename(self, framestr):
        return [framestr + '.png']

@register(TYPEDICT)
class flying_flow(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.drop_last = 1 # this is used to let the loader know how much frames are short

    def load_data(self, trajdir, framestr, ind_env):
        '''
        ind_env: the index of the frame in the opposite direction of the frame 
                 this is used to handle some modalities miss a few frames at last
        new: the load_data function will be return a list of numpy arrays, in most cases, the lengh of the list will be one
        '''
        filenamelist = self.framestr2filename(framestr, trajdir)
        trajdir = trajdir.replace('frames_finalpass', 'optical_flow').replace('frames_cleanpass', 'optical_flow')
        if 'left' in trajdir:
            trajdir = trajdir.replace('left', 'into_future/left')
        elif 'right' in trajdir:
            trajdir = trajdir.replace('right', 'into_future/right')
            
        if ind_env > self.drop_last:
            datalist = self.load_frame(trajdir, filenamelist)
            datalist = self.resize_data(datalist)
            datalist = self.transpose(datalist)
        else: # the frame does not exist, create a dummy frame
            datalist = []
            for datashape, datatype in zip(self.data_shapes, self.data_types):
                data = np.zeros(datashape, dtype=datatype)
                datalist.append(data)
        return datalist 

    def framestr2filename(self, framestr, trajdir):
        if 'left' in trajdir:
            return ['OpticalFlowIntoFuture_' + framestr + '_L.pfm']
        elif 'right' in trajdir:
            return ['OpticalFlowIntoFuture_' + framestr + '_R.pfm']
        else:
            raise NotImplementedError

    def load_frame(self, trajdir, filenamelist):
        flow_all, _ = repeat_function(readPFM, {'file': join(trajdir, filenamelist[0])}, 
                                repeat_times=10, error_msg="loading flow " + filenamelist[0])

        flow = flow_all[:,:,:2].copy()
        mask = flow_all[:,:,2].copy()

        if self.listlen == 1:
            return [flow]
        else:
            return [flow, mask]

if __name__=="__main__":
    from utils import visdepth, visflow
    import cv2
    for frameid in range(1,10):
        # image left
        trajfolder = '/home/amigo/tmp/data/flyingchairs/data'
        datatype = chair_img([(200, 400)])
        datalist0 = datatype.load_data(trajfolder, str(frameid).zfill(5)+'_img1', 100)
        # image right
        datatype = chair_flow([(200, 400)])
        datalist1 = datatype.load_data(trajfolder, str(frameid).zfill(5)+'_img1', 100)
        
        trajfolder = '/home/amigo/tmp/data/sintel/training/final/alley_1'
        # image color
        datatype = sintel_img([(200, 400)])
        datalist2 = datatype.load_data(trajfolder, str(frameid).zfill(4), 100)
        print(len(datalist2), datalist2[0].shape)
        # rgb map
        datatype = sintel_flow([(200, 400)])
        datalist3 = datatype.load_data(trajfolder, str(frameid).zfill(4), 100)
        print(len(datalist3), datalist3[0].shape)

        trajfolder = '/home/amigo/tmp/data/sceneflow/frames_cleanpass/TRAIN/A/0000/left/'
        # height map
        datatype = flying_img([(200, 400)])
        datalist4 = datatype.load_data(trajfolder, str(frameid+5).zfill(4), 100)
        print(len(datalist4), datalist4[0].shape,  datalist4[0].dtype)
        # height map
        # import ipdb;ipdb.set_trace()
        datatype = flying_flow([(200, 400)])
        datalist5 = datatype.load_data(trajfolder, str(frameid+5).zfill(4), 100)
        print(len(datalist5), datalist5[0].shape,  datalist5[0].dtype)

        disp = cv2.hconcat((datalist0[0].transpose(1,2,0), visflow(datalist1[0].transpose(1,2,0))))
        cv2.imshow('img', disp)
        disp2 = cv2.hconcat((datalist2[0].transpose(1,2,0), visflow(datalist3[0].transpose(1,2,0))))
        cv2.imshow('img2', disp2)
        disp3 = cv2.hconcat((datalist4[0].transpose(1,2,0), visflow(datalist5[0].transpose(1,2,0))))
        cv2.imshow('img3', disp3)        
        cv2.waitKey(0)

    import ipdb;ipdb.set_trace()
