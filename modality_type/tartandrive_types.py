import cv2
from .ModBase import SimpleModBase, FrameModBase, register, TYPEDICT, repeat_function
from os.path import join
import numpy as np
from .ply_io import read_ply

'''
We assume one modality may contains multiple data types. 
E.g. costmap contains cost and velocity
The load_data function returns a list of data

In the low level, each modality corresponds to a folder in the traj folder
This file defines the interfaces of the Modality: 
    - folder name
    - function that convert framestr to file
    - data type list
    - data shape list
'''

class IMUBase(SimpleModBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.data_shapes = [(6,)]

        self.freq_mult = 10
        self.drop_last = 10

        self.folder_name = 'imu'

    def data_padding(self, k):
        return np.zeros((10,) + tuple(self.data_shapes[k]), dtype=np.float32)

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

    def resize_data(self, lidarlist):
        return lidarlist

class RGBModBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        listlen = len(datashapelist) # this is usually one
        self.data_types = []
        for k in range(listlen):
            self.data_shapes[k] = (3,) + tuple(self.data_shapes[k] )
            self.data_types.append(np.uint8)

    def load_frame(self, trajdir, filenamelist):
        # read image
        imglist = []
        for filename in filenamelist:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                img = repeat_function(cv2.imread, {'filename':join(trajdir,filename), 'flags':cv2.IMREAD_UNCHANGED}, repeat_times=10)
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

class GreyModBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        listlen = len(datashapelist) # this is usually one
        self.data_types = []
        for k in range(listlen):
            self.data_shapes[k] = tuple(self.data_shapes[k] )
            self.data_types.append(np.uint8)

    def load_frame(self, trajdir, filenamelist):
        # read image
        imglist = []
        for filename in filenamelist:
            img = repeat_function(cv2.imread, {'filename':join(trajdir,filename), 'flags':cv2.IMREAD_UNCHANGED}, repeat_times=10)
            imglist.append(img)
        return imglist

    def resize_data(self, imglist):
        # resize image
        for k, img in enumerate(imglist):
            h, w = img.shape[0], img.shape[1]
            target_h, target_w = self.data_shapes[k][0], self.data_shapes[k][1]
            if h != target_h or w != target_w:
                imglist[k] = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR )
        return imglist

    def transpose(self, imglist):
        return imglist

class DepthModBase(FrameModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        listlen = len(datashapes) # this is usually one
        self.data_types = []
        for k in range(listlen):
            self.data_types.append(np.float32)

    def load_frame(self, trajdir, filenamelist):
        depthlist = []
        for filename in filenamelist:
            depth_rgba = repeat_function(cv2.imread, {'filename':join(trajdir,filename), 'flags':cv2.IMREAD_UNCHANGED}, repeat_times=10)
            depth = depth_rgba.view("<f4")
            depth = np.squeeze(depth, axis=-1)
            depthlist.append(depth)
        return depthlist

    def resize_data(self, depthlist):
        # resize image
        for k, depth in enumerate(depthlist):
            (h, w) = depth.shape
            target_h, target_w = self.data_shapes[k]
            if h != target_h or w != target_w:
                depthlist[k] = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST )
        return depthlist

class FlowModBase(FrameModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        # we assume that the flow might return flow or (flow, mask)
        # we also assume that the flow will always be returned, the mask is optional
        self.listlen = len(datashapes) # this is usually one
        self.data_shapes[0] = (2,) + tuple(self.data_shapes[0]) # add one dim to the 
        self.data_type = [np.float32, np.uint8] # for flow and mask

    def load_frame(self, filenamelist):
        # if filename is None: 
        #     return np.zeros((10,10,2), dtype=np.float32), np.zeros((10,10), dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow16 = repeat_function(cv2.imread, {'filename':filenamelist[0], 'flags':cv2.IMREAD_UNCHANGED}, repeat_times=10)
        flow32 = flow16[:,:,:2].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0

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

@register(TYPEDICT)
class grey_left(GreyModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "image_left"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.png')]

@register(TYPEDICT)
class grey_right(GreyModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "image_right"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.png')]

@register(TYPEDICT)
class rgb_left(RGBModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "image_left_color"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.png')]

@register(TYPEDICT)
class rgb_map(RGBModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "rgb_map"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.npy')]

@register(TYPEDICT)
class rgb_map_ff(RGBModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "rgb_map_ff"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.npy')]

    def transpose(self, imglist):
        reslist = []
        for img in imglist:
            # flip the data because the raw format issue
            img = cv2.flip(img, 1)
            reslist.append(img.transpose(2,0,1))
        return reslist

@register(TYPEDICT)
class rgb_map_ff_v2(RGBModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "rgb_map_ff"
    
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
            if filename.endswith('.png') or filename.endswith('.jpg'):
                img = cv2.imread(join(trajdir,filename), cv2.IMREAD_UNCHANGED)
                assert img is not None, "Error loading RGB {}".format(filename)
            elif filename.endswith('.npy'):
                img = np.load(join(trajdir,filename))
            else:
                raise NotImplementedError
            # bgr-rgb
            img = img[:,:,::-1].copy()
            img = img.transpose(1,0,2)
            imglist.append(img)
        return imglist

    def transpose(self, imglist):
        reslist = []
        for img in imglist:
            # flip the data because the raw format issue
            reslist.append(img.transpose(2,0,1))
        return reslist

@register(TYPEDICT)
class rgb_map_v2(rgb_map_ff_v2):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "rgb_map"

@register(TYPEDICT)
class rgb_map_ff_v2_2cm(rgb_map_ff_v2):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "rgb_map_2cm_ff"

@register(TYPEDICT)
class rgb_map_v2_2cm(rgb_map_ff_v2):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "rgb_map_2cm"

@register(TYPEDICT)
class height_map(FrameModBase):
 
    def __init__(self, datashapelist):
        '''
        the heightmap has four channels
        '''
        super().__init__(datashapelist)
        self.channel_num = 4
        listlen = len(datashapelist) # this is usually one
        self.data_types = []
        for k in range(listlen):
            self.data_shapes[k] =  (self.channel_num,) + tuple(self.data_shapes[k])
            self.data_types.append(np.float32)
        self.folder_name = "height_map"

    def load_frame(self, trajdir, filenamelist):
        # read image
        maplist = []
        for filename in filenamelist:
            heightmap = np.load(join(trajdir,filename))
            assert heightmap is not None, "Error loading map {}".format(filename)
            maplist.append(heightmap)
        return maplist

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
        return [join(self.folder_name, framestr + '.npy')]

@register(TYPEDICT)
class height_map_ff_format(height_map):
 
    def __init__(self, datashapelist):
        '''
        load the four-channel heightmap
        convert it to the two-channel ff format 
        '''
        super().__init__(datashapelist)
        self.channel_num = 2
        listlen = len(datashapelist) # this is usually one
        for k in range(listlen):
            self.data_shapes[k] =  (self.channel_num,) + tuple(datashapelist[k])

    def load_frame(self, trajdir, filenamelist):
        # read image
        maplist = []
        for filename in filenamelist:
            heightmap = np.load(join(trajdir,filename))
            assert heightmap is not None, "Error loading map {}".format(filename)
            mask = heightmap[:,:,0] < 10
            mean = heightmap[:,:,2]
            mean[heightmap[:,:,0]>10] = 0.
            heightmap_ff = np.stack((mean,mask), axis=-1)
            maplist.append(heightmap_ff)
        return maplist


@register(TYPEDICT)
class height_map_ff(height_map):
 
    def __init__(self, datashapelist):
        '''
        the heightmap has four channels
        '''
        super().__init__(datashapelist)
        self.channel_num = 2
        listlen = len(datashapelist) # this is usually one
        self.data_shapes = datashapelist # needs to be filled in derived classes
        for k in range(listlen):
            self.data_shapes[k] =  (self.channel_num,) + tuple(self.data_shapes[k])
        self.folder_name = "height_map_ff"

    def transpose(self, imglist):
        reslist = []
        for img in imglist:
            # flip the data because the raw format issue
            img = cv2.flip(img, 1)
            reslist.append(img.transpose(2,0,1))
        return reslist

@register(TYPEDICT)
class height_map_ff_v2(height_map):
 
    def __init__(self, datashapelist):
        '''
        the heightmap has four channels
        '''
        super().__init__(datashapelist)
        self.channel_num = 2
        listlen = len(datashapelist) # this is usually one
        self.data_shapes = datashapelist # needs to be filled in derived classes
        for k in range(listlen):
            self.data_shapes[k] =  (self.channel_num,) + tuple(self.data_shapes[k])
        self.folder_name = "height_map_ff"

    def load_frame(self, trajdir, filenamelist):
        # read image
        maplist = []
        for filename in filenamelist:
            heightmap = np.load(join(trajdir,filename))
            assert heightmap is not None, "Error loading map {}".format(filename)

            heightmap_sum = heightmap.sum(axis=0)
            mask = np.abs(heightmap_sum) > 0
            mean = heightmap[4,:,:]
            heightmap_mask = np.stack((mean,mask.astype(np.float32)), axis=-1)
            heightmap_mask = heightmap_mask.transpose(1,0,2)
            maplist.append(heightmap_mask)
        return maplist

    def transpose(self, imglist):
        reslist = []
        for img in imglist:
            reslist.append(img.transpose(2,0,1))
        return reslist

@register(TYPEDICT)
class height_map_v2(height_map_ff_v2):
 
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.folder_name = "height_map"

@register(TYPEDICT)
class height_map_ff_v2_2cm(height_map_ff_v2):
 
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.folder_name = "height_map_2cm_ff"

@register(TYPEDICT)
class height_map_v2_2cm(height_map_ff_v2):
 
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        self.folder_name = "height_map_2cm"

@register(TYPEDICT)
class costmap(FrameModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.channel_num = 2
        self.folder_name = "costmap"
        self.vel_max_len = 50
        self.data_types = [np.uint8, np.float32] # one for cost one for velocity
        self.data_shapes = [(2,) + tuple(datashapes[0]), (self.vel_max_len,2)] # hard code the shape of the velocity

    def load_frame(self, trajdir, filenamelist):
        costmap = np.load(join(trajdir,filenamelist[0]))
        velocityfile = join(trajdir,filenamelist[0].replace('.npy', '_vel.txt'))
        vel = np.loadtxt(velocityfile, dtype=np.float32)

        if len(vel.shape) == 1:
            if vel.size == 0:
                vel = np.zeros((1, self.channel_num), dtype=np.float32)
            else:
                vel = vel.reshape(1, self.channel_num)
            
        if vel.shape[0]>= self.vel_max_len:
             vel = vel[:self.vel_max_len,:]
        else: # pad vel with zero
            vel = np.concatenate((vel, np.zeros((self.vel_max_len-vel.shape[0], 2), dtype=np.float32)))

        return [costmap, vel]

    def resize_data(self, datalist): 
        costmap = datalist[0]
        h, w = costmap.shape[0], costmap.shape[1]
        target_h, target_w = self.data_shapes[0][1], self.data_shapes[0][2]
        if h != target_h or w != target_w:
            costmap = cv2.resize(costmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR )

        return [costmap, datalist[1]]

    def transpose(self, imglist):
        imglist[0] = imglist[0].transpose(2,0,1)
        return imglist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.npy')]

@register(TYPEDICT)
class costmap_v2(costmap):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "costmap_v3"

@register(TYPEDICT)
class costmap_v2_2cm(costmap):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        self.folder_name = "costmap_imu"

@register(TYPEDICT)
class depth_left_tartan(DepthModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "depth_left"
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.npy')]


@register(TYPEDICT)
class imu_v1(IMUBase):
    '''
    This defines modality that is light-weight
    such as IMU, pose, wheel_encoder
    '''
    def get_filename(self):
        return [join(self.folder_name, 'imu.npy')]

# @register(TYPEDICT)
# class lidar(LiDARBase):
#     def __init__(self, datashape):
#         super().__init__(datashape)
#         self.folder_name = 'lidar'
#         self.file_suffix = 'lcam_front_lidar'

#     def framestr2filename(self, framestr):
#         return join(self.folder_name, framestr + '_' + self.file_suffix + '.ply')

def get_vis_heightmap(heightmap, scale=0.5, hmin=-1, hmax=4):
    FLOATMAX = 1000000.0

    mask = heightmap[:,:,0]>1000
    disp1 = np.clip((heightmap[:, :, 0] - hmin)*100, 0, 255).astype(np.uint8)
    disp2 = np.clip((heightmap[:, :, 1] - hmin)*100, 0, 255).astype(np.uint8)
    disp3 = np.clip((heightmap[:, :, 2] - hmin)*100, 0, 255).astype(np.uint8)
    disp4 = np.clip(heightmap[:, :, 3]*1000, 0, 255).astype(np.uint8)
    disp1[mask] = 0
    disp2[mask] = 0
    disp3[mask] = 0
    disp4[mask] = 0
    disp_1 = np.concatenate((disp1, disp2) , axis=1)
    disp_2 = np.concatenate((disp3, disp4) , axis=1)
    disp = np.concatenate((disp_1, disp_2) , axis=0)
    disp = cv2.resize(disp, (0, 0), fx=scale, fy=scale)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return disp_color

def get_vis_heightmap2(heightmap, scale=0.5, hmin=-4, hmax=4):
    disp1 = np.clip((heightmap[:, :, 0] - hmin)*30, 0, 255).astype(np.uint8)
    disp2 = np.clip(heightmap[:, :, 1]*255, 0, 255).astype(np.uint8)
    disp = np.concatenate((disp1, disp2) , axis=1)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return disp_color

def get_vis_costmap(costmap):
    disp = np.clip((costmap[0,:,:]),0,255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return disp_color

if __name__=="__main__":
    # trajfolder = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_10'
    # for frameid in range(300):
    #     # image left
    #     datatype = grey_left([(544, 512)])
    #     datalist0 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     # image right
    #     datatype = grey_right([(544, 512)])
    #     datalist1 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     # image color
    #     datatype = rgb_left([(200, 400)])
    #     datalist2 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     print(len(datalist2), datalist2[0].shape)
    #     # rgb map
    #     datatype = rgb_map([(600, 600)])
    #     datalist3 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     print(len(datalist3), datalist3[0].shape)
    #     # height map
    #     datatype = height_map([(600, 600)])
    #     datalist4 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     print(len(datalist4), datalist4[0].shape,  datalist4[0].dtype)
    #     visheight = get_vis_heightmap(datalist4[0].transpose(1,2,0), scale=1.0)
    #     # cost map
    #     datatype = costmap([(600, 600)])
    #     datalist5 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     print(len(datalist5), datalist5[0].shape, datalist5[1].shape,  datalist5[0].dtype, datalist5[1])
    #     # imu
    #     # rgb map
    #     datatype = rgb_map_ff([(600, 600)])
    #     datalist6 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     print(len(datalist6), datalist6[0].shape)
    #     # height map
    #     datatype = height_map_ff([(600, 600)])
    #     datalist7 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     print(len(datalist7), datalist7[0].shape,  datalist7[0].dtype)
    #     visheight2 = get_vis_heightmap2(datalist7[0].transpose(1,2,0), scale=1.0)

    #     datatype = height_map_ff_format([(600, 600)])
    #     datalist8 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
    #     print(len(datalist8), datalist8[0].shape,  datalist8[0].dtype)
    #     visheight3 = get_vis_heightmap2(datalist8[0].transpose(1,2,0), scale=1.0)
    #     # import ipdb;ipdb.set_trace()

    #     # import ipdb;ipdb.set_trace()
    #     disp = cv2.hconcat((datalist3[0].transpose(1,2,0), datalist6[0].transpose(1,2,0)))
    #     cv2.imshow('img', disp)
    #     disp2 = cv2.hconcat((visheight, cv2.vconcat((visheight2, visheight3))))
    #     disp2 = cv2.resize(disp2, (0,0), fx=0.8, fy=0.8)
    #     cv2.imshow('img2', disp2)        
    #     cv2.waitKey(0)

    trajfolder = '/cairo/arl_bag_files/2023_traj/meadows_2023-09-14-12-07-28'
    for frameid in range(300):
        # image left
        # image color
        # import ipdb;ipdb.set_trace()
        datatype = rgb_left([(400, 960)])
        datalist2 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        print(len(datalist2), datalist2[0].shape)
        # rgb map
        datatype = rgb_map_ff_v2([(600, 240)])
        datalist3 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        print(len(datalist3), datalist3[0].shape)
        # height map
        datatype = height_map_ff_v2([(600, 240)])
        datalist4 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        print(len(datalist4), datalist4[0].shape,  datalist4[0].dtype)
        visheight = get_vis_heightmap2(datalist4[0].transpose(1,2,0), scale=1.0)
        # cost map
        datatype = costmap_v2([(600, 240)])
        datalist5 = datatype.load_data(trajfolder, str(frameid).zfill(6), 100)
        viscostmap = get_vis_costmap(datalist5[0])
        print(len(datalist5), datalist5[0].shape, datalist5[1].shape,  datalist5[0].dtype, datalist5[1])

        # import ipdb;ipdb.set_trace()
        disp = cv2.hconcat((datalist3[0].transpose(1,2,0), visheight, viscostmap))
        disp2 = cv2.vconcat((datalist2[0].transpose(1,2,0), disp))
        disp2 = cv2.resize(disp2, (0,0), fx=0.8, fy=0.8)
        cv2.imshow('img2', disp2)        
        cv2.waitKey(0)

    import ipdb;ipdb.set_trace()
