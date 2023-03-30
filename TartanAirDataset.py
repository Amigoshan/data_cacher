from __future__ import print_function

import cv2
import numpy as np
from os.path import join

from .utils import flow16to32, depth_rgba_float32

class TartanAirDataset(object):
    def __init__(self, modlist):
        self.modlist = modlist

    def load_motion(self, trajstr, framenum):
        # motionfile = join(trajstr, 'motion_left.npy')
        motionfile = join(trajstr, 'motion_lcam_front.npy')
        motion = np.load(motionfile)
        return motion[framenum]

    def load_imu(self, trajstr, framenum):
        # accfile = join(trajstr, 'imu', 'accel_left.npy')
        # gyrofile = join(trajstr, 'imu', 'gyro_left.npy')
        accfile = join(trajstr, 'imu', 'acc.npy')
        gyrofile = join(trajstr, 'imu', 'gyro.npy')
        acc = np.load(accfile)
        gyro = np.load(gyrofile)
        return np.concatenate((acc[framenum*10], gyro[framenum*10]), axis=-1)

    def get_data(self, trajstr, framestr):
        # parse the idx to trajstr
        sample = {}
        framenum = int(framestr)
        for datatype in self.modlist: 
            datafilelist = self.getDataPath(trajstr, framestr, datatype)
            if datatype == 'img0' or datatype == 'img1':
                imglist = self.load_image(datafilelist)
                if imglist is None:
                    print("!!!READ IMG ERROR {}, {}, {}".format(trajstr, framestr, datafilelist))
                sample[datatype] = imglist
            elif datatype == 'depth0' or datatype == 'depth1':
                depthlist = self.load_depth(datafilelist)
                sample[datatype] = depthlist
            elif datatype[:4] == 'flow':
                flowlist = self.load_flow(datafilelist)
                sample['flow'] = flowlist # do not distinguish flow flow2 anymore
            elif datatype == 'motion':
                motionlist = self.load_motion(trajstr, framenum)
                sample[datatype] = motionlist
            elif datatype == 'imu': 
                imulist = self.load_imu(trajstr, framenum)
                sample[datatype] = imulist
            else:
                print('Unknow Datatype {}'.format(datatype))
        return sample

    def getDataPath_v1(self, trajstr, framestr, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        if datatype == 'img0':
            return trajstr + '/image_left/' + framestr + '_left.png'
        if datatype == 'img1':
            return trajstr + '/image_right/' + framestr + '_right.png'
        if datatype == 'disp0' or datatype == 'depth0':
            return trajstr + '/depth_left/' + framestr + '_left_depth.png'
        if datatype == 'disp1' or datatype == 'depth1':
            return trajstr + '/depth_right/' + framestr + '_right_depth.png'

        if datatype == 'flow':
            flownum = 1
            flowfolder = 'flow'
            framestr2 = str(int(framestr) + flownum).zfill(len(framestr))
            return trajstr + '/' + flowfolder + '/' + framestr + '_' + framestr2 + '_flow.png'

    def getDataPath(self, trajstr, framestr, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        if datatype == 'img0':
            return trajstr + '/image_lcam_front/' + framestr + '_lcam_front.png'
        if datatype == 'img1':
            return trajstr + '/image_rcam_front/' + framestr + '_rcam_front.png'
        if datatype == 'disp0' or datatype == 'depth0':
            return trajstr + '/depth_lcam_front/' + framestr + '_lcam_front_depth.png'
        if datatype == 'disp1' or datatype == 'depth1':
            return trajstr + '/depth_rcam_front/' + framestr + '_rcam_front_depth.png'

        if datatype == 'flow':
            flownum = 1
            flowfolder = 'flow_lcam_front'
            framestr2 = str(int(framestr) + flownum).zfill(len(framestr))
            return trajstr + '/' + flowfolder + '/' + framestr + '_' + framestr2 + '_flow.png'

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        flow16 = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        assert flow16 is not None, "Error loading flow {}".format(fn)
        flow32, mask = flow16to32(flow16)
        return flow32

    def load_depth(self, fn):
        depth_rgba = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        assert depth_rgba is not None, "Error loading depth {}".format(fn)
        depth = depth_rgba_float32(depth_rgba)

        return depth

    def load_image(self, fn):
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        return img


if __name__ == '__main__':
    import time

    # test the efficiency of the dataloader for sequential data
    np.set_printoptions(suppress=True)    # test add noise and mask to flow
    # rootdir = '/home/wenshan/tmp/data/tartan'
    rootdir = "/home/amigo/tmp/data/tartan"
    framefile = './data/tartan_train.txt'
    typestr = "depth0,img0,img1,imu,motion,flow"#,imu,motion
    modlist = typestr.split(',')

    dataset = TartanAirDataset(modlist=modlist)
    sample = dataset.get_data(trajstr='amusement/Data_fast/P001', framestr='000000')
    import ipdb;ipdb.set_trace()
    # for k in range(100):
        # try:
        #     sample = dataiter.next()
        # except StopIteration:
        #     dataiter = iter(dataloader)
        #     sample = dataiter.next()
        # print(k,time.time()-lasttime, sample.keys(),sample['imu'].shape)
        # lasttime = time.time()

        # # data visualization
        # disps = []
        # for kk in range(10):
        #     img = sample['img0'][0,kk,:].numpy().transpose(1,2,0)
        #     img = (img).astype(np.uint8)
        #     disps.append(img)
        # # import ipdb;ipdb.set_trace()
        # disps = np.array(disps)
        # disps = disps.reshape(-1,disps.shape[-2],3)
        # disps = cv2.resize(disps,(0,0),fx=0.3,fy=0.3)
        # cv2.imshow('img',disps)
        # cv2.waitKey(0)
    print("total time", time.time()-starttime)
