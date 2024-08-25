from torch.utils.data import Dataset
import numpy as np
from os.path import join

class CacherDataset(Dataset):
    '''
    Design choice: one dataset is used for one modality
    Load the data from hard drive to RAM
    This is similar to the original TartanAirDataset, but without considering sequencing, frame skipping, data augmentation, normalization, etc. 
    Only image-like data are supported, including image, depth and flow
    Resize the data if necessary
    Note if the flow or disp is resized, the pixel value won't be changed! 
    '''
    def __init__(self, modality, trajlist, trajlenlist, framelist, datarootdir=""):
        '''
        modality: the object of the modality_type, e.g. imiage_lcam_front
        '''
        self.modality = modality
        self.trajlist = trajlist
        self.trajlenlist = trajlenlist
        self.framelist = framelist
        self.dataroot = datarootdir
        self.trajnum = len(trajlist)

        self.framenum = sum(trajlenlist)
        self.acc_trajlen = [0,] + np.cumsum(trajlenlist).tolist() # [0, num[0], num[0]+num[1], ..]

    def __len__(self):
        return self.framenum

    def idx2traj(self, idx):
        '''
        return: 1. the relative dir of trajectory 
                2. the frame string 
                3. is the frame at the end of the current trajectory (for loading flow)
        '''
        # import ipdb;ipdb.set_trace()
        for k in range(self.trajnum):
            if idx < self.acc_trajlen[k+1]:
                break

        remainingframes = idx-self.acc_trajlen[k]
        # frameind = self.acc_trajlen[k] + remainingframes
        framestr = self.framelist[k][remainingframes]
        frameindex_inv = self.trajlenlist[k] - remainingframes # is this the last few frames where there might no flow data exists

        return self.trajlist[k], framestr, frameindex_inv

    def __getitem__(self, idx):
        # load images from the harddrive
        trajstr, framestr, ind_inv = self.idx2traj(idx)
        trajdir = join(self.dataroot, trajstr)
        data = self.modality.load_data(trajdir, framestr, ind_inv)

        return data

class SimpleDataloader(object):
    '''
    Design choice: one dataset is used for one modality
    Load the data from hard drive to RAM
    This is similar to the original TartanAirDataset, but without considering sequencing, frame skipping, data augmentation, normalization, etc. 
    This is for loading the low dimention data such as IMU, motion and wheel encoder
    Note that since the data is in low dimention, we won't use pytorch dataloader
    '''
    def __init__(self, modality, trajlist, trajlenlist, framelist, datarootdir=""):
        '''
        modality: the object of the modality_type, e.g. rgb_lcam_front
        '''
        self.modality = modality
        self.trajlist = trajlist
        self.trajlenlist = trajlenlist
        self.framelist = framelist
        self.dataroot = datarootdir
        self.trajnum = len(trajlist)
        self.framenum = sum(trajlenlist)

    def __len__(self):
        return self.framenum

    def load_data(self, trajidx):
        # load numpy file for each trajectory and concate them together
        assert trajidx<self.trajnum, "Traj {} exceeds the total number of trajectories {}".format(trajidx, self.trajnum)
        trajstr = self.trajlist[trajidx]
        trajdir = join(self.dataroot, trajstr)
        datalist = self.modality.load_data(trajdir, self.framelist[trajidx])
        for data in datalist:
            assert data.shape[0] == self.trajlenlist[trajidx]*self.modality.freq_mult, \
                "Traj {} mod {} data loaded size {} different from the required size {}".format(trajdir, 
                self.modality.name, data.shape[0], self.trajlenlist[trajidx])
        return datalist

if __name__=="__main__":
    from .modality_type.tartanair_types import rgb_lcam_front, depth_lcam_front, flow_lcam_front
    from .datafile_editor import read_datafile
    from .utils import visdepth, visflow
    import cv2
    from .modality_type.ModBase import TYPEDICT

    # datafile = '/home/amigo/tmp/test_root/coalmine/analyze/data_coalmine_Data_easy_P000.txt'
    # trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafile)
    # rgbtype = rgb_lcam_front((320, 320))
    # depthtype = depth_lcam_front((320, 320))
    # flowtype = flow_lcam_front((320, 320))
    # dataset0 = CacherDataset(rgbtype, trajlist, trajlenlist, framelist, datarootdir="/home/amigo/tmp/test_root")
    # dataset1 = CacherDataset(depthtype, trajlist, trajlenlist, framelist, datarootdir="/home/amigo/tmp/test_root")
    # dataset2 = CacherDataset(flowtype, trajlist, trajlenlist, framelist, datarootdir="/home/amigo/tmp/test_root")
    # for k in range(5,55,5):
    #     print('frame',k)
    #     ss=dataset0[k]
    #     ss2=dataset1[k]
    #     ss3=dataset2[k]
    #     depthvis = visdepth(80./ss2)
    #     flowvis = visflow(ss3)
    #     disp = cv2.hconcat((ss, depthvis, flowvis))
    #     cv2.imshow('img', disp)
    #     cv2.waitKey(0)

    # # from .modality_type.stereo_types import sceneflow_left, sceneflow_right, sceneflow_disp
    # # datafile = '/home/amigo/workspace/pytorch/geometry_vision/data/sceneflow_stereo_local_test.txt'
    # from .modality_type.stereo_types import kitti_left, kitti_right, kitti_disp
    # datafile = '/home/amigo/workspace/pytorch/geometry_vision/data/kitti_stereo.txt'
    # trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafile)
    # rgbtype = kitti_left((320, 320))
    # rgbtype2 = kitti_right((320, 320))
    # disptype = kitti_disp((320, 320))
    # dataset0 = CacherDataset(rgbtype, trajlist, trajlenlist, framelist, datarootdir="/bigdata/tartanvo_data/kitti/stereo")
    # dataset1 = CacherDataset(rgbtype2, trajlist, trajlenlist, framelist, datarootdir="/bigdata/tartanvo_data/kitti/stereo")
    # dataset2 = CacherDataset(disptype, trajlist, trajlenlist, framelist, datarootdir="/bigdata/tartanvo_data/kitti/stereo")

    # for k in range(5,55,5):
    #     print('frame',k)
    #     ss=dataset0[k].transpose(1,2,0)
    #     ss2=dataset1[k].transpose(1,2,0)
    #     ss3=dataset2[k]
    #     # import ipdb;ipdb.set_trace()
    #     depthvis = visdepth(ss3)
    #     disp = cv2.hconcat((ss, ss2, depthvis))
    #     cv2.imshow('img', disp)
    #     cv2.waitKey(0)

    # from .modality_type.euroc_types import euroc_lcam, euroc_ldisp, euroc_lflow, euroc_rcam
    # datafile = '/home/wenshan/workspace/pytorch/geometry_vision/data/euroc/euroc.txt'
    # trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafile)
    # rgbtype = euroc_lcam((320, 320))
    # rgbtype2 = euroc_rcam((320, 320))
    # disptype = euroc_ldisp((320, 320))
    # flowtype = euroc_lflow((320, 320))
    # dataset0 = CacherDataset(rgbtype, trajlist, trajlenlist, framelist, datarootdir="/home/wenshan/tmp/vo_data/euroc")
    # dataset1 = CacherDataset(rgbtype2, trajlist, trajlenlist, framelist, datarootdir="/home/wenshan/tmp/vo_data/euroc")
    # dataset2 = CacherDataset(disptype, trajlist, trajlenlist, framelist, datarootdir="/home/wenshan/tmp/vo_data/euroc")
    # dataset3 = CacherDataset(flowtype, trajlist, trajlenlist, framelist, datarootdir="/home/wenshan/tmp/vo_data/euroc")

    # for k in range(5,35500,10):
    #     print('frame',k)
    #     ss=dataset0[k].transpose(1,2,0)
    #     ss2=dataset1[k].transpose(1,2,0)
    #     ss3=dataset2[k]
    #     ss4=dataset3[k].transpose(1,2,0)
    #     # import ipdb;ipdb.set_trace()
    #     depthvis = visdepth(ss3)
    #     flowvis = visflow(ss4)
    #     disp = cv2.hconcat((ss, ss2, depthvis, flowvis))
    #     cv2.imshow('img', disp)
    #     cv2.waitKey(100)        

    from .modality_type.tartandrive_types import rgb_left, costmap
    from torch.utils.data import DataLoader
    datafile = 'data_cacher/data/tartandrive.txt'
    dataroot = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output'
    trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafile)
    rgbtype = rgb_left([(320, 320)])
    rgbtype2 = costmap([(320, 320)])
    dataset0 = CacherDataset(rgbtype, trajlist, trajlenlist, framelist, datarootdir=dataroot)
    dataset1 = CacherDataset(rgbtype2, trajlist, trajlenlist, framelist, datarootdir=dataroot)

    dataloader0 = DataLoader(dataset0, batch_size=1, shuffle=False, num_workers=4)
    dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=4)

    dataiter0 = iter(dataloader0)
    dataiter1 = iter(dataloader1)

    while True:
        try:
            ss0 = dataiter0.next()
            print(ss0[0].shape)
        except StopIteration:
            dataiter0 = iter(dataloader0)

        try:
            ss1 = dataiter1.next()
            print(ss1[0].shape, ss1[1].shape)
        except StopIteration:
            dataiter1 = iter(dataloader1)
