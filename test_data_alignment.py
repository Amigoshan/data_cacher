import time
import numpy as np
#from torch.utils.data import DataLoader
from .TartanAirDataset import TartanAirDataset
from .MultiDatasets import MultiDatasets
from .utils import visflow, visdepth
import cv2
import torch
from os.path import split

def check_same(t1,t2,outstr=''):
    diff = t1-t2
    if diff.max() > 1e-5:
        print("==> Data mismatch",outstr )
        return False
    if diff.min() < -1e-5:
        print("==> Data mismatch",outstr )
        return False
    return True


if __name__ == '__main__':

    # test the efficiency of the dataloader for sequential data
    np.set_printoptions(suppress=True)    # test add noise and mask to flow
    # rootdir = '/home/wenshan/tmp/data/tartan'
    rootdir = '/ocean/projects/cis220039p/shared/tartanair_v2'
    # rootdir = '/home/amigo/tmp/data/tartan'
    # framefile = 'data/tartan_train_local.txt'
    typestr = "depth0,img0,img1,imu,motion,flow"#,imu,motion
    modalitylens = [2,3,1,20,3,1] #,100,10
    frame_skip = 1
    frame_stride = 2
    batch = 32
    workernum = 0

    dataset = TartanAirDataset(modlist=typestr.split(','))

    dataset_specfile = 'data_cacher/dataspec/flowvo_train_local_v1.yaml'
    # dataset_specfile = 'data_cacher/dataspec/flowvo_train_local_v2.yaml'
    trainDataloader = MultiDatasets(dataset_specfile, 
                       'psc', 
                       batch=batch, 
                       workernum=0, 
                       shuffle=True)
    tic = time.time()
    num = 1000                       
    for k in range(num):
        sample = trainDataloader.load_sample()
        print(sample.keys())

        # try:
        #     sample2 = dataiter.next()
        # except StopIteration:
        #     dataiter = iter(dataloader)
        #     sample2 = dataiter.next()
        # print(sample2.keys())
        #    sample = dataset.get_data(trajstr='amusement/Data_fast/P001', framestr='000000')

        # import ipdb;ipdb.set_trace()
        # time.sleep(0.02)
        mm = sample['motion']
        acc= sample['acc']
        gyro=sample['gyro']
        imu = torch.cat((acc,gyro),dim=-1)


        # mmm = sample2['motion']
        # imu2 = sample2['imu']

        # check_same(mm,mmm)
        # check_same(imu, imu2)

        for b in range(batch):
            ss=sample['img0'][b][0].numpy()
            ss2=sample['depth0'][b][0].numpy()
            ss3=sample['flow'][b][0].numpy()

            trajframestr = sample['trajdir'][b]
            trajstr, framestr = split(trajframestr)
            sample2 = dataset.get_data(trajstr, framestr)
            sss=sample2['img0']
            sss2=sample2['depth0']
            sss3=sample2['flow']

            # import ipdb;ipdb.set_trace()
            match = True
            match = match and check_same(ss,sss,'img0')
            match = match and check_same(ss2, sss2, 'depth0')
            match = match and check_same(ss3, sss3, 'flow')

            mmm = sample2['motion']
            imu2 = sample2['imu']
            match = match and check_same(mm[b][0],mmm, 'motion')
            match = match and check_same(imu[b][0], imu2, 'imu')

            if not match:
                import ipdb;ipdb.set_trace()

#            depthvis = visdepth(80./ss2)
#            flowvis = visflow(ss3)
#            disp = cv2.hconcat((ss, depthvis, flowvis))
#            cv2.imshow('img', disp)
#            cv2.waitKey(10)

#            depthvis = visdepth(80./sss2)
#            flowvis = visflow(sss3)
#            disp = cv2.hconcat((sss, depthvis, flowvis))
#            cv2.imshow('img2', disp)
#            cv2.waitKey(10)


        # import ipdb;ipdb.set_trace()

    print((time.time()-tic))
    trainDataloader.stop_cachers()


    
