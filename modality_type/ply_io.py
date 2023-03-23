from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd

def read_ply(filename):
    plydata = PlyData.read(filename)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=np.float32)
    property_names = data[0].dtype.names

    for i, name in enumerate(property_names):
        data_np[:,i] = data_pd[name]
    return data_np

if __name__=="__main__":
    filename = '/home/amigo/tmp/test_root/coalmine/Data_easy/P000/lidar/000000_lcam_front_lidar.ply'
    lidar = read_ply(filename)
    import ipdb;ipdb.set_trace()