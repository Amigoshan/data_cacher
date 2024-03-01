import numpy as np
import cv2
import os


def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    if mean is not None and std is not None:
        for t, m, s in zip(tensImg, mean, std):
            t.mul_(s).add_(m) 
    tensImg = np.clip(tensImg * float(255),0,255)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1,2,0)).astype(np.uint8)
    return tensImg

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask > 0
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr

def visdepth(disp, scale=3):
    res = np.clip(disp*scale, 0, 255).astype(np.uint8)
    res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
    # res = np.tile(res[:,:,np.newaxis], (1, 1, 3))
    return res

def visseg(segnp):
    _CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    seg_colors = np.loadtxt(_CURRENT_PATH + '/seg_rgbs.txt', delimiter=',',dtype=np.uint8)
    segvis = np.zeros(segnp.shape+(3,), dtype=np.uint8)

    segvis = seg_colors[ segnp, : ]
    segvis = segvis.reshape( segnp.shape+(3,) )

    return segvis

import open3d as o3d
def vispcd( pc_np, vis_size=(1920, 480), o3d_cam=None):
    # pcd: numpy array
    w, h = (1920, 480)   # default o3d window size

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)

    if o3d_cam:
        camerafile = o3d_cam
        w, h = camerafile['w'], camerafile['h']
        cam = o3d.camera.PinholeCameraParameters()
        
        intr_mat, ext_mat = camerafile['intrinsic'], camerafile['extrinsic']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, 
                        intr_mat[0,0], intr_mat[1,1], 
                        intr_mat[0,-1], intr_mat[1,-1])
        intrinsic.intrinsic_matrix = intr_mat
        cam.intrinsic = intrinsic
        cam.extrinsic = ext_mat

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    vis.add_geometry(pcd)

    if o3d_cam:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(cam)
    
    vis.poll_events()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    img = np.array(img)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, vis_size)

    return img


# uncompress the data
def flow16to32(flow16):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_32b = (flow16 -32768) / 64
    '''
    flow32 = flow16[:,:,:2].astype(np.float32)
    flow32 = (flow32 - 32768) / 64.0

    mask8 = flow16[:,:,2].astype(np.uint8)
    return flow32, mask8

# mask = 1  : CROSS_OCC 
#      = 10 : SELF_OCC
#      = 100: OUT_OF_FOV
#      = 200: OVER_THRESHOLD
def flow32to16(flow32, mask8):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_16b = (flow_32b * 64) + 32768  
    '''
    # mask flow values that out of the threshold -512.0 ~ 511.984375
    mask1 = flow32 < -512.0
    mask2 = flow32 > 511.984375
    mask = mask1[:,:,0] + mask2[:,:,0] + mask1[:,:,1] + mask2[:,:,1]
    # convert 32bit to 16bit
    h, w, c = flow32.shape
    flow16 = np.zeros((h, w, 3), dtype=np.uint16)
    flow_temp = (flow32 * 64) + 32768
    flow_temp = np.clip(flow_temp, 0, 65535)
    flow_temp = np.round(flow_temp)
    flow16[:,:,:2] = flow_temp.astype(np.uint16)
    mask8[mask] = 200
    flow16[:,:,2] = mask8.astype(np.uint16)

    return flow16

def flow32to16_unc(flow32, flow_unc):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_16b = (flow_32b * 64) + 32768  
    flow_unc (float32) [0, 1]
    flow_unc_16b = flow_unc * 65535
    '''
    # convert 32bit to 16bit
    h, w, c = flow32.shape
    flow16 = np.zeros((h, w, 3), dtype=np.uint16)
    flow_temp = (flow32 * 64) + 32768
    flow_temp = np.clip(flow_temp, 0, 65535)
    flow_temp = np.round(flow_temp)
    flow16[:,:,:2] = flow_temp.astype(np.uint16)
    flow16[:,:,2] = (flow_unc * 65535).astype(np.uint16)
    return flow16

def flow16to32_unc(flow16):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_32b = (flow16 -32768) / 64
    '''
    flow32 = flow16[:,:,:2].astype(np.float32)
    flow32 = (flow32 - 32768) / 64.0

    unc = flow16[:,:,2].astype(np.float32) / 65535.
    return flow32, unc

def depth_rgba_float32(depth_rgba):
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


def depth_float32_rgba(depth):
    '''
    depth: float32, h x w
    store depth in uint8 h x w x 4
    and use png compression
    '''
    depth_rgba = depth[...,np.newaxis].view("<u1")
    return depth_rgba


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
