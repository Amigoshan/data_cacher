# This file handles the root directories on different platform
# E.g. it allows you use the same spec file on your local machine and the remote cluster
DataRoot = {}
DataRoot['local'] = {
                        'sceneflow':    '/home/amigo/tmp/data/sceneflow',
                        'tartan':       '/home/amigo/tmp/data/tartan',
                        'tartan2':      '/home/amigo/tmp/test_root',
                        'shibuya':      '/prague/tartanvo_data/AirDOS_shibuya',
                        'tartan-cvpr':  '/cairo/tartanair_test_cvpr',
                        'chairs':       '/home/amigo/tmp/data/flyingchairs',
                        'flying':       '/home/amigo/tmp/data/sceneflow',
                        'sintel':       '/home/amigo/tmp/data/sintel/training',
                        'euroc':        '/home/wenshan/tmp/vo_data/euroc',
                        'kitti-stereo': '/peru/tartanvo_data/kitti/stereo',
                        'kitti-vo':     '/home/wenshan/tmp/vo_data/kittivo',
                        'tartandrive':  '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output',
                        'tartandrive2':  '/cairo/arl_bag_files/2023_traj'
}

DataRoot['desktop'] = {
                        'sceneflow':    '/data/stereo_data/sceneflow',
                        'tartan':       '/data/tartanair',
                        'tartan2':      '/data/tartanair_v2',
                        'shibuya':      '/prague/tartanvo_data/AirDOS_shibuya',
                        'euroc':        '/home/wenshan/tmp/vo_data/euroc',
                        'kitti-vo':     '/home/wenshan/tmp/vo_data/kittivo',
                        'kitti-stereo': '/data/stereo_data/kitti/training',
                        'tartan2_event':'/data/tartanair_v2_event'
}

DataRoot['wsl'] = {
'tartan2': '/mnt/e'
}

DataRoot['cluster'] = {
                        'sceneflow':    '/data2/datasets/yaoyuh/StereoData/SceneFlow',
                        'tartan':       '/project/learningvo/tartanair_v1_5',
                        'tartan2':      '/compute/zoidberg/tartanair_v2',
                        'chairs':       '/project/learningvo/flowdata/FlyingChairs_release',
                        'flying':       '/data2/datasets/yaoyuh/StereoData/SceneFlow',
                        'sintel':       '/project/learningvo/flowdata/sintel/training',
                        'euroc':        '/project/learningvo/euroc',
                        'kitti-stereo': '/project/learningvo/stereo_data/kitti/training',
                        'tartandrive':  '/project/learningphysics' 
}

DataRoot['zoidberg'] = {
                        'sceneflow':    '/data2/datasets/yaoyuh/StereoData/SceneFlow',
                        'tartan':       '/project/learningvo/tartanair_v1_5',
                        'tartan2':	    '/scratch/tartanair_v2',
                        'chairs':       '/project/learningvo/flowdata/FlyingChairs_release',
                        'sintel':       '/project/learningvo/flowdata/sintel/training',
                        'flying':       '/data2/datasets/yaoyuh/StereoData/SceneFlow',
                        'kitti-stereo': '/project/learningvo/stereo_data/kitti/training', 
                        'tartan2':      '/scratch/tartanair_v2',
}


DataRoot['dgx'] = {
                        'sceneflow':    '/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow',
                        'tartan-c':     '/tmp2/wenshan/tartanair_v1_5',
                        'chairs':       '/tmp2/wenshan/flyingchairs',
                        'flying':       '/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow',
                        'sintel':       '/tmp2/wenshan/sintel/training',
                        'kitti-stereo': '/tmp2/wenshan/kitti/training', 
}

DataRoot['psc'] = {
                        #'tartan':       '/ocean/projects/cis220039p/shared/TartanAir/tartanair_v1',
                        'tartan':       '/jet/projects/cis220039p/tartanair_v1',
                        'tartan2':      '/ocean/projects/cis220039p/shared/tartanair_v2',   
                        'euroc':        '/ocean/projects/cis220039p/wenshanw/euroc',
                        'tartandrive2': '/ocean/projects/cis220039p/shared/tartandrive/2023_traj/v1',
                        'tartan2_event':'/ocean/projects/cis220039p/shared/tartanair_v2_event',
                        'blinkflow':    '/ocean/projects/cis220039p/shared/blinkflow/train',
                        'chairs':       '/ocean/projects/cis220039p/shared/vo/flyingchairs',
                        'flying':       '/ocean/projects/cis220039p/shared/vo/SceneFlow',
                        'sintel':       '/ocean/projects/cis220039p/shared/vo/sintel',
                        'euroc':        '/ocean/projects/cis220039p/shared/vo/euroc',
                        'kitti-vo':     '/ocean/projects/cis220039p/shared/vo/kitti/vo',
                        'kitti-flow':   '/ocean/projects/cis220039p/shared/vo/kitti/stereo_flow_cropped',
                        'chairs':       '/ocean/projects/cis220039p/shared/vo/flyingchairs',
                        'sintel':        '/ocean/projects/cis220039p/shared/vo/sintel/training',
                        'flying':        '/ocean/projects/cis220039p/shared/vo/SceneFlow',
}




STEREO_DR = {'sceneflow':   {'local':   ['/home/amigo/tmp/data/sceneflow', '/home/amigo/tmp/data/sceneflow'],
                            'cluster':  ['/data/datasets/yaoyuh/StereoData/SceneFlow', '/data/datasets/yaoyuh/StereoData/SceneFlow'],
                            'azure':    ['SceneFlow', 'SceneFlow'],
                            'dgx':      ['/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow', '/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/SceneFlow','/ocean/projects/cis210086p/wenshanw/SceneFlow'],
                            }, 
            'tartan':       {'local':   ['/home/amigo/tmp/data/tartan', '/home/amigo/tmp/data/tartan'],
                            'local_test':  ['/peru/tartanair', '/peru/tartanair'],
                            'cluster':  ['/data/datasets/wenshanw/tartan_data', '/data/datasets/wenshanw/tartan_data'],
                            'cluster2':  ['/project/learningvo/tartanair_v1_5', '/project/learningvo/tartanair_v1_5'],
                            'azure':    ['', ''],
                            'dgx':      ['/tmp2/wenshan/tartanair_v1_5', '/tmp2/wenshan/tartanair_v1_5'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/tartanair_v1_5','/ocean/projects/cis210086p/wenshanw/tartanair_v1_5'],
                            },
            'kitti':       {'local':    ['/prague/tartanvo_data/kitti/stereo', '/prague/tartanvo_data/kitti/stereo'], # DEBIG: stereo
                            'cluster':  ['/project/learningvo/stereo_data/kitti/training', '/project/learningvo/stereo_data/kitti/training'],
                            'azure':    ['', ''], # NO KITTI on AZURE yet!!
                            'dgx':      ['/tmp2/wenshan/kitti/training', '/tmp2/wenshan/kitti/training'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/kitti/training','/ocean/projects/cis210086p/wenshanw/kitti/training'],
                            },
            'euroc':       {'local':   ['/prague/tartanvo_data/euroc', '/prague/tartanvo_data/euroc'],
                            },
            }


# Datasets for FlowVo
FLOWVO_DR = {'tartan':      {'local':   '/home/amigo/tmp/data/tartan', # '/home/amigo/tmp/data/tartanair_pose_and_imu',# 
                            'local2':   '/home/amigo/tmp/data/tartanair_pose_and_imu', #'/cairo/tartanair_test_cvpr', # '/home/amigo/tmp/data/tartan', # 
                            'local_test':  '/peru/tartanair',
                            'cluster':  '/data/datasets/wenshanw/tartan_data',
                            'cluster2':  '/project/learningvo/tartanair_v1_5',
                            'azure':    '',
                            'dgx':      '/tmp2/wenshan/tartanair_v1_5',
                            'psc':      '/ocean/projects/cis210086p/wenshanw/tartanair_v1_5',
                            }, 
             'euroc':       {'local':   '/prague/tartanvo_data/euroc', 
                            'cluster2':  '/project/learningvo/euroc',
                            },
             'kitti':       {'local':   '/prague/tartanvo_data/kitti/vo', 
                            },
}

# Datasets for Flow
FLOW_DR =   {'flyingchairs':{'local':   ['/home/amigo/tmp/data/flyingchairs', '/home/amigo/tmp/data/flyingchairs'],
                            'cluster':  ['/project/learningvo/flowdata/FlyingChairs_release', '/project/learningvo/flowdata/FlyingChairs_release'],
                            'azure':    ['FlyingChairs_release', 'FlyingChairs_release'],
                            'dgx':      ['/tmp2/wenshan/flyingchairs', '/tmp2/wenshan/flyingchairs'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/flyingchairs','/ocean/projects/cis210086p/wenshanw/flyingchairs'],
                            }, 
            'flyingthings': {'local':   ['/home/amigo/tmp/data/sceneflow', '/home/amigo/tmp/data/sceneflow/frames_cleanpass'],
                            'cluster':  ['/data/datasets/yaoyuh/StereoData/SceneFlow', '/project/learningvo/flowdata/optical_flow'],
                            'azure':    ['SceneFlow','SceneFlow'],
                            'dgx':      ['/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow', '/tmp2/wenshan/optical_flow'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/SceneFlow','/ocean/projects/cis210086p/wenshanw/optical_flow'],
                            }, 
            'sintel':       {'local':   ['/home/amigo/tmp/data/sintel/training', '/home/amigo/tmp/data/sintel/training'],
                            'cluster':  ['/project/learningvo/flowdata/sintel/training', '/project/learningvo/flowdata/sintel/training'],
                            'azure':    ['sintel/training', 'sintel/training'],
                            'dgx':      ['/tmp2/wenshan/sintel/training', '/tmp2/wenshan/sintel/training'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/sintel/training','/ocean/projects/cis210086p/wenshanw/sintel/training'],
                            }, 
            'tartan':       {'local':   ['/home/amigo/tmp/data/tartan', '/home/amigo/tmp/data/tartan'],
                            'local_test':  ['/peru/tartanair', '/peru/tartanair'],
                            'cluster':  ['/data/datasets/wenshanw/tartan_data', '/data/datasets/wenshanw/tartan_data'],
                            'cluster2':  ['/project/learningvo/tartanair_v1_5', '/project/learningvo/tartanair_v1_5'],
                            'azure':    ['', ''],
                            'dgx':      ['/tmp2/wenshan/tartanair_v1_5', '/tmp2/wenshan/tartanair_v1_5'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/tartanair_v1_5','/ocean/projects/cis210086p/wenshanw/tartanair_v1_5'],
                            }, 
            'euroc':        {'local':   ['/prague/tartanvo_data/euroc', '/prague/tartanvo_data/euroc'],
                            'cluster2':  ['/project/learningvo/euroc', '/project/learningvo/euroc'],
                            },
            'kitti':        {'local':   ['/prague/tartanvo_data/kitti/vo', '/prague/tartanvo_data/kitti/vo'],
                            },
    
}

