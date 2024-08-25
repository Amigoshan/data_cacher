from os.path import isfile, join, isdir, split
from os import listdir

def read_datafile(inputfile):
    '''
    trajlist: [TRAJ0, TRAJ1, ...]
    trajlenlist: [TRAJLEN0, TRAJLEN1, ...]
    framelist: [[FRAMESTR0, FRAMESTR1, ...],[FRAMESTR_K, FRAMESTR_K+1, ...], ...]
    '''
    with open(inputfile,'r') as f:
        lines = f.readlines()
    trajlist, trajlenlist, framelist = [], [], []
    ind = 0
    while ind<len(lines):
        line = lines[ind].strip()
        try:
            traj, trajlen = line.split(' ')
        except:
            raise Exception("Datafile Error: line: {}, ind: {}".format(line, ind))

        # Break path and rebuild it to avoid problems with Windows and Linux.
        traj = join(*traj.split("\\"))
        
        trajlen = int(trajlen)
        trajlist.append(traj)
        trajlenlist.append(trajlen)
        ind += 1
        frames = []
        for k in range(trajlen):
            if ind>=len(lines):
                print("Datafile Error: {}, line {}...".format(inputfile, ind))
                raise Exception("Datafile Error: {}, line {}...".format(inputfile, ind))
            line = lines[ind].strip()
            frames.append(line)
            ind += 1
        framelist.append(frames)
    totalframenum = sum(trajlenlist)
    print('{}: Read {} trajectories, including {} frames'.format(inputfile, len(trajlist), totalframenum))
    return trajlist, trajlenlist, framelist, totalframenum

def write_datafile(datafile, trajstr, framelist):
    trajlen = len(framelist)
    datafile.write(trajstr)
    datafile.write(' ')
    datafile.write(str(trajlen))
    datafile.write('\n')
    for indstr in framelist:
        datafile.write(indstr)
        datafile.write('\n')

def enumerate_trajs(data_root_dir, env_folders = None, data_folders = ['Data_easy','Data_hard']):
    '''
    env_folders: ['AbandonedFactory', 'AbandonedCabel', ...], if None, all the envs in the data_root_dir will be returned
    Return a dict:
        res['env0']: ['env0/Data_easy/P000', 'env0/Data_easy/P001', ...], 
        res['env1']: ['env1/Data_easy/P000', 'env1/Data_easy/P001', ...], 
    '''
    if env_folders is None: # find all the env folders if not given
        env_folders = listdir(data_root_dir)   

    env_folders = [ee for ee in env_folders if isdir(join(data_root_dir, ee))]
    env_folders.sort()
    print('Detected envs {}'.format(env_folders))
    env_traj_dict = {}
    for env_folder in env_folders:
        env_dir = join(data_root_dir, env_folder)
        print('Working on env {}'.format(env_dir))

        for data_folder in data_folders:
            data_dir = join(env_dir, data_folder)
            if not isdir(data_dir):
                print('  !!data folder missing '+ data_dir)
                continue
            # print('    Opened data folder {}'.format(data_dir))

            trajfolders = listdir(data_dir)
            trajfolders = [ join(env_folder, data_folder, tf) for tf in trajfolders if tf[0]=='P' ]
            trajfolders.sort()

            if len(trajfolders) > 0:
                if env_folder not in env_traj_dict:
                    env_traj_dict[env_folder] = []

                env_traj_dict[env_folder].extend(trajfolders)
                print('    Found {} trajectories in env {}'.format(len(trajfolders), data_dir))
        print('---')
    return env_traj_dict

def enumerate_frames(modfolder, surfix = '.png'):
    '''
    Return a list of frame index in the modfolder
    '''
    files = listdir(modfolder)
    files = [ff.split('_')[0] for ff in files if ff.endswith(surfix)]
    files.sort()
    return files

def generate_datafile(datafilename, root_dir, env_list = None, check_modality = 'image_lcam_front', file_surfix = '.png'):
    '''
    root_dir: the root folder for all environments
    env_list: ['AbandonedFactory', 'AbandonedCabel', ...], if None, all the envs in the data_root_dir will be returned

    Output data file:
            * firstline: <env_root_dir>/<datafolder>/<trajfolder> <num_of_frames>
            * each line correspond to a file index: 000xxx
            * each line does not contain image suffix

    Example: generate_datafile("/home/amigo/tmp/test_datafile.txt", root_dir = '/home/amigo/tmp/test_root')
    '''
    # check and create output folders
    env_traj_dict = enumerate_trajs(root_dir, env_folders = env_list)

    outdatafile = open(datafilename, 'w')
    frame_count = 0
    for env_name in env_traj_dict:
        trajlist = env_traj_dict[env_name]
        for trajstr in trajlist:
            traj_dir = join(root_dir, trajstr)
            mod_dir = join(traj_dir, check_modality)
            assert isdir(mod_dir), "Error: folder not exist: {}".format(mod_dir)
            frames = enumerate_frames(mod_dir)

            write_datafile(outdatafile, trajstr, frames)
            frame_count += len(frames)

    print('')
    print('Saved {} frames from {} envs in {}'.format(frame_count, len(env_traj_dict), datafilename))
    outdatafile.close()

def remove_envs_from_datafile(datafilename, newdatafilename, env_list): 
    '''
    Example: remove_envs_from_datafile('./data/data_tartanairv2.txt', 
                                        './data/test_data_tartanairv2.txt', 
                                        ['GreatMarshExposure', 'RetroOfficeExposure'])
    '''
    trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafilename)
    outdatafile = open(newdatafilename, 'w')

    trajcount = 0
    for trajstr, frames in zip(trajlist, framelist):
        envstr = trajstr.split('/')[0]
        if envstr in env_list:
            trajcount += 1
            continue # do not write this to the new datafile

        write_datafile(outdatafile, trajstr, frames)

    print('Removed {} trajs from {} trajs'.format(trajcount, len(trajlist)))
    outdatafile.close()

def remove_traj_from_datafile(datafilename, newdatafilename, trajlist):
    '''
    trajlist: ['env0/Data_easy/P000', 'env1/Data_easy/P003', ...]
    '''
    trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafilename)
    outdatafile = open(newdatafilename, 'w')

    trajcount = 0
    for trajstr, frames in zip(trajlist, framelist):
        if trajstr in trajlist:
            trajcount += 1
            continue # do not write this to the new datafile

        write_datafile(outdatafile, trajstr, frames)

    print('Removed {} trajs from {} trajs'.format(trajcount, len(trajlist)))
    outdatafile.close()

def rename_envs_datafile(datafilename, newdatafilename, envdict):
    trajlist, _, framelist, _ = read_datafile(datafilename)
    outdatafile = open(newdatafilename, 'w')

    rename_count = 0
    for trajstr, frames in zip(trajlist, framelist):
        envstr = trajstr.split('/')[0]
        if envstr in envdict:
            newenvstr = envdict[envstr]
            newtrajstr = trajstr.replace(envstr, newenvstr)
            rename_count += 1
        else:
            newtrajstr = envstr
        write_datafile(outdatafile, newtrajstr, frames)

    print('Renamed {} trajs from {} trajs'.format(rename_count, len(trajlist)))
    outdatafile.close()
    
def generate_datafile(datafile_name, trajstrlist, frameslist):
    outdatafile = open(datafile_name, 'w')
    for trajstr, frames in zip(trajstrlist, frameslist):
        write_datafile(outdatafile, trajstr, frames)
    outdatafile.close()
    print('Generated datafile {} which contains {} trajs'.format(datafile_name, len(trajstrlist)))
    
def breakdown_trajectories(trajstrlist, frameslist, max_traj_len = 200):
    new_trajstr, new_framelist = [], []
    for trajstr, frames in zip(trajstrlist, frameslist):
        while len(frames) > max_traj_len:
            subframes = frames[:max_traj_len]
            frames = frames[max_traj_len:]
            new_trajstr.append(trajstr)
            new_framelist.append(subframes)
        if len(frames) > 0:
            new_trajstr.append(trajstr)
            new_framelist.append(frames)
    return new_trajstr, new_framelist

if __name__=="__main__":
    # generate_datafile("/home/amigo/tmp/test_datafile.txt", root_dir = '/home/amigo/tmp/test_root')
    # remove_envs_from_datafile('./data/data_tartanairv2.txt', './data/test_data_tartanairv2.txt', ['GreatMarshExposure', 'RetroOfficeExposure'])
    # from MapDict import mapdict
    # rename_envs_datafile('../data/tartanv2/tartan2_flow_front.txt', '../data/tartanv2_new/tartan2_flow_front.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_flow_front_train.txt', '../data/tartanv2_new/tartan2_flow_front_train.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_flow_front_test.txt', '../data/tartanv2_new/tartan2_flow_front_test.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_flow_front_more_train.txt', '../data/tartanv2_new/tartan2_flow_front_train_more.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_flow_front_more_test.txt', '../data/tartanv2_new/tartan2_flow_front_test_more.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_flow_front_more.txt', '../data/tartanv2_new/tartan2_flow_front_more.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_flow_front_more_train.txt', '../data/tartanv2_new/tartan2_flow_front_train_more.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_flow_front_more_test.txt', '../data/tartanv2_new/tartan2_flow_front_test_more.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_stereo_back.txt', '../data/tartanv2_new/tartan2_stereo_back.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_stereo_bottom.txt', '../data/tartanv2_new/tartan2_stereo_bottom.txt', mapdict)
    # rename_envs_datafile('../data/tartanv2/tartan2_stereo_front.txt', '../data/tartanv2_new/tartan2_stereo_front.txt', mapdict)

    inputfile = 'data/tartan_train.txt'
    outputfile = 'data/test_datafile.txt'
    trajlist, trajlenlist, framelist, totalframenum = read_datafile(inputfile)
    trajlist, framelist = breakdown_trajectories(trajlist, framelist)
    generate_datafile(outputfile, trajlist,framelist)
    read_datafile(outputfile)