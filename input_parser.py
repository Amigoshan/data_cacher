import os


def parse_inputfile(inputfile):
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
        traj, trajlen = line.split(' ')

        # Break path and rebuild it to avoid problems with Windows and Linux.
        traj = os.path.join(*traj.split("\\"))
        
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
