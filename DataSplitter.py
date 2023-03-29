import numpy as np

class DataSplitter(object):
    '''
    shuffle the trajectories into subsets
    return the next set when queried
    if all the trajectories are emuerated, shuffle again  
    '''
    def __init__(self, trajlist, trajlenlist, framelist, framenum, shuffle=True):
        '''
        trajlist: the relative path of the trajectories [traj0, traj1, ...] 
        trajlenlist: the length of the trajectories [len0, len1, ...]
        framelist: the frames [[traj0_frame0, traj0_frame1, ...],
                               [traj1_frame0, traj1_frame1, ...],
                               ...]
        framenum: the framenum for each subset
        startendlist: start and end index for each trajectory, [[startind, endind], [startind, endind], ..]
                      this is used for loading simple modalities such as motion and IMU
        '''
        self.trajlist, self.trajlenlist, self.framelist = trajlist, trajlenlist, framelist
        self.framenum = framenum
        self.shuffle = shuffle
        self.trajnum = len(trajlist)
        self.totalframenum = sum(trajlenlist)

        self.trajinds = np.arange(self.trajnum, dtype=np.int32)
        self.curind = -1
        self.leftover = [] # [traj, framelist, startind]
        self.subtrajlist, self.subtrajlenlist, self.subframelist = [], [], []

    def add_traj(self, framecount, trajstr, trajlen, framelist, startind):
        self.subtrajlist.append(trajstr)
        if framecount + trajlen > self.framenum: # the leftover traj is still too long
            addnum = self.framenum - framecount
            self.subtrajlenlist.append(addnum)
            self.subframelist.append(framelist[:addnum])
            framecount += addnum
            self.leftover = [trajstr, framelist[addnum:], startind + addnum]
        else:
            self.subtrajlenlist.append(trajlen)
            self.subframelist.append(framelist)
            framecount += trajlen
            self.leftover = []
        return framecount

    def get_next_split(self):
        framecount = 0 
        self.subtrajlist, self.subtrajlenlist, self.subframelist = [], [], []

        # append the remaining traj from last time
        if len(self.leftover) > 0:
            trajstr = self.leftover[0]
            framelist = self.leftover[1]
            startind = self.leftover[2]
            trajlen = len(framelist)
            framecount = self.add_traj(framecount, trajstr, trajlen, framelist, startind)

        while framecount < self.framenum:
            self.curind = (self.curind + 1) % self.trajnum
            if self.curind == 0 and self.shuffle: # shuffle the trajectory 
                self.trajinds = np.random.permutation(self.trajnum)

            # add the current trajectory to the lists
            trajind = self.trajinds[self.curind]
            trajlen = self.trajlenlist[trajind]
            trajstr = self.trajlist[trajind]
            framelist = self.framelist[trajind]
            framecount = self.add_traj(framecount, trajstr, trajlen, framelist, 0)


        return self.subtrajlist, self.subtrajlenlist, self.subframelist, self.framenum

    def get_next_trajectory(self):
        self.curind = (self.curind + 1) % self.trajnum
        subtrajlist = [self.trajlist[self.curind]]
        subtrajlenlist = [self.trajlenlist[self.curind]]
        subframelist = [self.framelist[self.curind]]

        return subtrajlist, subtrajlenlist, subframelist, self.trajlenlist[self.curind]