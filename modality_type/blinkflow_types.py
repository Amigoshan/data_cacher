from os.path import join
import numpy as np
import cv2
from .ModBase import SimpleModBase, FrameModBase, register, TYPEDICT

class RGBModBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist)
        listlen = len(datashapelist) # this is usually one
        self.data_types = []
        for k in range(listlen):
            self.data_shapes[k] = (3,) + tuple(self.data_shapes[k] )
            self.data_types.append(np.float32)

    def load_frame(self, trajdir, filenamelist):
        # read image
        imglist = []
        for filename in filenamelist:
            if filename.endswith('.npy'):
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

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + '.npy')]

class FlowModBase(FrameModBase):
    def __init__(self, datashapes):
        super().__init__(datashapes)
        # we assume that the flow might return flow or (flow, mask)
        # we also assume that the flow will always be returned, the mask is optional
        self.listlen = len(datashapes) # this is usually one
        self.data_shapes[0] = (2,) + tuple(self.data_shapes[0]) # add one dim to the 
        self.data_types = [np.float32, np.uint8] # for flow and mask

        self.folder_name = "" # to be filled in the derived class
        self.file_suffix = "" # to be filled in the derived class

    def load_frame(self, trajdir, filenamelist):
        flow_32bit = np.load(join(trajdir, filenamelist[0]))
        assert flow_32bit is not None, "Error loading flow {}".format(join(trajdir, filenamelist[0]))
        flow = flow_32bit[:,:,:2]

        if self.listlen == 1:
            return [flow]

        mask8 = flow_32bit[:,:,2].astype(np.uint8)
        return [flow, mask8]

    def resize_data(self, flowmasklist):
        # resize image
        flow = flowmasklist[0]
        target_h, target_w = self.data_shapes[0][1], self.data_shapes[0][2]
        (h, w, _) = flow.shape
        if h != target_h or w != target_w:
            scale_w, scale_h = float(target_w) / w, float(target_h) / h
            flow = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR )
            flow[0,:,:] = flow[0,:,:] * scale_w
            flow[1,:,:] = flow[1,:,:] * scale_h
        if self.listlen == 1: 
            return [flow]
        
        mask = flowmasklist[1]
        target_h, target_w = self.data_shapes[1]
        (h, w) = mask.shape
        if h != target_h or w != target_w:
            scale_w, scale_h = float(target_w) / w, float(target_h) / h
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST )
        return [flow, mask]

    def transpose(self, flowlist):
        reslist = []
        for img in flowlist:
            if len(img.shape) == 3:
                reslist.append(img.transpose(2,0,1))
            else:
                reslist.append(img)
        return reslist

    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + self.file_suffix + '.npy')]

class EventsBase(FrameModBase):
    def __init__(self, datashapelist):
        super().__init__(datashapelist) # point dimention, e.g. 3 for tartanvo, 6 if rgb is included
        lenlist = len(datashapelist)
        self.data_types = []
        for k in range(lenlist):
            self.data_types.append(np.float32)

        self.folder_name = "" # to be filled in the derived class
        self.file_suffix = "_event_tensor" # to be filled in the derived class

    def load_frame(self, trajdir, filenamelist):
        eventtensorlist = []
        for filename in filenamelist:
            if filename.endswith('.npz'):
                eventtensor = np.load(join(trajdir,filename))['event_tensor']
            else:
                raise NotImplementedError
            eventtensorlist.append(eventtensor)
        return eventtensorlist
    
    def transpose(self, events):
        return events

    def resize_data(self, events):
        return events
    
    def framestr2filename(self, framestr):
        '''
        This is very dataset specific
        Basically it handles how each dataset naming the frames and organizing the data
        '''
        return [join(self.folder_name, framestr + self.file_suffix + '.npz')]

@register(TYPEDICT)
class blinkflow_cam(RGBModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "hdr"

@register(TYPEDICT)
class blinkflow_flow(FlowModBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "forward_flow"

@register(TYPEDICT)
class blinkflow_events(EventsBase):
    def __init__(self, datashape):
        super().__init__(datashape)
        self.folder_name = "event_tensors"