import torch
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from .MultiDatasets import MultiDatasets


class MultiDatasetsIterable(IterableDataset):
    def __init__(self, multidatasets, fullbatch=True, notrepeat=False, maxrepeatnum=3):
        self.multidatasets = multidatasets
        self.fullbatch = fullbatch
        self.notrepeat = notrepeat
        self.maxrepeatnum = maxrepeatnum

    def __iter__(self):
        return self

    def __next__(self):
        sample = self.multidatasets.load_sample(fullbatch=self.fullbatch, notrepeat=self.notrepeat, maxrepeatnum=self.maxrepeatnum)
        return sample


class MultiDatasetsPL(pl.LightningDataModule):
    """Lightning DataModule that wraps MultiDatasets."""

    def __init__(self,
                 training_dataset_specfile,
                 testing_dataset_specfile,
                 platform,
                 training_batch,
                 testing_batch,
                 workernum=0,
                 shuffle=True,
                 verbose=False,
                 fullbatch=True,
                 notrepeat=False,
                 maxrepeatnum=3):
        super().__init__()
        self.training_dataset_specfile = training_dataset_specfile
        self.testing_dataset_specfile = testing_dataset_specfile
        self.platform = platform
        self.training_batch = training_batch
        self.testing_batch = testing_batch
        self.num_workers = workernum
        self.shuffle = shuffle
        self.verbose = verbose
        self.fullbatch = fullbatch
        self.notrepeat = notrepeat
        self.maxrepeatnum = maxrepeatnum

        self.training_multidatasets = None
        self.testing_multidatasets = None

    def setup(self, stage=None):
        if self.training_multidatasets is None:
            self.training_multidatasets = MultiDatasets(
                self.training_dataset_specfile,
                self.platform,
                batch=self.training_batch,
                workernum=self.num_workers,
                shuffle=self.shuffle,
                verbose=self.verbose,
            )
        if self.testing_multidatasets is None:
            self.testing_multidatasets = MultiDatasets(
                self.testing_dataset_specfile,
                self.platform,
                batch=self.testing_batch,
                workernum=self.num_workers,
                shuffle=False, # usually we don't shuffle the testing data
                verbose=self.verbose,
            )

    def train_dataloader(self):
        return MultiDatasetsIterable(
            self.training_multidatasets,
            fullbatch=self.fullbatch,
            notrepeat=self.notrepeat,
            maxrepeatnum=self.maxrepeatnum,
        )

    def test_dataloader(self):
        return MultiDatasetsIterable(
            self.testing_multidatasets,
            fullbatch=self.fullbatch,
            notrepeat=True,
            maxrepeatnum=self.maxrepeatnum,
        )

    def teardown(self, stage=None):
        if self.training_multidatasets is not None:
            self.training_multidatasets.stop_cachers()
        if self.testing_multidatasets is not None:
            self.testing_multidatasets.stop_cachers()

# test code for this wrapper
if __name__ == "__main__":
    import cv2
    from .example_multidatasets_tav2_events import vis_event_tensor
    multidatasets = MultiDatasetsPL(
        training_dataset_specfile='data_cacher/dataspec/test_events.yaml',
        testing_dataset_specfile='data_cacher/dataspec/test_events_h5.yaml',
        platform='local',
        training_batch=4,
        testing_batch=2,
        workernum=0,
        shuffle=True,
        verbose=True,
        fullbatch=False,
        notrepeat=False,
        maxrepeatnum=3,
    )

    multidatasets.setup()
    trainloader = multidatasets.train_dataloader()
    testloader = multidatasets.test_dataloader()

    for k in range(1000000):
        train_sample = next(iter(trainloader))
        test_sample = next(iter(testloader))
        # import ipdb;ipdb.set_trace()
            # visualize the data
        eventtensor = train_sample['events_left'] # the key is the same as the one defined in the dataspec file
        for i in range(eventtensor.shape[0]):
            vis_im = vis_event_tensor(eventtensor[i][0].numpy())
            cv2.imshow('train_event_tensor', vis_im)
            cv2.waitKey(1)

        eventtensor = test_sample['events_left'] # the key is the same as the one defined in the dataspec file
        for i in range(eventtensor.shape[0]):
            vis_im = vis_event_tensor(eventtensor[i][0].numpy())
            cv2.imshow('test_event_tensor', vis_im)
            cv2.waitKey(1)

        print("Train sample dataset name {}, test sample dataset name {}".format(train_sample['dataset_info']['dataset_name'], test_sample['dataset_info']['dataset_name']))    

    multidatasets.teardown()