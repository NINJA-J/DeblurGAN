import torch.utils.data
from data.aligned_dataset import AlignedDataset

class CustomDatasetDataLoader:
    def name(self):
        return 'CustomDataSetDataLoader'

    def __init__(self, opt):
        print("Opt.nThreads = ", opt.nThreads)

        self.opt = opt
        self.dataSet = AlignedDataset(opt)
        self.dataLoader = torch.utils.data.DataLoader(
            self.dataSet,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self.dataLoader

    def __len__(self):
        return min(len(self.dataSet), self.opt.max_dataset_size)
