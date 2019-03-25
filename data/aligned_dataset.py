import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
# from data.base_dataset import BaseDataset
# from data.image_folder import make_dataset
from PIL import Image

IMG_EXTENSIONS = [
    'jpg', 'JPG', 'jpeg', 'JPEG',
    'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP',
]

class AlignedDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.imgPairFolder = os.path.join(opt.dataroot, opt.phase)
        self.imgPairPaths = []

        for root, _, fNames in sorted(os.walk(os.path.join(opt.dataroot, opt.phase))):
            for fName in fNames:
                if fName.split(sep='.')[-1] in IMG_EXTENSIONS:
                    self.imgPairPaths.append(os.path.join(root, fName))
        self.imgPairPaths = sorted(self.imgPairPaths)

        # torchvision.transforms 图像处理管道，通过transforms.Compose组合， 类似nn.Sequence
        # ToTensor 将 PIL 图像转化为Tensor
        # Normalize(mean, std) 对N个管道的图片实现均值(M1,...,Mn)和方差(S1,...Sn)的归一化

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgPairPath = self.imgPairPaths[index]

        imgPair = self.transform(
            Image.open(
                self.imgPairPaths[index]
            ).convert('RGB').resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        )

        w = int(imgPair.size(2) / 2)
        h = imgPair.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # sharp img
        sImg = imgPair[:, h_offset:h_offset + self.opt.fineSize,     w_offset:    w_offset + self.opt.fineSize]
        # blurred img
        bImg = imgPair[:, h_offset:h_offset + self.opt.fineSize, w + w_offset:w + w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(sImg.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            sImg = sImg.index_select(2, idx)  # sharp   Img
            bImg = bImg.index_select(2, idx)  # blurred Img

        return {
            'A': sImg,
            'B': bImg,
            'A_paths': imgPairPath,
            'B_paths': imgPairPath
        }

    def __len__(self):
        return len(self.imgPairPaths)

    def name(self):
        return 'AlignedDataset'
