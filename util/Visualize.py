import time
import matplotlib.pyplot as plt
import torch

import visdom

class Visualizer:
    def __init__(self, opt=None):
        self.vis = visdom.Visdom(env="Test")

        self.bImgPad = self.vis.images(torch.randn(64, 64), opts={'title':'blurred  Image'})
        self.rImgPad = self.vis.images(torch.randn(64, 64), opts={'title':'restored Image'})
        self.sImgPad = self.vis.images(torch.randn(64, 64), opts={'title':'sharp    Image'})


    def sendText(self, text):
        self.infoPad.text(0, 0, text)
        plt.pause(5)

if __name__=='__main__':
    visualizer = Visualizer()
    visualizer.sendText("This is a text")

    # time.sleep(20)
