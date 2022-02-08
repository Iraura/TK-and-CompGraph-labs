from PIL import Image
import numpy as np


def task1(w, h):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.show()


class Color (object):

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


class Picture (object):

    def __init__(self, height, weight):
        self.height = height
        self.weight = weight



if __name__ == '__main__':
    task1(512, 512)
