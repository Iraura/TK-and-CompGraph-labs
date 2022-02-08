from PIL import Image
import numpy as np


def task1(w, h):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.show()


if __name__ == '__main__':
    task1(512, 512)
