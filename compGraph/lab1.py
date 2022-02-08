from PIL import Image
import numpy as np


def create_coloured_square(w, h, colour_array):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:512, 0:512] = colour_array
    img = Image.fromarray(data, 'RGB')
    img.show()


def create_square_with_diff_colours(w, h):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    i = 0
    while i < w:
        data[i, 0:h] = [i % 256, 0, 0]
        i = i + 1
    j = 0
    while j < h:
        data[0, j:w] = [0, j % 256, 0]
        j = j + 1
    img = Image.fromarray(data, 'RGB')
    img.show()


def task_1():
    create_coloured_square(512, 512, [0, 0, 0])
    create_coloured_square(512, 512, [255, 255, 255])
    create_coloured_square(512, 512, [255, 0, 0])
    create_square_with_diff_colours(512, 512)


if __name__ == '__main__':
    task_1()
