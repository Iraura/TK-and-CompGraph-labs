from PIL import Image
import numpy as np


class Colour:
    colour_array = [0, 0, 0]

    def __init__(self, colour_array):
        self.colour_array = colour_array


class Picture:
    picture_array = np.zeros((512, 512, 3), dtype=np.uint8)

    def __init__(self, h, w, d):
        self.picture_array = np.zeros((h, w, d), dtype=np.uint8)

    def create_from_array(self, array):
        self.picture_array = array

    def set_pixel(self, x, y, color: Colour):
        self.picture_array[int(x), int(y)] = color.colour_array

    def show_picture(self):
        img = Image.fromarray(self.picture_array, 'RGB')
        img.show()


def create_coloured_square(h, w, colour_array):
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


# exception if we go from last index to center
def line_builder_variant_1(x1, y1, x0, y0, pic: Picture, colour: Colour):
    t = 0.0
    while t < 1.0:
        x = x0 * (1.0 - t) + x1 * t
        y = y0 * (1.0 - t) + y1 * t
        pic.set_pixel(x, y, colour)
        t += 0.01


# if x1 = 0 then we will not get any image because of (x1 - x0) < 0
def line_builder_variant_2(x1, y1, x0, y0, pic: Picture, colour: Colour):
    x = x0
    while x <= x1:
        t = (x - x0) / (x1 - x0)
        y = y0 * (1.0 - t) + y1 * t
        pic.set_pixel(x, y, colour)
        x += 1


def line_builder_variant_3(x1, y1, x0, y0, pic: Picture, colour: Colour):
    sleep = False
    x = x0
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        sleep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    while x <= x1:
        t = (x - x0) / (x1 - x0)
        y = y0 * (1.0 - t) + y1 * t
        if sleep:
            pic.set_pixel(x, y, colour)
        else:
            pic.set_pixel(y, x, colour)
        x += 1


def task_1():
    create_coloured_square(512, 512, [0, 0, 0])
    create_coloured_square(512, 512, [255, 255, 255])
    create_coloured_square(512, 512, [255, 0, 0])
    create_square_with_diff_colours(512, 512)


if __name__ == '__main__':
    # task_1()
    pic = Picture(512, 512, 3)
    colour = Colour([255, 255, 255])
    line_builder_variant_3(511, 511, 256, 256, pic, colour)
    pic.show_picture()
