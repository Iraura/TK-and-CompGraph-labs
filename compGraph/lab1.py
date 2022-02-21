from PIL import Image
import numpy as np


# task №2
class Colour:
    colour_array = [0, 0, 0]

    def __init__(self, colour_array):
        self.colour_array = colour_array


class Picture:
    h = 512
    w = 512
    picture_array = np.zeros((h, w, 3), dtype=np.uint8)
    default_colour = [0, 0, 0]

    def __init__(self, h, w, col: Colour):
        self.h = h
        self.w = w
        self.picture_array = np.zeros((h, w, 3), dtype=np.uint8)
        self.default_colour = col.colour_array
        self.picture_array[0:h, 0:w] = col.colour_array

    def create_from_array(self, array):
        self.picture_array = array

    def set_pixel(self, x, y, color: Colour):
        self.picture_array[int(x), int(y)] = color.colour_array

    def show_picture(self):
        img = Image.fromarray(self.picture_array, 'RGB')
        img.show()

    def clear(self):
        self.picture_array[0:self.h, 0:self.w] = self.default_colour


def create_string_matrix_from_obj_file(filename):
    f = open(filename)
    s = f.read().split('\n')
    source = list()
    for i in s:
        if len(i) != 0 and i[0] == 'v' and i[1] == ' ':
            source.append(i)
    workspace = list()
    for i in source:
        workspace.append(i.split())
    return workspace


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


def star_builder(variant, delta_t, pic_star, pic_colour):
    for i in range(13):
        a = (2 * np.pi * i) / 13
        variant(100 + 95 * np.cos(a), 100 + 95 * np.sin(a), 100, 100, pic_star, pic_colour, delta_t)
    pic.show_picture()


# exception if we go from last index to center
def line_builder_variant_1(x1, y1, x0, y0, pic: Picture, colour: Colour, delta_t=0.1):
    t = 0.0
    while t < 1.0:
        x = x0 * (1.0 - t) + x1 * t
        y = y0 * (1.0 - t) + y1 * t
        pic.set_pixel(x, y, colour)
        t += delta_t


# if x1 = 0 then we will not get any image because of (x1 - x0) < 0
def line_builder_variant_2(x1, y1, x0, y0, pic: Picture, colour: Colour, delta_t):
    x = x0
    while x <= x1:
        t = (x - x0) / (x1 - x0)
        y = y0 * (1.0 - t) + y1 * t
        pic.set_pixel(x, y, colour)
        x += 1


def line_builder_variant_3(x1, y1, x0, y0, pic: Picture, colour: Colour, delta_t=0.0):
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


def line_builder_variant_4(x1, y1, x0, y0, pic: Picture, colour: Colour, delts_t=0.0):
    sleep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        sleep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    d_x = x1 - x0
    d_y = y1 - y0
    d_error = abs(d_y / float(d_x))
    error = 0.
    y = y0

    for x in range(int(x0), int(x1) + 1):
        if sleep:
            pic.set_pixel(y, x, colour)
        else:
            pic.set_pixel(x, y, colour)
        error += d_error
        if error > 0.5:
            if y1 > y0:
                y += 1
            else:
                y -= 1
            error -= 1.


def task_1():
    create_coloured_square(512, 512, [0, 0, 0])
    create_coloured_square(512, 512, [255, 255, 255])
    create_coloured_square(512, 512, [255, 0, 0])
    create_square_with_diff_colours(512, 512)


def task_3():
    default_picture_colour = Colour([255, 255, 255])
    colour = Colour([0, 0, 0])
    pic = Picture(400, 400, default_picture_colour)

    # task №3
    delta_t = 0.01
    star_builder(line_builder_variant_1, delta_t, pic, colour)
    pic.clear()

    delta_t = 0.1
    star_builder(line_builder_variant_1, delta_t, pic, colour)
    pic.clear()

    star_builder(line_builder_variant_2, delta_t, pic, colour)
    pic.clear()

    star_builder(line_builder_variant_3, delta_t, pic, colour)
    pic.clear()

    star_builder(line_builder_variant_4, delta_t, pic, colour)
    pic.clear()


if __name__ == '__main__':
    default_picture_colour = Colour([255, 255, 255])
    colour = Colour([0, 0, 0])
    pic = Picture(1000, 1000, default_picture_colour)
    from_file_list = create_string_matrix_from_obj_file('StormTrooper.obj')
    for i in range(1, len(from_file_list)):
        x1 = float(from_file_list[i][1])
        line_builder_variant_1(float(from_file_list[i][1]), float(from_file_list[i][2]),
                               float(from_file_list[i - 1][1]), float(from_file_list[i - 1][2]), pic, colour, 1000)
    pic.show_picture()
