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
        self.z_matrix = np.zeros((w, h))

    def create_from_array(self, array):
        self.picture_array = array

    def set_pixel(self, x, y, color: Colour):
        self.picture_array[int(x), int(y)] = color.colour_array

    def show_picture(self):
        img = Image.fromarray(self.picture_array, 'RGB')
        img.show()

    def clear(self):
        self.picture_array[0:self.h, 0:self.w] = self.default_colour


# считывание вершин с obj файла
def create_string_pixel_matrix_from_obj_file(filename):
    # открываем obj файл
    f = open(filename)
    s = f.read().split('\n')
    source = list()

    # считываем вершины, описанные структурой v x1 y1
    for i in s:
        if len(i) != 0 and i[0] == 'v' and i[1] == ' ':
            source.append(i)
    workspace = list()
    for i in source:
        workspace.append(i.split())
    return workspace


# считывание полигонов с obj файла
def create_string_polygon_matrix_from_obj_file(filename):
    # открываем obj файл
    f = open(filename)
    s = f.read().split('\n')
    source = list()

    # считываем вершины полигонов, описанные структурой f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
    for i in s:
        if len(i) != 0 and i[0] == 'f' and i[1] == ' ':
            source.append(i)
    workspace = list()
    for i in source:
        workspace.append(i.split())

    result = list()

    # записть только первых значений из v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
    for i in workspace:
        result.append([i[1].split('/')[0], i[2].split('/')[0], i[3].split('/')[0]])

    return result


# функция создания изображения для 1-го задания
def create_coloured_square(h, w, colour_array):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:512, 0:512] = colour_array
    img = Image.fromarray(data, 'RGB')
    img.show()


# функция создания изображения с градиентом
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


# функция отрисовки звезды из линий
def star_builder(variant, delta_t, pic_star, pic_colour):
    for i in range(13):
        a = (2 * np.pi * i) / 13
        variant(100 + 95 * np.cos(a), 100 + 95 * np.sin(a), 100, 100, pic_star, pic_colour, delta_t)
    pic_star.show_picture()


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


# 3-ий вариант отрисовки линий
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


# 4-ый вариант отрисовки линий (алгоритм Брезенхема)
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
    # создание черного изображения
    create_coloured_square(512, 512, [0, 0, 0])

    # создание белого изображения
    create_coloured_square(512, 512, [255, 255, 255])

    # создание красного изображения
    create_coloured_square(512, 512, [255, 0, 0])

    # создание градиента
    create_square_with_diff_colours(512, 512)


def task_3():
    default_picture_colour = Colour([255, 255, 255])
    colour = Colour([0, 0, 0])
    pic = Picture(400, 400, default_picture_colour)

    # task №3
    # построение звезды 1-ым способом при дельта = 0.01
    # для последующих рисунков очищаем изображение, заполняя его дефолтным цветом
    delta_t = 0.01
    star_builder(line_builder_variant_1, delta_t, pic, colour)
    pic.clear()

    # построение звезды 1-ым способом при дельта = 0.1
    delta_t = 0.1
    star_builder(line_builder_variant_1, delta_t, pic, colour)
    pic.clear()

    # построение звезды 2-ым способом
    star_builder(line_builder_variant_2, delta_t, pic, colour)
    pic.clear()

    # построение звезды 3-им способом
    star_builder(line_builder_variant_3, delta_t, pic, colour)
    pic.clear()

    # построение звезды 4-ым способом
    star_builder(line_builder_variant_4, delta_t, pic, colour)
    pic.clear()


def task_5_6(multy, sum):
    default_picture_colour = Colour([0, 0, 0])  # цвет фона
    colour = Colour([255, 255, 255])  # цвет рисунка
    pic = Picture(1000, 1000, default_picture_colour)

    # массив вершин
    top_array = create_string_pixel_matrix_from_obj_file('StormTrooper.obj')

    # массив полигонов
    polygon_map = create_string_polygon_matrix_from_obj_file('StormTrooper.obj')

    # отрисовка вершин изображения 1-ым способом отрисовки
    for i in range(1, len(top_array)):
        line_builder_variant_1(float(top_array[i][1]) * multy + sum, float(top_array[i][2]) * multy + sum,
                               float(top_array[i - 1][1]) * multy + sum,
                               float(top_array[i - 1][2]) * multy + sum,
                               pic, colour, 1000)

    # отрисовка полигонов изображения
    for i in polygon_map:
        i_0 = int(i[0]) if int(i[0]) > 0 else len(top_array) - 1 + int(i[0])  # первая вершина полигона
        i_1 = int(i[1]) if int(i[1]) > 0 else len(top_array) - 1 + int(i[1])  # вторая вершина полигона
        i_2 = int(i[2]) if int(i[2]) > 0 else len(top_array) - 1 + int(i[2])  # третья вершина полигона

        # первое ребро полигона (вершины 1 и 2)
        line_builder_variant_4(float(top_array[i_0 - 1][1]) * multy + sum, float(top_array[i_0 - 1][2]) * multy + sum,
                               float(top_array[i_1 - 1][1]) * multy + sum + 1,
                               float(top_array[i_1 - 1][2]) * multy + sum + 1,
                               pic, colour, 1000)

        # второе ребро (вершины 1 и 3)
        line_builder_variant_4(float(top_array[i_0 - 1][1]) * multy + sum, float(top_array[i_0 - 1][2]) * multy + sum,
                               float(top_array[i_2 - 1][1]) * multy + sum + 1,
                               float(top_array[i_2 - 1][2]) * multy + sum + 1,
                               pic, colour, 1000)

        # третье ребро (вершины 2 и 3)
        line_builder_variant_4(float(top_array[i_1 - 1][1]) * multy + sum, float(top_array[i_1 - 1][2]) * multy + sum,
                               float(top_array[i_2 - 1][1]) * multy + sum + 1,
                               float(top_array[i_2 - 1][2]) * multy + sum + 1,
                               pic, colour, 1000)

        x0 = float(top_array[i_0 - 1][1]) * multy + sum
        y0 = float(top_array[i_0 - 1][2]) * multy + sum
        z0 = float(top_array[i_0 - 1][3]) * multy + sum
        x1 = float(top_array[i_1 - 1][1]) * multy + sum + 1
        y1 = float(top_array[i_1 - 1][2]) * multy + sum + 1
        z1 = float(top_array[i_1 - 1][3]) * multy + sum
        x2 = float(top_array[i_2 - 1][1]) * multy + sum
        y2 = float(top_array[i_2 - 1][2]) * multy + sum
        z2 = float(top_array[i_2 - 1][3]) * multy + sum
        task_9_print_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, pic)
    pic.show_picture()


def task_8_bara_sentral_coords(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    print(lambda0 + lambda1 + lambda2)
    return np.array([lambda0, lambda1, lambda2])


def task_9_print_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, pic: Picture):
    default_picture_colour = Colour([0, 0, 0])  # цвет фона
    pic = Picture(1000, 1000, default_picture_colour)

    xmin = float(min(x0, x1, x2))
    ymin = float(min(y0, y1, y2))
    xmax = float(max(x0, x1, x2))
    ymax = float(max(y0, y1, y2))

    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > pic.h): xmax = pic.h
    if (ymax > pic.w): ymax = pic.w

    n = np.cross([x1 - x0, y1 - y0, z1 - z0],
                 [x1 - x2, y1 - y2, z1 - z2])

    l = [0, 0, 1]

    cos_alpha = (n @ l) / np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    if cos_alpha > 0:
        return

    color = Colour([255 * abs(cos_alpha), 0, 0])

    for x in range(round(xmin), round(xmax)):
        for y in range(round(ymin), round(ymax)):
            lambdas = task_8_bara_sentral_coords(x, y, x0, y0, x1, y1, x2, y2)
            if np.all(lambdas >= 0):
                z_val = lambdas[0] * z0 + lambdas[1] * z1 + lambdas[2] * z2
                if z_val > pic.z_matrix[x][y]:
                    pic.z_matrix[x][y] = z_val
                    pic.set_pixel(x, y, color)


if __name__ == '__main__':
    # task_1()
    # task_3()
    task_5_6(250, 500)
