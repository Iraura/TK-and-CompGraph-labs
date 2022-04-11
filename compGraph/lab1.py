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
        self.z_matrix = np.zeros((h, w))

    def create_from_array(self, array):
        self.picture_array = array

    def set_pixel(self, x, y, color: Colour):
        if self.w > x > 0 and self.h > y > 0:
            self.picture_array[int(y), int(x)] = color.colour_array

    def show_picture(self):
        img = Image.fromarray(self.picture_array, 'RGB')
        img.show()

    def clear(self):
        self.picture_array[0:self.h, 0:self.w] = self.default_colour


# считывание вершин с obj файла
def read_pixel_matrix_from_file(filename, letter1, letter2):
    # открываем obj файл
    with open(filename, 'r') as f:
        s = f.read().split('\n')
    source = list()

    # считываем вершины, описанные структурой v x1 y1
    for i in s:
        if len(i) != 0 and i[0] == letter1 and i[1] == letter2:
            source.append(i)
    workspace = list()
    for i in source:
        e = i.split()
        del e[0]
        result = [float(item) for item in e]
        workspace.append(result)
    f.close()
    return workspace


# считывание полигонов с obj файла
def read_polygon_matrix_from_file(filename):
    # открываем obj файл
    with open(filename, 'r') as f:
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
        result.append([int(i[1].split('/')[0]), int(i[2].split('/')[0]), int(i[3].split('/')[0])])
    f.close()
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
def line_v_1(x1, y1, x0, y0, pic: Picture, colour: Colour, delta_t=0.1):
    t = 0.0
    while t < 1.0:
        x = x0 * (1.0 - t) + x1 * t
        y = y0 * (1.0 - t) + y1 * t
        pic.set_pixel(x, y, colour)
        t += delta_t


# if x1 = 0 then we will not get any image because of (x1 - x0) < 0
def line_v_2(x1, y1, x0, y0, pic: Picture, colour: Colour, delta_t):
    x = x0
    while x <= x1:
        t = (x - x0) / (x1 - x0)
        y = y0 * (1.0 - t) + y1 * t
        pic.set_pixel(x, y, colour)
        x += 1


# 3-ий вариант отрисовки линий
def line_v_3(x1, y1, x0, y0, pic: Picture, colour: Colour, delta_t=0.0):
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
def line_v_4(x1, y1, x0, y0, pic: Picture, colour: Colour, delts_t=0.0):
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
    star_builder(line_v_1, delta_t, pic, colour)
    pic.clear()

    # построение звезды 1-ым способом при дельта = 0.1
    delta_t = 0.1
    star_builder(line_v_1, delta_t, pic, colour)
    pic.clear()

    # построение звезды 2-ым способом
    star_builder(line_v_2, delta_t, pic, colour)
    pic.clear()

    # построение звезды 3-им способом
    star_builder(line_v_3, delta_t, pic, colour)
    pic.clear()

    # построение звезды 4-ым способом
    star_builder(line_v_4, delta_t, pic, colour)
    pic.clear()


def task_5_6(multy, sum):
    default_picture_colour = Colour([0, 0, 0])  # цвет фона
    pic = Picture(1000, 1000, default_picture_colour)

    # массив вершин
    top_array = read_pixel_matrix_from_file('rabbit.obj', "v", " ")
    normals = read_pixel_matrix_from_file("rabbit.obj", "v", "n")
    # массив полигонов
    polygon_map = read_polygon_matrix_from_file('rabbit.obj')

    # # отрисовка вершин изображения 1-ым способом отрисовки
    # for i in range(1, len(top_array)):
    #     line_v_1(top_array[i][0] * multy + sum, top_array[i][1] * multy + sum,
    #                            top_array[i - 1][0] * multy + sum,
    #                            top_array[i - 1][1] * multy + sum,
    #                            pic, colour, 1000)

    # отрисовка полигонов изображения
    index = 0
    for i in polygon_map:
        i_0 = i[0] if i[0] > 0 else len(top_array) - 1 + i[0]  # первая вершина полигона
        i_1 = i[1] if i[1] > 0 else len(top_array) - 1 + i[1]  # вторая вершина полигона
        i_2 = i[2] if i[2] > 0 else len(top_array) - 1 + i[2]  # третья вершина полигона

        # # первое ребро полигона (вершины 1 и 2)
        # line_v_4(top_array[i_0 - 1][0] * multy + sum, top_array[i_0 - 1][1] * multy + sum,
        #                        top_array[i_1 - 1][0] * multy + sum + 1,
        #                        top_array[i_1 - 1][1] * multy + sum + 1,
        #                        pic, colour, 1000)
        #
        # # второе ребро (вершины 1 и 3)
        # line_v_4(top_array[i_0 - 1][0] * multy + sum, top_array[i_0 - 1][1] * multy + sum,
        #                        top_array[i_2 - 1][0] * multy + sum + 1,
        #                        top_array[i_2 - 1][1] * multy + sum + 1,
        #                        pic, colour, 1000)
        #
        # # третье ребро (вершины 2 и 3)
        # line_v_4(top_array[i_1 - 1][0] * multy + sum, top_array[i_1 - 1][1] * multy + sum,
        #                        top_array[i_2 - 1][0] * multy + sum + 1,
        #                        top_array[i_2 - 1][1] * multy + sum + 1,
        #                        pic, colour, 1000)

        x0_y0_z0 = task_17(multilizate_coords(top_array[i_0 - 1], multy, sum, pic))
        x1_y1_z1 = task_17(multilizate_coords(top_array[i_1 - 1], multy, sum, pic))
        x2_y2_z2 = task_17(multilizate_coords(top_array[i_2 - 1], multy, sum, pic))

        x0 = x0_y0_z0[0]
        y0 = x0_y0_z0[1]
        z0 = x0_y0_z0[2]
        x1 = x1_y1_z1[0]
        y1 = x1_y1_z1[1]
        z1 = x1_y1_z1[2]
        x2 = x2_y2_z2[0]
        y2 = x2_y2_z2[1]
        z2 = x2_y2_z2[2]

        # x0 = top_array[i_0 - 1][0] * multy + sum
        # y0 = top_array[i_0 - 1][1] * multy + sum
        # z0 = top_array[i_0 - 1][2] * multy + sum
        # x1 = top_array[i_1 - 1][0] * multy + sum + 1
        # y1 = top_array[i_1 - 1][1] * multy + sum + 1
        # z1 = top_array[i_1 - 1][2] * multy + sum
        # x2 = top_array[i_2 - 1][0] * multy + sum
        # y2 = top_array[i_2 - 1][1] * multy + sum
        # z2 = top_array[i_2 - 1][2] * multy + sum

        task_9_print_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, pic, normals, index)
        index += 1

        # task_9_print_triangle(y0, x0, z0, y1, x1, z1, y2, x2, z2, pic)
    pic.show_picture()


def multilizate_coords(top_array, ax, ay, pic: Picture, tx=0.02, ty=-0.025, tz=0.5):
    u0 = pic.w // 2
    v0 = pic.h // 2

    result = top_array.copy()
    x_shift = result[0] + tx
    y_shift = result[1] + ty
    z_shift = result[2] + tz
    result[0] = x_shift * ax + u0 * z_shift
    result[1] = y_shift * ay + z_shift * v0
    result[2] = z_shift
    return result


def task_8_bara_sentral_coords(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    return np.array([lambda0, lambda1, lambda2])


def task_9_print_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, pic: Picture, normals, index):
    x_min = float(min(x0, x1, x2))
    y_min = float(min(y0, y1, y2))
    x_max = float(max(x0, x1, x2))
    y_max = float(max(y0, y1, y2))

    # if (x_min < 0): x_min = 0
    # if (y_min < 0): y_min = 0
    # if (x_max > pic.h): x_max = pic.h
    # if (y_max > pic.w): y_max = pic.w

    n = np.cross([x1 - x0, y1 - y0, z1 - z0],
                 [x1 - x2, y1 - y2, z1 - z2])

    v = [0, 0, 1]

    l0 = get_l(normals[index][0], v)
    l1 = get_l(normals[index][1], v)
    l2 = get_l(normals[index][2], v)

    cos_alpha = (n @ v) / np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    if cos_alpha > 0:
        return

    color = Colour([255 * abs(cos_alpha), 0, 0])

    for x in range(round(x_min), round(x_max)):
        for y in range(round(y_min), round(y_max)):
            lambdas = task_8_bara_sentral_coords(x, y, x0, y0, x1, y1, x2, y2)
            brightness_value = 255 * (lambdas[0] * l0 + lambdas[1] * l1 + lambdas[2] * l2)
            color = Colour([brightness_value, 0, 0])
            if np.all(lambdas >= 0):
                z_val = lambdas[0] * z0 + lambdas[1] * z1 + lambdas[2] * z2
                if pic.h > x > 0 and pic.w > y > 0:
                    if z_val > pic.z_matrix[x][y]:
                        pic.z_matrix[x][y] = z_val
                        pic.set_pixel(x, y, color)
                else:
                    continue


def task_17(points):
    alpha = 0 * 180 / np.pi
    betta = 0.1 * 180 / np.pi
    gamma = 0 * 180 / np.pi

    cos_alpha = np.cos(alpha)
    cos_betta = np.cos(betta)
    cos_gamma = np.cos(gamma)
    sin_alpha = np.sin(alpha)
    sin_betta = np.sin(betta)
    sin_gamma = np.sin(gamma)

    rotate_x_matrix = np.array([[1, 0, 0],
                                [0, cos_alpha, sin_alpha],
                                [0, -sin_alpha, cos_alpha]])

    rotate_y_matrix = np.array([[cos_betta, 0, sin_betta],
                                [0, 1, 0],
                                [-sin_betta, 0, cos_betta]])

    rotate_z_matrix = np.array([[cos_gamma, sin_gamma, 0],
                                [-sin_gamma, cos_gamma, 0],
                                [0, 0, 1]])

    first_matmul_xy = np.matmul(rotate_x_matrix, rotate_y_matrix)
    R_matrix = np.matmul(first_matmul_xy, rotate_z_matrix)

    rotated_points = np.matmul(R_matrix, points)
    return rotated_points.T.tolist()


def mult_vectors(vector1, vector2):
    return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]


def get_vector_length(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def get_l(n, l_vector):
    return mult_vectors(n, l_vector) / (get_vector_length(n) * get_vector_length(l_vector))


if __name__ == '__main__':
    # task_1()
    # task_3()
    task_5_6(7500, 7500)  # trooper
    # task_5_6(5, 500) # fox
    # task_5_6(100, 100)
