# -*- coding:utf-8 -*-
import numpy as np
import cv2
import copy


class OneLine(object):

    def __init__(self, row, col):
        '''
        map 是原地图 0表示位置不可用，1表示可用
        status 表示已经访问过的点，1表示未访问，0表示已访问
        :param row: 行
        :param col: 列
        '''
        self.map = np.zeros((row, col))
        self.status = np.ones((row, col))
        self.init = None

    def set_position_available(self, x, y):
        self.map[x][y] = 1

    def set_map(self, map):
        self.map = map

    def set_init_point(self, point):
        self.init = point
        self.status[point[0], point[1]] = 0
        self.state = self.map * self.status

    def is_finished(self, state):
        if (np.sum(state) == 0):
            return True
        else:
            return False

    def move_to(self, map, status, curr_point, op_code):
        next_status = copy.copy(status)

        if (op_code == 'r'):  # 右划
            next_point = [curr_point[0], curr_point[1] + 1]
        elif (op_code == 'l'):  # 左划
            next_point = [curr_point[0], curr_point[1] - 1]
        elif (op_code == 'u'):  # 上划
            next_point = [curr_point[0] - 1, curr_point[1]]
        elif (op_code == 'd'):  # 下划
            next_point = [curr_point[0] + 1, curr_point[1]]
        else:
            next_point = curr_point

        next_status[next_point[0], next_point[1]] = 0
        next_state = map * next_status
        return {'map': map, 'status': next_status, 'state': next_state, 'curr_point': next_point}

    def get_available_operation(self, state, curr_point):
        op_list = []
        dim = state.shape
        if (curr_point[0] > 0 and state[curr_point[0] - 1, curr_point[1]] == 1):
            op_list.append('u')
        if (curr_point[0] < dim[0] - 1 and state[curr_point[0] + 1, curr_point[1]] == 1):
            op_list.append('d')
        if (curr_point[1] > 0 and state[curr_point[0], curr_point[1] - 1] == 1):
            op_list.append('l')
        if (curr_point[1] < dim[1] - 1 and state[curr_point[0], curr_point[1] + 1] == 1):
            op_list.append('r')
        return op_list

    def search_path(self, map, status, state, curr_point, path):
        # 如果找到解，返回
        if (self.is_finished(state)):
            return path
        # 否则进行尝试
        available_op = self.get_available_operation(state, curr_point)
        # 如果当前没走到死路
        while (len(available_op) != 0):
            op = available_op.pop()
            mov = self.move_to(map, status, curr_point, op)
            path.append(op)
            p = self.search_path(mov['map'], mov['status'], mov['state'], mov['curr_point'], path)
            if (p != False):
                return p
        path.pop()
        return False

    def start_find(self):
        return self.search_path(self.map, self.status, self.state, self.init, [])


class PictureOcr(object):
    def equ(self, rgb1, rgb2):
        if (np.linalg.norm(rgb1 - rgb2) < 3):
            return True
        return False

    def __init__(self, pic_path):
        self.block_color = [209, 209, 209]
        self.background_color = [248, 248, 248]
        self.start_y = 400  # 竖着的
        self.end_y = 1600
        self.start_x = 40
        self.end_x = 1050
        self.pic_path = pic_path
        self.image = cv2.imread(self.pic_path)

        self.left = None
        self.right = None
        self.top = None
        self.bottom = None
        self.dims = None
        self.init_point = None
        self.block_width = None
        self.block_l = 0
        self.block_r = 0

    def get_pos(self, pos):
        if (pos == 'top'):
            for y in range(self.start_y, self.end_y):
                for x in range(self.start_x, self.end_x):
                    if (self.equ(self.image[y][x], self.block_color)):
                        return y
        if (pos == 'bottom'):
            for y in range(self.end_y, self.start_y, -1):
                for x in range(self.start_x, self.end_x, 50):
                    if (self.equ(self.image[y][x], self.block_color)):
                        return y
        if (pos == 'left'):
            for x in range(self.start_x, self.end_x):
                for y in range(self.start_y, self.end_y, 50):
                    if (self.equ(self.image[y][x], self.block_color)):
                        return x
        if (pos == 'right'):
            for x in range(self.end_x, self.start_x, -1):
                for y in range(self.start_y, self.end_y, 50):
                    if (self.equ(self.image[y][x], self.block_color)):
                        return x

    def init_model(self):
        self.top = self.get_pos('top')
        self.bottom = self.get_pos('bottom')
        self.left = self.get_pos('left')
        self.right = self.get_pos('right')

        block_flag = False
        for x in range(self.left, self.right):
            if (not block_flag and self.equ(self.image[self.top + 40][x], self.block_color)):
                self.block_l = x
                block_flag = True
                continue
            if (block_flag and self.equ(self.image[self.top + 40][x], self.background_color)):
                self.block_r = x
                break
        block_width = self.block_r - self.block_l
        print(self.block_r, self.block_l)
        print(block_width)
        colomn = int((self.right - self.left) / block_width)
        row = int((self.bottom - self.top) / block_width)

        revised_block_width = int((self.right + self.bottom - self.left - self.top) / (colomn + row))

        # 寻找初始点
        init_flag = False
        for i in range(row):
            if (init_flag):
                break
            for j in range(colomn):
                point = self.image[int(self.top + (i + 0.5) * revised_block_width)][
                    int(self.left + (j + 0.5) * revised_block_width)]
                if (not (self.equ(point, self.background_color) or self.equ(point, self.block_color))):
                    init_flag = True
                    self.init_point = [i, j]

        self.dims = [row, colomn]
        self.block_width = revised_block_width

    def get_block_position(self, i, j):
        return [int(self.top + (i + 0.5) * self.block_width), int(self.left + (j + 0.5) * self.block_width)]

    def get_block_position_color(self, i, j):
        return self.image[int(self.top + (i + 0.5) * self.block_width), int(self.left + (j + 0.5) * self.block_width)]

    def get_matrix(self):
        map = np.zeros(self.dims)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if (not self.equ(self.get_block_position_color(i, j), self.background_color)):
                    map[i][j] = 1
        return map

    def translate_op(self, op):
        points = []
        curr_point = self.init_point
        for o in op:
            if (o == 'l'):
                curr_point[1] -= 1
            if (o == 'r'):
                curr_point[1] += 1
            if (o == 'u'):
                curr_point[0] -= 1
            if (o == 'd'):
                curr_point[0] += 1
            points.append(self.get_block_position(curr_point[0], curr_point[1]))
        return points

#
# img = PictureOcr('./imgs/test_pic.png')
# img.init_model()
# u = OneLine(img.dims[0], img.dims[1])
# u.set_map(img.get_matrix())
# u.set_init_point(img.init_point)
# print(u.map)
# print(u.init)
# result = u.start_find()
# print(result)
# print(img.translate_op(result))

# u = OneLine(6, 6)
# u.set_map(np.array([[0, 1, 0, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1],
#                     [0, 1, 1, 1, 0, 1],
#                     [1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1]]))
# u.set_init_point([5, 1])
# u.start_find()
