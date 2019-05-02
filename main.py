# -*- coding: utf-8 -*-
import os
import utils
import time
import sys

os.system('adb shell /system/bin/screencap -p /sdcard/test_pic.png')
os.system('adb pull /sdcard/test_pic.png ./imgs/test_pic.png')

pic = utils.PictureOcr('./imgs/test_pic.png')
pic.init_model()
one_line = utils.OneLine(pic.dims[0], pic.dims[1])
one_line.set_map(pic.get_matrix())
one_line.set_init_point(pic.init_point)
result = one_line.start_find()
points = pic.translate_op(result)


for point in points:
    os.system('adb shell input tap ' + str(point[1]) + ' ' + str(point[0]))