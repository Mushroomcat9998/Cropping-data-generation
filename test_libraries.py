import os
import cv2
import time
import numba
import numpy as np
from numba import jit, njit

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point, LineString

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bg = np.zeros((500, 500), dtype=np.uint8)
    bg2 = np.zeros((500, 500), dtype=np.uint8)
    label = np.array([[150, 120], [240, 120],
                      [150, 400], [240, 400]])

    #
    # cv2.line(bg, label[0], label[1], 255, 1)
    # cv2.line(bg, label[1], label[3], 255, 1)
    # cv2.line(bg, label[0], label[2], 255, 1)
    # cv2.line(bg, label[2], label[3], 255, 1)
    # cv2.imshow('result', bg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # @jit(nopython=True)
    def ray_tracing(point, label):
        polygon = Polygon([tuple(label[0]),
                           tuple(label[1]),
                           tuple(label[3]),
                           tuple(label[2])])

        return polygon.contains(point)

    # @jit(nopython=True)
    def is_inside_sm_parallel(polygon, img):
        h, w = img.shape[:2]
        for y in range(h):
            for x in range(w):
                point = Point(x, y)
                if ray_tracing(point, polygon):
                    # if point in [[150, 120], [240, 120], [150, 400], [240, 400]]:
                    img[y, x] = 255
        return img


    # for i in range(1):
    #     start = time.time()
    #     print(len(is_inside_sm_parallel(label, bg)))
    #     end = time.time()
    #     print("Elapsed (with compilation) = %s" % (end - start))
    # cv2.imshow('result', is_inside_sm_parallel(label, bg))
    # cv2.waitKey(0)

    print(os.stat('background_tiny/0A633E21-6D4A-411C-A235-74972D98F9EF.jpg').st_size / 2**10,
          os.path.getsize('background_tiny/0A633E21-6D4A-411C-A235-74972D98F9EF.jpg') / 2**10)

    # # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
    # start = time.time()
    #
    # end = time.time()
    # print("Elapsed (after compilation) = %s" % (end - start))
