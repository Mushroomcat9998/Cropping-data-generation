import cv2
import numpy as np
import sys

img = cv2.imread('091.jpg')

# Optional, expand the image to ensure that the content does not exceed the visible range
img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
w, h = img.shape[:2]

anglex = 0
angley = 0
anglez = 0  # is rotation
fov = 42
r = 0


def rad(x):
    return x * np.pi / 180


def get_warp():
    global anglex, angley, anglez, fov, w, h, r
    # The distance between the lens and the image, 21 is a half angle of view,
    # the distance of z is calculated to ensure that the entire image is displayed exactly at this viewing angle
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # Homogeneous transformation matrix
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # Generation of four pairs of points
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # Project onto the imaging plane
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)
    return warpR


def gen_image():
    global anglex, angley, anglez, fov, r

    for f in range(0, 70, 10):
        fov = 42 + f

        anglex = 0
        for y in range(10, 60, 10):
            angley = y
            warp = get_warp()
            result = cv2.warpPerspective(img, warp, (h, w))
            cv2.imwrite('samples/posY_{}_{}.jpg'.format(y, fov), result)

            angley = -y
            warp = get_warp()
            result = cv2.warpPerspective(img, warp, (h, w))
            cv2.imwrite('samples/negY_{}_{}.jpg'.format(y, fov), result)

        angley = 0
        for x in range(10, 60, 10):
            anglex = x
            warp = get_warp()
            result = cv2.warpPerspective(img, warp, (h, w))
            cv2.imwrite('samples/posX_{}_{}.jpg'.format(x, fov), result)

            anglex = -x
            warp = get_warp()
            result = cv2.warpPerspective(img, warp, (h, w))
            cv2.imwrite('samples/negX_{}_{}.jpg'.format(x, fov), result)


def control():
    global anglex, angley, anglez, fov, r

    # Keyboard control
    if 27 == c:  # Esc quit
        sys.exit()
    if c == ord('w'):
        anglex += 10
    if c == ord('s'):
        anglex -= 10
    if c == ord('a'):
        angley += 10
    if c == ord('d'):
        angley -= 10
    if c == ord('u'):
        anglez += 10
    if c == ord('p'):
        anglez -= 10
    if c == ord('t'):
        fov += 20
    if c == ord('r'):
        fov -= 20
    if c == ord(' '):
        anglex = angley = anglez = 0
    # if c == ord('e'):
    #     print("======================================")
    #     print('Rotation Matrix:')
    #     print(r)
    #     print('angle alpha(anglex):')
    #     print(anglex)
    #     print('angle beta(angley):')
    #     print(angley)
    #     print('dz(anglez):')
    #     print(anglez)


while True:
    warp = get_warp()

    result = cv2.warpPerspective(img, warp, (h, w))
    # cv2.namedWindow('result', 2)
    cv2.imshow("result", result)
    c = cv2.waitKey(0)
    control()


