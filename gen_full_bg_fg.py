import os
import cv2
import glob
import time
import random
import imutils
import numpy as np
from numba import jit


def rad(x):
    return x * np.pi / 180


def get_warp(img, angle_x, angle_y, angle_z, fov):
    h, w = img.shape[:2]
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))

    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(angle_x)), -np.sin(rad(angle_x)), 0],
                   [0, -np.sin(rad(angle_x)), np.cos(rad(angle_x)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angle_y)), 0, np.sin(rad(angle_y)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angle_y)), 0, np.cos(rad(angle_y)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(angle_z)), np.sin(rad(angle_z)), 0, 0],
                   [-np.sin(rad(angle_z)), np.cos(rad(angle_z)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    radius = rx.dot(ry).dot(rz)

    p_center = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - p_center
    p2 = np.array([w, 0, 0, 0], np.float32) - p_center
    p3 = np.array([0, h, 0, 0], np.float32) - p_center
    p4 = np.array([w, h, 0, 0], np.float32) - p_center

    dst1 = radius.dot(p1)
    dst2 = radius.dot(p2)
    dst3 = radius.dot(p3)
    dst4 = radius.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + p_center[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + p_center[1]

    warp = cv2.getPerspectiveTransform(org, dst)
    result = cv2.warpPerspective(img, warp, (w, h))
    return warp, result


def fit_image(img):
    org = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for y in range(img.shape[0]):
        if max(gray[y]) > 7:
            gray = gray[y:, :]
            img = img[y:, :, :]
            break
    up = org.shape[0] - img.shape[0]

    for y in reversed(range(img.shape[0])):
        if max(gray[y]) > 5:
            img = img[:y, :, :]
            break
    return img, up


def get_label(img, img_name, warp_mt,
              top, bot, left, right,
              fit=0, vis=False):
    h, w = img.shape[:2]
    inside = np.array([[left, top],
                       [w - right, top],
                       [left, h - bot],
                       [w - right, h - bot]], np.float32)

    labels = []
    for point in inside:
        inputs = point.copy().tolist()
        inputs.append(1.0)
        outputs = np.array(inputs).dot(warp_mt.T)
        point_output = (outputs / outputs[-1])[:-1]
        labels.append(point_output)
    labels = np.array(labels, np.int64)

    if fit > 0:
        img, _ = fit_image(img)

    labels = np.array(labels, np.int64)
    labels[0][1] -= fit
    labels[1][1] -= fit
    labels[2][1] -= fit
    labels[3][1] -= fit

    if vis:
        visual = img.copy()
        cv2.line(visual, labels[0], labels[1], (0, 0, 255), 3)
        cv2.line(visual, labels[1], labels[3], (0, 0, 255), 3)
        cv2.line(visual, labels[0], labels[2], (0, 0, 255), 3)
        cv2.line(visual, labels[2], labels[3], (0, 0, 255), 3)
        vis_path = os.path.join('vis', img_name)
        cv2.imwrite(vis_path, visual)

    return np.array([labels[0], labels[1], labels[3], labels[2]])


def export_label(folder, labels, name):
    label_name = name[:-4] + '.txt'
    label_path = os.path.join(folder, label_name)
    with open(label_path, 'w') as f:
        coordinates = ', '.join(str(c) for c in labels.flatten())
        content = coordinates + ', 0'
        f.write(content)


class ImageTransformer:
    def __init__(self, folder,
                 org, img_name,
                 top, bot, left, right,
                 mode_border, fixed_height=736,
                 vis=True):
        self.folder = folder
        self.img_name = img_name
        self.img = imutils.resize(org, height=fixed_height)

        self.ratio = fixed_height / org.shape[0]
        self.top = top * self.ratio
        self.bot = bot * self.ratio
        self.left = left * self.ratio
        self.right = right * self.ratio

        self.vis = vis
        self.mode = mode_border

    def x_transform(self):
        f = random.randrange(0, 80, 20)
        fov = 42 + f
        # angle_x = random.randrange(-55, 80, 15)
        # angle_x = 0
        angle_x = random.randrange(-25, 5, 5)
        warp, result = get_warp(self.img, angle_x, 0, 0, fov)

        angle_x_name = 'posX_' + str(angle_x) if angle_x >= 0 else 'negX_' + str(abs(angle_x))
        save_name = '{}_{}_f_{}_{}.jpg'.format(self.img_name, angle_x_name, fov, self.mode)

        label = get_label(result, save_name, warp,
                          self.top, self.bot, self.left, self.right,
                          vis=self.vis)

        return result, label, save_name

    def y_transform(self):
        f = random.randrange(0, 80, 20)
        fov = 42 + f
        # angle_y = random.randrange(-50, 70, 20)
        angle_y = random.randrange(-10, 20, 5)
        warp, result = get_warp(self.img, 0, angle_y, 0, fov)

        angle_y_name = 'posY_' + str(angle_y) if angle_y >= 0 else 'negY_' + str(abs(angle_y))
        save_name = '{}_{}_f_{}_{}.jpg'.format(self.img_name, angle_y_name, fov, self.mode)

        label = get_label(result, save_name, warp,
                          self.top, self.bot, self.left, self.right,
                          vis=self.vis)

        return result, label, save_name

    def xz_transform(self):
        f = random.randrange(0, 50, 20)
        fov = 42 + f
        # angle_z = random.randrange(-20, 30, 10)
        # angle_x = random.randrange(-20, 30, 10)
        angle_z = random.randrange(-10, 20, 5)
        angle_x = random.randrange(-20, 5, 5)
        warp, result = get_warp(self.img, angle_x, 0, angle_z, fov)

        angle_x_name = 'posX_' + str(angle_x) if angle_x >= 0 else 'negX_' + str(abs(angle_x))
        angle_z_name = 'posZ_' + str(angle_z) if angle_z >= 0 else 'negZ_' + str(abs(angle_z))
        save_name = '{}_{}_{}_f_{}_{}.jpg'.format(self.img_name, angle_x_name, angle_z_name, fov, self.mode)

        label = get_label(result, save_name, warp,
                          self.top, self.bot, self.left, self.right,
                          vis=self.vis)

        return result, label, save_name

    def yz_transform(self):
        f = random.randrange(0, 50, 20)
        fov = 42 + f
        # angle_z = random.randrange(-20, 30, 10)
        # angle_y = random.randrange(-20, 30, 10)
        angle_z = random.randrange(-10, 10, 5)
        angle_y = random.randrange(-10, 10, 5)
        warp, result = get_warp(self.img, 0, angle_y, angle_z, fov)

        angle_y_name = 'posY_' + str(angle_y) if angle_y >= 0 else 'negY_' + str(abs(angle_y))
        angle_z_name = 'posZ_' + str(angle_z) if angle_z >= 0 else 'negZ_' + str(abs(angle_z))
        save_name = '{}_{}_{}_f_{}_{}.jpg'.format(self.img_name, angle_y_name, angle_z_name, fov, self.mode)

        label = get_label(result, save_name, warp,
                          self.top, self.bot, self.left, self.right,
                          vis=self.vis)

        return result, label, save_name

    def fit_height_transform(self):
        f = random.randrange(0, 80, 20)
        fov = 42 + f
        angle_x = random.randrange(-55, 5, 15)
        warp, result = get_warp(self.img, angle_x, 0, 0, fov)
        result_fit, up = fit_image(result)

        angle_x_name = 'negX_' + str(abs(angle_x))
        save_name = '{}_fit_{}_f_{}.jpg'.format(self.img_name, angle_x_name, fov)

        label = get_label(result, save_name, warp,
                          self.top, self.bot, self.left, self.right,
                          fit=up, vis=self.vis)

        return result_fit, label, save_name


def border(img):
    mode_dict = {0: 'center',
                 1: 'bot_left',
                 2: 'top_left',
                 3: 'bot_right',
                 4: 'top_right'}
    prob = [0.8, 0.08, 0.02, 0.08, 0.02]
    mode = np.random.choice(5, p=prob)
    padding_x = img.shape[0] // 5 * 2
    # padding_y = img.shape[1] // 5 * 2
    padding_y = img.shape[1] // 20

    '''Padding mode'''
    if mode == 1:
        top, bot, left, right = padding_y, padding_y // 2, padding_x // 2, padding_x
    elif mode == 2:
        top, bot, left, right = padding_y // 2, padding_y, padding_x // 2, padding_x
    elif mode == 3:
        top, bot, left, right = padding_y, padding_y // 2, padding_x, padding_x // 2
    elif mode == 4:
        top, bot, left, right = padding_y // 2, padding_y, padding_x, padding_x // 2
    else:
        top, bot, left, right = padding_y, padding_y, padding_x, padding_x

    bordered = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, 0)

    return bordered, top, bot, left, right, mode_dict[mode]


@jit(nopython=True)
def is_inside(point, poly):
    x, y = point[0], point[1]
    n = len(poly)
    inside = False
    x_ints = 0.0
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_ints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_ints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@jit(nopython=True)
def distance(point, p1, p2):
    x0, y0 = point[0], point[1]
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    dist = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / np.sqrt(np.square(x2-x1) + np.square(y2-y1))
    return dist


@jit(nopython=True)
def bg2fg(bg, fg, polygon):
    h, w = bg.shape[:2]
    # max_dist = math.sqrt(2)
    max_dist = 1
    for y in range(h):
        for x in range(w):
            point = np.array([x, y])
            if is_inside(point, polygon) and distance(point, polygon[0], polygon[1]) > max_dist\
                    and distance(point, polygon[1], polygon[2]) > max_dist\
                    and distance(point, polygon[2], polygon[3]) > max_dist\
                    and distance(point, polygon[3], polygon[0]) > max_dist:
                bg[y, x] = fg[y, x]
    return bg


def combine_fg_bg(foreground, background, label):
    h_fg, w_fg = foreground.shape[:2]

    background = cv2.resize(background, (w_fg, h_fg))
    # start = time.time()

    # h, w = bg.shape[:2]
    # polygon = Polygon([tuple(label[0]),
    #                    tuple(label[1]),
    #                    tuple(label[3]),
    #                    tuple(label[2])])
    # line1 = LineString([tuple(label[0]), tuple(label[1])])
    # line2 = LineString([tuple(label[1]), tuple(label[3])])
    # line3 = LineString([tuple(label[3]), tuple(label[2])])
    # line4 = LineString([tuple(label[2]), tuple(label[0])])
    # for y in range(h):
    #     for x in range(w):
    #         point = Point(x, y)
    #         if point.within(polygon)\
    #                 and line1.distance(point) > np.sqrt(2)\
    #                 and line2.distance(point) > np.sqrt(2)\
    #                 and line3.distance(point) > np.sqrt(2)\
    #                 and line4.distance(point) > np.sqrt(2):
    #             bg[y, x] = foreground[y, x]

    # print(time.time() - start)

    # _, threshold = cv2.threshold(fg_gray, 1, 255, cv2.THRESH_BINARY_INV)
    # kernel = np.ones((7, 7), np.uint8)
    # threshold = cv2.dilate(threshold, None, iterations=1)
    # threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    # bitwise = cv2.bitwise_and(bg, threshold)
    # result = cv2.bitwise_or(foreground, bitwise)

    return bg2fg(background, foreground, label)


if __name__ == '__main__':
    start_all = time.time()
    # print(start_all)
    NUM_IMG = 3000
    for num in range(NUM_IMG):
        BG_PATH = random.choice(glob.glob('background_test/*.jpg'))
        # IMG_PATH = random.choice(glob.glob('org_img_test/*.jpg'))
        IMG_PATH = random.choice(glob.glob(r'D:\Laos_eKYC\Dataset\IDCard\*.*'))

        BG_NAME = BG_PATH.split('\\')[-1][:-4]
        IMG_NAME = IMG_PATH.split('\\')[-1][:-4]

        BG = cv2.imread(BG_PATH)
        IMG = cv2.imread(IMG_PATH)

        bor, t, b, l, r, MODE = border(IMG)
        trans = ImageTransformer('temp',
                                 bor, IMG_NAME,
                                 t, b, l, r,
                                 MODE, vis=False)
        list_transform = [trans.x_transform, trans.y_transform,
                          trans.xz_transform, trans.yz_transform,
                          trans.fit_height_transform]

        # FG, LABEL, SAVE_NAME = np.random.choice(list_transform, p=[0.4, 0.025, 0.2, 0.025, 0.35])()
        FG, LABEL, SAVE_NAME = np.random.choice(list_transform, p=[1., 0., 0., 0., 0.])()
        CB_IMG = combine_fg_bg(FG, BG, LABEL)
        cv2.imshow('out', CB_IMG)
        cv2.waitKey(0)

        # SAVE_PATH = 'test/img/' + BG_NAME + '_' + SAVE_NAME
        # cv2.imwrite(SAVE_PATH, CB_IMG)
        # export_label(r'test\\gt', LABEL, BG_NAME + '_' + SAVE_NAME)

        '''Test tiny imgs'''
        # BG_TINY_PATH = os.path.join('background_tiny', BG_NAME + '.jpg')
        # BG_TINY = cv2.imread(BG_TINY_PATH)
        # CB_IMG_TINY = combine_fg_bg(FG, BG_TINY, LABEL)
        # SAVE_TINY_PATH = 'img_tiny/' + BG_NAME + '_' + SAVE_NAME
        # cv2.imwrite(SAVE_TINY_PATH, CB_IMG_TINY)
        #
        # print(os.path.getsize(SAVE_PATH) / 2**10,
        #       os.path.getsize(SAVE_TINY_PATH) / 2**10)

        if (num + 1) % 100 == 0:
            print('{} images generated'.format(num + 1))

    print(time.time() - start_all)
