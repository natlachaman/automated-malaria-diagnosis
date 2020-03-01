#!/usr/bin/env python3
# -*- coding: utf-8 -*
import numpy as np
from scipy import misc
import scipy as scp
import skimage
import os
import copy

from keras.preprocessing import image
from scipy.ndimage import zoom


def add_noise(img, var=0.0005):
    img_noise = skimage.util.random_noise(img, mode='gaussian', seed=0, var=var)
    return np.round(img_noise * 255.)


def blurr_image(img, sigma=3):
    # img_ = img
    img_blur = np.zeros(shape=img.shape)
    img_blur[:, :, 0] = scp.ndimage.gaussian_filter(img[:, :, 0], sigma=sigma)
    img_blur[:, :, 1] = scp.ndimage.gaussian_filter(img[:, :, 1], sigma=sigma)
    img_blur[:, :, 2] = scp.ndimage.gaussian_filter(img[:, :, 2], sigma=sigma)
    return np.round(img_blur)


def rotate_point(point, angle, img):
    theta = np.pi / 180 * angle * -1
    img_center = np.array([img.shape[1] / 2 - 1, img.shape[0] / 2 - 1]).reshape((1, 2))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rect_point = np.array([point[0] - img_center[0, 0], img_center[0, 1] - point[1]], dtype=float).reshape((1, 2))
    rect_point2 = np.matmul(rect_point, rotation_matrix)
    rect_point3 = np.array([rect_point2[0, 0] + img_center[0, 0], img_center[0, 1] - rect_point2[0, 1]]).reshape((1, 2))

    return rect_point3[0, 0], rect_point3[0, 1]


def rotate_rectangle(points, angle, img):
    ymin, xmin, ymax, xmax = points

    point1 = rotate_point((xmin, ymin), angle, img)
    point2 = rotate_point((xmax, ymin), angle, img)
    point3 = rotate_point((xmin, ymax), angle, img)
    point4 = rotate_point((xmax, ymax), angle, img)

    xmin_t = min(point1[0], point2[0], point3[0], point4[0])
    xmax_t = max(point1[0], point2[0], point3[0], point4[0])
    ymin_t = min(point1[1], point2[1], point3[1], point4[1])
    ymax_t = max(point1[1], point2[1], point3[1], point4[1])

    return ymin_t, xmin_t, ymax_t, xmax_t


def flip_rectangle(points, axis, img):
    ymin, xmin, ymax, xmax = points
    h, w, _ = img.shape

    if axis == 0:
        ymin = h - ymin
        ymax = h - ymax
    elif axis == 1:
        xmin = w - xmin
        xmax = w - xmax
    else:
        raise ValueError('axis can either be "0" (vertical) or "1" (horizontal)', str(axis))

    if xmin > xmax:
        xt = xmax
        xmax = xmin
        xmin = xt
    if ymin > ymax:
        xt = ymax
        ymax = ymin
        ymin = xt

    return ymin, xmin, ymax, xmax


def rotate(img, bb, angle):
    rot_img = np.asarray(scp.ndimage.rotate(img, angle), dtype=np.float32)
    rot_bb = []
    for i in bb:
        rot_bb.append(rotate_rectangle(i, angle, img))

    return rot_img, np.array(rot_bb)


def flip(img, bb, axis):
    flip_img = np.asarray(np.flip(img, axis=axis), dtype=np.float32)
    flip_bb = []
    for i in bb:
        flip_bb.append(flip_rectangle(i, axis, img))

    return flip_img, np.array(flip_bb)


def crop_to_images(image, bboxes, labels, size, return_coord=False):

    # select bounding boxes from crop
    def crop_bboxes(bboxes, labels, boundaries):
        # print('boundaries', boundaries)
        boxes, objclass = [], []
        for (ymin, xmin, ymax, xmax), l in zip(bboxes, labels):
            x, y = xmin + int((xmax - xmin) / 2.), ymin + int((ymax - ymin) / 2.)
            # 5 pixels tolerance
            if boundaries[0] - 10 < x <= boundaries[1] + 10 and boundaries[2] - 10 < y <= boundaries[3] + 10:
                boxes.append((ymin - boundaries[2],
                              xmin - boundaries[0],
                              ymax - boundaries[2],
                              xmax - boundaries[0]))
                objclass.append(l)
        return np.array(boxes), np.array(objclass)

    num_samples_width = int(np.ceil(image.shape[1] / float(size)))
    num_samples_height = int(np.ceil(image.shape[0] / float(size)))

    # meshgrid of center coordinates
    half = int(size / 2.)
    x, y = np.meshgrid(np.arange(half, num_samples_width * size, size),
                       np.arange(half, num_samples_height * size, size))
    x, y = x.ravel(), y.ravel()

    # correct for center coordinates that are out of range
    is_outside = x + half > image.shape[1]
    if any(is_outside):
        idx = np.where(is_outside)[0]
        x[idx] = image.shape[1] - half

    is_outside = y + half > image.shape[0]
    if any(is_outside):
        idx = np.where(is_outside)[0]
        y[idx] = image.shape[0] - half

    # crop and gather only boxes that fall inside the crop
    new_images, new_bboxes, coordinates = [], [], []
    for xi, yi in zip(x, y):
        xmin, xmax = xi - half, xi + half
        ymin, ymax = yi - half, yi + half

        new_images.append(image[ymin:ymax, xmin:xmax, :])
        new_bboxes.append(crop_bboxes(bboxes, labels, (xmin, xmax, ymin, ymax)))
        coordinates.append([ymin, xmin, ymax, xmax])

    if return_coord:
        return new_images, new_bboxes, coordinates
    else:
        return new_images, new_bboxes


def clipped_zoom_in(img, bb, zoom_factor, **kwargs):

    h, w = img.shape[:2]
    if bb.any():
        ymin, xmin, ymax, xmax = bb.T
        bw, bh = xmax - xmin, ymax - ymin
        x, y = xmin + np.int32(bw / 2.), ymin + np.int32(bh / 2.)

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # bounding box of the clip region within the input array
    top = (zh - h) // 2
    left = (zw - w) // 2
    out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

    # clip and zoom coordinates too
    if bb.any():
        x, y = (x - left) * zoom_factor, (y - top) * zoom_factor
        bw, bh = bw * zoom_factor, bh * zoom_factor

    # `out` might still be slightly larger than `img` due to rounding, so
    # trim off any extra pixels at the edges
    trim_top = ((out.shape[0] - h) // 2)
    trim_left = ((out.shape[1] - w) // 2)
    out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # clip coordinates once more
    if bb.any():
        x, y = x - trim_left, y - trim_top

        # delete x,y coordinates that fall outside 'out' (5 pixels tolerance)
        bb_out = []
        for i, (xi, yi) in enumerate(zip(x, y)):
            if -5 < xi < w + 5 and -5 < yi < h + 5:
                xmin, xmax = int(xi - (bw[i] / 2.)), int(xi + (bw[i] / 2.))
                ymin, ymax = int(yi - (bh[i] / 2.)), int(yi + (bh[i] / 2.))
                bb_out.append([ymin, xmin, ymax, xmax])

        return out, np.array(bb_out)
    else:
        return out, np.zeros((0, 4))