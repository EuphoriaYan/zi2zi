# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
from imageio import imread
import numpy as np
from io import StringIO, BytesIO
from PIL import Image
from scipy import ndimage


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    # return StringIO(bytes_img)
    return BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    mat = imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h, _ = img.shape
    # old realization(scipy < 1.0.0)
    # enlarged = scipy.misc.imresize(img, [nw, nh])
    # new realization
    enlarged = np.array(Image.fromarray(np.uint8(img)).resize((nw, nh), Image.ANTIALIAS))
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def rotate_image(img, angle):
    w, h, _ = img.shape
    # img_rotate = misc.imrotate(img, angle, interp="bilinear")
    img = Image.fromarray(img)
    img_rotate = img.rotate(angle,
                            resample=Image.BILINEAR,
                            fillcolor=(255, 255, 255))
    img_rotate = np.array(img_rotate)
    # img_rotate = ndimage.rotate(img, angle, reshape=False)
    return img_rotate


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    imageio.imsave(img_path, concated)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    # old realization(scipy < 1.0.0)
    # images = [imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    # new realization
    images = [np.array(
        Image.fromarray(np.uint8(f)).resize(
            (int(f.shape[0]*0.33), int(f.shape[1]*0.33)),
            Image.ANTIALIAS
        )
    ) for f in frames]
    imageio.mimsave(gif_file, images, format='GIF', duration=0.1)
    return gif_file
