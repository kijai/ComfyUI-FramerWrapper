# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import numpy as np
import cv2

import torch
import torch.nn.functional as F


def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    # Guass Map
    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)

    # isotropicGrayscaleImage = cv2.resize(isotropicGrayscaleImage, (40, 40))
    return isotropicGrayscaleImage


def draw_heatmap(img, center_coordinate, heatmap_template, side, width, height):
    x1 = max(center_coordinate[0] - side, 1)
    x2 = min(center_coordinate[0] + side, width - 1)
    y1 = max(center_coordinate[1] - side, 1)
    y2 = min(center_coordinate[1] + side, height - 1)
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

    if (x2 - x1) < 1 or (y2 - y1) < 1:
        print(center_coordinate, "x1, x2, y1, y2", x1, x2, y1, y2)
        return img

    need_map = cv2.resize(heatmap_template, (x2-x1, y2-y1))

    img[y1:y2,x1:x2] = need_map

    return img


def generate_gassian_heatmap(pred_tracks, pred_visibility=None, image_size=None, side=20):
    width, height = image_size
    num_frames, num_points = pred_tracks.shape[:2]

    point_index_list = [point_idx for point_idx in range(num_points)]
    heatmap_template = gen_gaussian_heatmap()


    image_list = []
    for frame_idx in range(num_frames):
        
        img = np.zeros((height, width), np.float32)
        for point_idx in point_index_list:
            px, py = pred_tracks[frame_idx, point_idx]

            if px < 0 or py < 0 or px >= width or py >= height:
                if (frame_idx == 0) or (frame_idx == num_frames - 1):
                    print(frame_idx, point_idx, px, py)
                continue

            if pred_visibility is not None:
                if (not pred_visibility[frame_idx, point_idx]):
                    continue

            img = draw_heatmap(img, (px, py), heatmap_template, side, width, height)

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        image_list.append(img)
    
    video_gaussion_map = torch.stack(image_list, dim=0)

    return video_gaussion_map