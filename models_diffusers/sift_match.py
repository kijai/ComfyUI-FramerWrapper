from scipy.interpolate import interp1d, PchipInterpolator

import numpy as np
from PIL import Image
import cv2
import torch


def sift_match(
    img1, img2,
    thr=0.5, 
    topk=5, method="max_dist",
):
    
    assert method in ["max_dist", "random", "max_score", "max_score_even"]

    # small threshold means less points
    # img1 is torch tensor with shape (B, H , W, C)
    # 1. to cv2 grayscale image
    img1_rgb = img1.squeeze().cpu().numpy() * 255
    img2_rgb = img2.squeeze().cpu().numpy() * 255

    # Check if the images are not empty
    if img1_rgb.size == 0 or img2_rgb.size == 0:
        raise ValueError("One of the images is empty after conversion from tensor to numpy array.")
    img1_rgb = img1_rgb.astype(np.uint8)
    img2_rgb = img2_rgb.astype(np.uint8)

    
    img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. use sift to extract keypoints and descriptors
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    point_list = []
    distance_list = []

    if method in ['max_score', 'max_score_even']:
        matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)

        anchor_points_list = []
        for m, n in matches[:topk]:
            print(m.distance / n.distance)

            # check evenly distributed
            if method == 'max_score_even':
                to_close = False
                for anchor_point in anchor_points_list:
                    pt1 = kp1[m.queryIdx].pt
                    dist = np.linalg.norm(np.array(pt1) - np.array(anchor_point))
                    if dist < 50:
                        to_close = True
                        break
                if to_close:
                    continue

            good.append([m])

            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            distance_list.append(dist)

            anchor_points_list.append(pt1)

            pt1 = torch.tensor(pt1)
            pt2 = torch.tensor(pt2)
            pt = torch.stack([pt1, pt2])  # (2, 2)
            point_list.append(pt)

    if method in ['max_dist', 'random']:
        for m, n in matches:
            if m.distance < thr * n.distance:
                good.append([m])

                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
                distance_list.append(dist)

                pt1 = torch.tensor(pt1)
                pt2 = torch.tensor(pt2)
                pt = torch.stack([pt1, pt2])  # (2, 2)
                point_list.append(pt)

        distance_list = np.array(distance_list)
        # only keep the points with the largest topk distance
        idx = np.argsort(distance_list)
        if method == "max_dist":
            idx = idx[-topk:]
        elif method == "random":
            topk = min(topk, len(idx))
            idx = np.random.choice(idx, topk, replace=False)
        elif method == "max_score":
            import pdb; pdb.set_trace()
            raise NotImplementedError
            # idx = np.argsort(distance_list)[:topk]
        else:
            raise ValueError(f"Unknown method {method}")

        point_list = [point_list[i] for i in idx]
        good = [good[i] for i in idx]

    # # cv2.drawMatchesKnn expects list of lists as matches.
    # draw_params = dict(
    #     matchColor=(255, 0, 0),
    #     singlePointColor=None,
    #     flags=2,
    # )
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, **draw_params)


    # # manually draw the matches, the images are put in horizontal
    # img3 = np.concatenate([img1_rgb, img2_rgb], axis=1)  # (h, 2w, 3)
    # for m in good:
    #     pt1 = kp1[m[0].queryIdx].pt
    #     pt2 = kp2[m[0].trainIdx].pt
    #     pt1 = (int(pt1[0]), int(pt1[1]))
    #     pt2 = (int(pt2[0]) + img1_rgb.shape[1], int(pt2[1]))
    #     cv2.line(img3, pt1, pt2, (255, 0, 0), 1)

    # manually draw the matches, the images are put in vertical. with 10 pixels margin
    margin = 10
    img3 = np.zeros((img1_rgb.shape[0] + img2_rgb.shape[0] + margin, max(img1_rgb.shape[1], img2_rgb.shape[1]), 3), dtype=np.uint8)
    # the margin is white
    img3[:, :] = 255
    img3[:img1_rgb.shape[0], :img1_rgb.shape[1]] = img1_rgb
    img3[img1_rgb.shape[0] + margin:, :img2_rgb.shape[1]] = img2_rgb
    # create a color list of 6 different colors
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for color_idx, m in enumerate(good):
        pt1 = kp1[m[0].queryIdx].pt
        pt2 = kp2[m[0].trainIdx].pt
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]) + img1_rgb.shape[0] + margin)
        # cv2.line(img3, pt1, pt2, (255, 0, 0), 1)
        # avoid the zigzag artifact in line
        # random_color = tuple(np.random.randint(0, 255, 3).tolist())
        color = color_list[color_idx % len(color_list)]
        cv2.line(img3, pt1, pt2, color, 1, lineType=cv2.LINE_AA)
        # add a empty circle to both start and end points
        cv2.circle(img3, pt1, 3, color, lineType=cv2.LINE_AA)
        cv2.circle(img3, pt2, 3, color, lineType=cv2.LINE_AA)

    #Image.fromarray(img3).save(output_path)
    #print(f"Save the sift matches to {output_path}")

    # (f, topk, 2), f=2 (before interpolation)
    if len(point_list) == 0:
        return None

    point_list = torch.stack(point_list)
    point_list = point_list.permute(1, 0, 2)

    return point_list, img3


def interpolate_trajectory(points_torch, num_frames, t=None):
    # points:(f, topk, 2), f=2 (before interpolation)

    num_points = points_torch.shape[1]
    points_torch = points_torch.permute(1, 0, 2)  # (topk, f, 2)

    points_list = []
    for i in range(num_points):
        # points:(f, 2)
        points = points_torch[i].cpu().numpy()

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        if t is None:
            t = np.linspace(0, 1, len(points))

        # fx = interp1d(t, x, kind='cubic')
        # fy = interp1d(t, y, kind='cubic')
        fx = PchipInterpolator(t, x)
        fy = PchipInterpolator(t, y)

        new_t = np.linspace(0, 1, num_frames)

        new_x = fx(new_t)
        new_y = fy(new_t)
        new_points = list(zip(new_x, new_y))

        points_list.append(new_points)

    points = torch.tensor(points_list)  # (topk, num_frames, 2)
    points = points.permute(1, 0, 2)  # (num_frames, topk, 2)

    return points


# diffusion feature matching
def point_tracking(
    F0,
    F1,
    handle_points,
    handle_points_init,
    track_dist=5,
):
    # handle_points: (num_points, 2)
    # NOTE: 
    # 1. all row and col are reversed 
    # 2. handle_points in (y, x), not (x, y)

    # reverse row and col
    handle_points = torch.stack([handle_points[:, 1], handle_points[:, 0]], dim=-1)
    handle_points_init = torch.stack([handle_points_init[:, 1], handle_points_init[:, 0]], dim=-1)

    with torch.no_grad():
        _, _, max_r, max_c = F0.shape

        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = max(0, int(pi[0]) - track_dist), min(max_r, int(pi[0]) + track_dist + 1)
            c1, c2 = max(0, int(pi[1]) - track_dist), min(max_c, int(pi[1]) + track_dist + 1)
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            # handle_points[i][0] = pi[0] - track_dist + row
            # handle_points[i][1] = pi[1] - track_dist + col
            handle_points[i][0] = r1 + row
            handle_points[i][1] = c1 + col

        handle_points = torch.stack([handle_points[:, 1], handle_points[:, 0]], dim=-1)  # (num_points, 2)

        return handle_points
