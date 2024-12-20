import importlib.metadata
import torch
import logging
from scipy.interpolate import PchipInterpolator
import numpy as np
import cv2
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.31.0'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")

def print_memory(device):
    memory = torch.cuda.memory_allocated(device) / 1024**3
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    log.info(f"Allocated memory: {memory=:.3f} GB")
    log.info(f"Max allocated memory: {max_memory=:.3f} GB")
    log.info(f"Max reserved memory: {max_reserved=:.3f} GB")
    #memory_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    #log.info(f"Memory Summary:\n{memory_summary}")

def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize // 2, imgSize // 2), imgSize // 2, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = (
                1
                / 2
                / np.pi
                / (40**2)
                * np.exp(-1 / 2 * ((i - imgSize / 2) ** 2 / (40**2) + (j - imgSize / 2) ** 2 / (40**2)))
            )

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage) * 255).astype(np.uint8)

    return isotropicGrayscaleImage


def get_vis_image(
    target_size=(512, 512),
    points=None,
    side=20,
    num_frames=14,
    # original_size=(512 , 512), args="", first_frame=None, is_mask = False, model_id=None,
):

    # images = []
    vis_images = []
    heatmap = gen_gaussian_heatmap()

    trajectory_list = []
    radius_list = []

    for index, point in enumerate(points):
        trajectories = [[int(i[0]), int(i[1])] for i in point]
        trajectory_list.append(trajectories)

        radius = 10
        radius_list.append(radius)

    if len(trajectory_list) == 0:
        vis_images = [Image.fromarray(np.zeros(target_size, np.uint8)) for _ in range(num_frames)]
        return vis_images

    for idxx, point in enumerate(trajectory_list[0]):
        new_img = np.zeros(target_size, np.uint8)
        vis_img = new_img.copy()
        # ids_embedding = torch.zeros((target_size[0], target_size[1], 320))

        if idxx >= num_frames:
            break

        # for cc, (mask, trajectory, radius) in enumerate(zip(mask_list, trajectory_list, radius_list)):
        for cc, (trajectory, radius) in enumerate(zip(trajectory_list, radius_list)):

            center_coordinate = trajectory[idxx]
            trajectory_ = trajectory[:idxx]
            side = min(radius, 50)

            y1 = max(center_coordinate[1] - side, 0)
            y2 = min(center_coordinate[1] + side, target_size[0] - 1)
            x1 = max(center_coordinate[0] - side, 0)
            x2 = min(center_coordinate[0] + side, target_size[1] - 1)

            if x2 - x1 > 3 and y2 - y1 > 3:
                need_map = cv2.resize(heatmap, (x2 - x1, y2 - y1))
                new_img[y1:y2, x1:x2] = need_map.copy()

                if cc >= 0:
                    vis_img[y1:y2, x1:x2] = need_map.copy()
                    if len(trajectory_) == 1:
                        vis_img[trajectory_[0][1], trajectory_[0][0]] = 255
                    else:
                        for itt in range(len(trajectory_) - 1):
                            cv2.line(
                                vis_img,
                                (trajectory_[itt][0], trajectory_[itt][1]),
                                (trajectory_[itt + 1][0], trajectory_[itt + 1][1]),
                                (255, 255, 255),
                                3,
                            )

        img = new_img

        # Ensure all images are in RGB format
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
            #vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
      

        # Convert the numpy array to a PIL image
        # pil_img = Image.fromarray(img)
        # images.append(pil_img)
        vis_images.append(vis_img)

    return vis_images
