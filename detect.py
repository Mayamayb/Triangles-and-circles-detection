import cv2
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import math
from utils import calc_rect, IOU

def find_circ_BB(image_path, prnt=False):
    """Finds the circles and BBs in image using Hough Circles transform to find center and radius.
        detect circles in the red channel of the image (as the circles or dominantly red)
        returns a list of BBs.
        """

    boxes = []

    # load the image, clone it for output, and then convert it to RGB
    image = cv2.imread(image_path)
    h,w = image.shape[:2]
    if prnt:
        output = image.copy()
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1)
    # smooth image
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # produce binary image with a threshold, use only Red channel (the circles are red)
    th = 60
    ret, threshold = cv2.threshold(blur[:, :, 2], th, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT,  1.2, 200, param1=150,param2=50)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw a rectangle  in the output image
            # corresponding to the center of the circle
            int(r)
            if prnt:
                rect = patches.Rectangle((x-r, y-r), 2*r, 2*r, linewidth=1, edgecolor='cyan', facecolor='none')
                ax.add_patch(rect)
            min_corner = [max(x-r, 0), max(y-r, 0)]
            max_corner = [min(x+r, w), min(y+r, h)]
            boxes.append([min_corner,max_corner])

    if prnt:
        ax.imshow(output)
    return boxes

def remove_bb(boxes):
    """ removes redundant bbs from list which have great iou with other bbs in list.
    """
    for i in range(len(boxes)):
        boxA = boxes[i]
        if boxA is None:
            continue
        boxA_rect = calc_rect(boxA)
        for j in range( len(boxes)):
            boxB = boxes[j]
            if boxB is None:
                continue
            if np.array_equal(boxA, boxB):
                continue
            boxB_rect = calc_rect(boxB)

            iou = IOU(boxA_rect, boxB_rect)
            if iou>0.3:
                boxes[j] =None
    #filter out None elements from list
    boxes_final = list(filter(None, boxes))

    return boxes_final




def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.lstsq(A, b)[0]
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def intersections(lines, img,prnt =False, min_diag = 250):
    """Finds the intersections between 3 lines and calculate triangle bb."""
    h,w = img.shape[:2]
    boxes = []
    for i in range(len(lines)):
        line1=lines[i]
        for j in range(i, len(lines)):
            line2 = lines[j]
            for k in range(j, len(lines)):
                line3 = lines[k]
                # 3 different lines
                if np.array_equal(line1, line2) or np.array_equal(line1, line3) or np.array_equal(line2, line3):
                    continue

                x1, y1 = intersection(line1, line2)
                x2, y2 = intersection(line1, line3)
                x3, y3 = intersection(line2, line3)
                # if 3 points are in image (3 corners of triangle)
                if (x1>=0 and x1<=w) and (x2>=0 and x2<=w) and (x3>=0 and x3<=w) and (y1>=0 and y1<=h) and (y2>=0 and y2<=h) and (y3>=0 and y3<=h):
                    min_corner = [min(x1, x2, x3), min(y1, y2, y3)]
                    max_corner = [max(x1, x2, x3), max(y1, y2, y3)]
                    distance = math.sqrt( ((min_corner[0]-max_corner[0])**2)+((min_corner[1]-max_corner[1])**2) )
                    if distance<min_diag:
                        continue
                    boxes.append([min_corner,max_corner])
    # removes redundant bbs from list which have great iou with other bbs in list.
    boxes_final = remove_bb(boxes)
    if prnt:
        fig, ax = plt.subplots(1)

        for box in boxes_final:
            rect = patches.Rectangle((box[0][0], box[0][1]), box[1][0] - box[0][0],
                                     box[1][1] - box[0][1], linewidth=1.5, edgecolor='cyan', facecolor='none',
                                     linestyle="dashed")
            ax.add_patch(rect)


        ax.imshow(img)

    return boxes_final

def find_tri_BB(image_path,prnt = False,  print_lines = False):
    """Finds the traingles BBs in image using hough lines transform, a triangle is an intersection of 3 lines.
        returns a list of BBs.
        """
    # load the image, clone it for output, and then convert it to RGB
    image = cv2.imread(image_path)
    output = image.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    # smooth image
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # produce binary image with a threshold, use only Green channel (the triangles are green)
    th=60
    ret, threshold = cv2.threshold(blur[:,:,1], th, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('tr', threshold)


    lines = cv2.HoughLines(threshold, 1, 7 / 180, 150)
    # lines = cv2.HoughLines(threshold, 1, np.pi / 180, 100)
    lines = np.squeeze(lines)
    # eliminate some redundant lines
    if len(lines.shape)<2:
        bb=[]
        return bb
    tree = cKDTree(lines)
    rows_to_fuse = tree.query_pairs(r=10, output_type='ndarray')
    lines = np.delete(lines,rows_to_fuse[:,0], axis=0)
    lines=lines[:50]


    if print_lines:
        fig, ax = plt.subplots(1)

        for [rho, theta] in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ax.imshow(output)

    bb = intersections(lines, output, prnt)
    return bb
#
