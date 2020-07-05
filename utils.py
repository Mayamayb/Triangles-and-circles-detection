import cv2
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json



def calc_rect(X):
    # returns left top corner, width, height
    w = abs(X[1][0] - X[0][0])
    h = abs(X[1][1] - X[0][1])
    return [X[0][0],X[0][1], w, h]

def show_image(img, GT=None, pred=None, ttl =None, save=False):
        """"plots and saves an image giving an absolute image path
            :param img: path to image
            :type img: string
            :param GT: if given GT file displays GT BBs
            :type GT: string,json file with dictioanry of BB of shapes
            :param pred: if given predictions file displays pred BBs
            :type pred: string, json file with dictioanry of BB of shapesparam Th: minimal threshold for iou excepted as TP
            :param ttl: title for displayed image
            :type ttl: string
            :param ttl: save image to current path with predictions
            :type ttl: bool
        """
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image)

        if GT:
            with open(GT, 'r') as f:
                GT_dict = json.load(f)
            circ = GT_dict['circle']
            for c in circ:
                # Create a Rectangle patch
                # uses bottom left and w ang h
                [x, y, w, h] = calc_rect(c)
                rect = patches.Rectangle((x,y), w, h, linewidth=1.5, edgecolor='magenta', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
            rect.set_label('GT circle')

            tri = GT_dict['triangle']
            for t in tri:
                [x, y, w, h] = calc_rect(t)
                rect = patches.Rectangle((x,y), w, h, linewidth=1.5, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
            rect.set_label('GT triangle')

        if pred:
            with open(pred, 'r') as f:
                pred_dict = json.load(f)
            circ = pred_dict['circle']
            for c in circ:
                [x, y, w, h]= calc_rect(c)
                rect = patches.Rectangle((x,y), w, h, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle="dashed")
                ax.add_patch(rect)
            rect.set_label('Est circle')

            tri = pred_dict['triangle']
            for t in tri:
                [x, y, w, h] = calc_rect(t)
                rect = patches.Rectangle((x,y), w, h, linewidth=1.5, edgecolor='y', facecolor='none', linestyle="dashed")
                ax.add_patch(rect)
            rect.set_label('Est triangle')

            ax.legend()
            if ttl:
                ax.set_title(ttl)
        if save:
            name = os.path.split(img)[-1]
            if ttl:
                name = ttl +'_'+name
            plt.savefig(os.path.join(os.path.realpath('.'), f'predictions_{name}'))


def eval_image(GT, pred, th=0.6, prnt=False):
    """"evaluates predicted BBs of an image for both shapes
            :param GT: path to GT json file with dictioanry of BB of shapes
            :type GT: string
            :param pred: json file with dictioanry of BB
            :type pred: string, of shapesparam Th: minimal threshold for iou excepted as TP
            :param ttl: title for displayed image
            :type ttl: string
            :return: tp, fp, missed, iou
            :rtype: ints for model evaluation
            """
    # read files
    with open(GT, 'r') as f:
        GT_dict = json.load(f)
    GTcirc = GT_dict['circle']
    GTtri = GT_dict['triangle']
    with open(pred, 'r') as f:
        pred_dict = json.load(f)
    predcirc = pred_dict['circle']
    predtri = pred_dict['triangle']
    # calculate TP, FP, Missed and IOU:
    tp_c, fp_c, missed_c, iou_c = conf(GTcirc, predcirc, th)
    tp_t, fp_t, missed_t, iou_t = conf(GTtri, predtri, th)
    if prnt:
        print(f'circle: TP ={tp_c}, FP = {fp_c}, missed = {missed_c}, IOU = {iou_c} ')
        print(f'triangle: TP ={tp_t}, FP = {fp_t}, missed = {missed_t}, IOU = {iou_t} ')
    return {'circle':[tp_c, fp_c, missed_c, iou_c],'triangle':[tp_t, fp_t, missed_t, iou_t] }

def precision_recall_curve(precision,recall, shape ,pred):
    """"save and display precision recall curve for a model's predictions for a certain shape"""
    name =os.path.split(pred)[-1]
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f'{name} for {shape}s : Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    name = os.path.split(pred)[-1]
    plt.savefig(os.path.join(os.path.realpath('.'),f'precision_recall_curve_{shape}_{name}.png'))
    # plt.show()

def IOU(boxA, boxB):
    """"returns Intersection over union of two Bounding boxes , iou=[0,1]"""
    # inputs format: [x_min,y_min, w,h]
    if (not ((boxB[0] <= boxA[0] <= boxB[0] + boxB[2]) or (boxA[0] <= boxB[0] <= boxA[0] + boxA[2]))):
        return 0.0
    # determine the (x, y)-coordinates of the intersection rectangle

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    # compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def conf(boxAList, boxBList, Th = 0.5):
    """calculates confusion metrics for BB shape detection (based on IOU)
        :param boxAList, boxBList: id to check
        :type boxAList, boxBList: list of ints
        param Th: minimal threshold for iou excepted as TP
        :type Th: int
        :return: tp, fp, missed, iou
        :rtype: ints for model evaluation

        """
    iou = []
    matches = {}
    tp = 0
    fp = len(boxBList)
    missed = len(boxAList)
    if not(boxBList): #no BB predicted
        iou = [0.0 for i in range(missed)]
        return tp, fp, missed, iou
    for i, A in enumerate(boxAList):
        iou_ = []
        # change format of BB (to min x,y and w,h)
        boxA = calc_rect(A)
        for B in boxBList:
            boxB = calc_rect(B)
            iouAB = IOU(boxA, boxB)
            iou_.append(iouAB)

        maxIou = max(iou_)
        maxIouIndex = iou_.index(max(iou_))
        iou.append(maxIou)
        # check if pred BB was already matched and matches better to new GT BB
        if (maxIouIndex in matches and maxIou > iou[matches[maxIouIndex]]):
            if (iou[matches[maxIouIndex]] > Th): # dont need to count new TP
                pass
            elif(maxIou > Th): #new pred BB is a TP
                tp += 1
                missed -= 1
                fp -= 1
            matches[maxIouIndex] = i
        if(not maxIouIndex in matches): # pred BB wasn't already matched to another GT BB
            matches[maxIouIndex] = i
            if(maxIou > Th ):
                tp += 1
                missed -= 1
                fp -= 1
    return tp, fp, missed, iou


