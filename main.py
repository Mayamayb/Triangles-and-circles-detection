import matplotlib.pyplot as plt
import numpy as np
from utils import *
import cv2
import os
import json
from detect import *


class Model:
    def __init__(self,img_path, gt_path, pred_path=None, iou_thresh = 0.5 ):
        """
        Define detection Model class
        by images, GT BB files, prediction BB and IOU threshold for defining GT.
        """
        self.imglist = self.files_list(img_path, 'jpg')
        self.gtlist = self.files_list(gt_path, 'json')
        if pred_path:
            self.predlist = self.files_list(pred_path, 'json')
        self.iou_th = iou_thresh

    def files_list(self, path, type='jpg'):
        # prepare list of absolute paths of files of certain type from folder path.
        flist = list()
        try:
            for fileName in os.listdir(path):
                if os.path.splitext(fileName)[1].casefold() != '.'+type:
                    continue
                else:
                    flist.append(os.path.join(os.path.realpath(path), fileName))
            # print('list of files created:\n', flist)
        except FileNotFoundError:
            print("No file or directory with the name {}".format(path))
            exit()
        return flist


    def model_eval(self, prnt=True):
        """"calcualtes model's recall and precision
        if prnt=True will print evaluation metrices and plot 10 images"""
        self.TP = {'circle': 0, 'triangle':0}
        self.FP = {'circle': 0, 'triangle':0}
        self.MISS = {'circle': 0, 'triangle':0}
        self.precision = {'circle': 0, 'triangle':0}
        self.recall = {'circle': 0, 'triangle':0}
        conf=[]
        precision_img={'circle': [], 'triangle': []}
        recall_img={'circle': [], 'triangle': []}
        F1Score_img={'circle': [], 'triangle': [],'together': []}
        tp = {'circle': 0, 'triangle': 0}
        fp = {'circle': 0, 'triangle': 0}
        miss = {'circle': 0, 'triangle': 0}

        for i, (img, GT, pred) in enumerate(zip(self.imglist, self.gtlist, self.predlist)):
            # print(f'Image {i}')
            conf.append(eval_image(GT, pred, self.iou_th))
            if prnt and i<10: # dont plot more than 20 images
                show_image(img, GT=GT, pred=pred)
            for shape in self.TP.keys():
                # for specific image and shape:
                [tp[shape],fp[shape],miss[shape]]=conf[i][shape][:3]
                # accumulate for later model(per shape) evaluation :
                self.TP[shape] += tp[shape]
                self.FP[shape] += fp[shape]
                self.MISS[shape] += miss[shape]
                if tp[shape]==0:
                    precision_img[shape].append(0.0)
                    recall_img[shape].append(0.0)
                    F1Score_img[shape].append(0.0)
                else:
                    precision_img[shape].append(tp[shape]/(tp[shape]+fp[shape]))
                    recall_img[shape].append(tp[shape]/(tp[shape]+miss[shape]))
                    F1Score_img[shape].append(2 * (precision_img[shape][i] * recall_img[shape][i]) / (precision_img[shape][i] + recall_img[shape][i]))

            # calc shared F1 score for finding topk results
            F1Score_img['together'].append(F1Score_img['circle'][i]*F1Score_img['triangle'][i])
        self.F1_img_list = F1Score_img['together']
        # print(F1Score_img['together'])
        for shape in self.TP.keys():
            str = f'\nEvaluation performance for {shape}s:\n'
            str += 'Total detections = {}/{}\nTotal False Positives = {}\nTotal missed = {}'.format(self.TP[shape],
                                                                                                          self.TP[shape] + self.MISS[shape],
                                                                                                          self.FP[shape],
                                                                                                          self.MISS[shape])
            if (self.TP[shape] > 0):
                self.precision[shape] = self.TP[shape] / (self.TP[shape] + self.FP[shape]) #= positive predictive value
                self.recall[shape] = self.TP[shape] / (self.TP[shape] + self.MISS[shape]) #=sensitivity



                # F1Score = 2 * (precision * recall) / (precision + recall)

                # extra metrics:
                # False_Positive_Rate = self.FP[shape] / (self.FP[shape] + self.TN[shape])
                # Specificity = 1-False_Positive_Rate

                str += '\nPrecision : {:.3f}\nRecall : {:.3f} \n'.format(self.precision[shape], self.recall[shape])


            if prnt:
                print(str)

        if prnt:
            plt.show()

    def topk(self, k=5 ):
        """" plot and save best and worst top k by F1 score of image, combining both shapes"""
        F1=self.F1_img_list
        assert len(F1)>k
        topk_i = sorted(range(len(F1)), key=lambda i: F1[i])[-k:]
        bottomk_i = sorted(range(len(F1)), key=lambda i: F1[i])[:k]
        for i,(t,b) in enumerate(zip(topk_i, bottomk_i)):
            # show best
            show_image(self.imglist[t], GT=self.gtlist[t], pred=self.predlist[t], ttl = f'Best F1={F1[t]:.2f}', save=True)
            # show worst
            show_image(self.imglist[b], GT=self.gtlist[b], pred=self.predlist[b], ttl = f'Worst F1={F1[b]:.2f}', save=True)
        plt.show()



    def make_predictions(self,pred_path, prnt=False):
        """"make prediction files to new directory
        input: pred_path: str of obsolute path
        """
        os.makedirs(pred_path, exist_ok = True)
        for i, img in enumerate(self.imglist):
            pred_dict = {'circle': [], 'triangle': []}
            print(f'Image {i}')
            pred_dict['circle'] = find_circ_BB(img, prnt)
            pred_dict['triangle'] = find_tri_BB(img, prnt)
            plt.show()
            name=os.path.basename(img)[:-4] + '.json'
            with open(os.path.join(pred_path,name),  'w') as outfile:
                json.dump(pred_dict, outfile,  default=convert)

def convert(o):
    # saving bb file format (to json)
    if isinstance(o, np.generic): return o.item()
    raise TypeError

def main(img_path, gt_path, pred_path):
    model1=Model(img_path, gt_path, pred_path, iou_thresh=0.3)
    model1.model_eval(prnt=True)
    new_pred_path = pred_path + '_new'
    # model1.make_predictions(new_pred_path)
    model2=Model(img_path, gt_path, new_pred_path, iou_thresh=0.3)
    model2.model_eval(prnt=True)

    model2.topk()
    # calculate precision recall curve (different confidence thesholds are tested)
    th_list = np.arange(0.0, 1.0,0.05).tolist()
    for pred in [new_pred_path, pred_path]: # for each model prediction
        model = list()
        precision_rate = {'circle': [], 'triangle': []}
        recall_rate = {'circle': [], 'triangle': []}
        for i, th in enumerate(th_list): # test different iuo thresholds
            # print(f"for threshold = {th}")
            model.append(Model(img_path, gt_path, pred, iou_thresh=th))
            model[i].model_eval(prnt=False)
            for shape in precision_rate.keys():
                precision_rate[shape].append(model[i].precision[shape])
                recall_rate[shape].append(model[i].recall[shape])
        plt.close()
        for shape in precision_rate.keys():
            print(f'{shape} : precision_rate (for {pred})')
            print(precision_rate[shape])
            print(f'{shape} : recall_rate (for {pred})')
            print(recall_rate[shape])

            precision_recall_curve(precision_rate[shape],recall_rate[shape], shape, pred )

    plt.show()




if __name__ == "__main__":
    imp_path =  os.path.join(os.path.realpath('.'), 'data','img')
    gt_path = os.path.join(os.path.realpath('.'), 'data','ground_truth')
    pred_path = os.path.join(os.path.realpath('.'), 'data','prediction')
    main(imp_path, gt_path,pred_path )