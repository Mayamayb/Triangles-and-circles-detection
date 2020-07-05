# Triangles-and-circles-detection
find triangles and circles in images

## Structure
### Poject consist of files:
1. `main.py` : consists of main loop and model class. It is most suitable to modify 'Main' function for running relevant code. the configarations are set to calculate 
  precision recall curve and plot top5 (best and worst prediction images)
2. `detect.py`: consists of functions for detecting the shapes. `main.py` imports from it
3. `utils.py` : consists of functions used in project.  `main.py`,`detect.py` imports from it 

## How to Test

### Display an image
1. Run `utils.show_image(img[,GT[,pred[, ttl[,save]]]])` with paths to files (GT, pred, ttl are optional)
### Evaluate image
1. Run `utils.eval_image(GT, pred [,th [,prnt]])` with paths to files 
2. This will return evaluation of image 
### Evaluate Model Quality
1. Run `model=Model(img_path, gt_path [,pred_path[, iou_thresh]])` to define model.
2. Run `model.make_predictions(new_pred_path)` to prepare new predictions.
3. Run `model.model_eval([prnt=False])` to evaluate model based on precision&recall
4. Run `utils.precision_recall_curve(precision_rate[shape],recall_rate[shape], shape, pred )` to plot precision-recall curve, with precision_rate, recall_rate  calculates as in the 'Main' function.
5. Run `model.topk()` to plot best and worst model prediction images based on F1_circlesxF1_triangles  (after running model.eval) 

### Results 
    `model.make_predictions(new_pred_path)
    model=Model(img_path, gt_path, new_pred_path, iou_thresh=0.3)
    model.model_eval(prnt=True)`
    
 PRINTED:
 
    Evaluation performance for circles:
    Total detections = 412/1441
    Total False Positives = 1059
    Total missed = 1029
    Precision : 0.280
    Recall : 0.286 


    Evaluation performance for triangles:
    Total detections = 886/2395
    Total False Positives = 2941
    Total missed = 1509
    Precision : 0.232
    Recall : 0.370 
#### example image plotted with my predictions:

![result image](https://github.com/Mayamayb/Triangles-and-circles-detection/blob/master/predictions_Best%20F1%3D0.75_075d44c6-49c1-48ab-a77c-0db3fdb80973.jpg?raw=true)

#### example image plotted with given predictions:
![result image2](https://github.com/Mayamayb/Triangles-and-circles-detection/blob/master/predictions_Best%20F1%3D1.00_1214f4de-8399-48ea-a2f6-6e31843bc3da.jpg?raw=true)

### Graphs:
### precision-recall curve:
#### given circle predictions evaluation:

![result image3]( https://github.com/Mayamayb/Triangles-and-circles-detection/blob/master/precision_recall_curve_circle_prediction.png?raw=true)

#### given triangles predictions evaluation:
![result image4]( https://github.com/Mayamayb/Triangles-and-circles-detection/blob/master/precision_recall_curve_triangle_prediction.png?raw=true)

#### my model's circle predictions evaluation:

![result image5]( https://github.com/Mayamayb/Triangles-and-circles-detection/blob/master/precision_recall_curve_circle_prediction_new.png?raw=true)

#### my model's triangles predictions evaluation:
![result image6]( https://github.com/Mayamayb/Triangles-and-circles-detection/blob/master/precision_recall_curve_triangle_prediction_new.png?raw=true)




