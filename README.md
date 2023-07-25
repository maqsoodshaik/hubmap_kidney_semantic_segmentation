# Hubmap_kidney_semantic_segmentation
## Task:
Segmenting instances of microvascular structures, including capillaries, arterioles, and venules. 
https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data
## Model:
Model is trained on 2D PAS-stained histology images from healthy human kidney tissue slides. U-net architecture is used for semantic segmentation.
Since labelled data is limited, Pretraining with random occlusions was utilized to achieve better performance in the downstream task.

## Target Mask
![TARGET MASK](https://github.com/maqsoodshaik/hubmap_kidney_semantic_segmentation/blob/main/target.png)

## Our Model Predicted Target Mask
![TARGET MASK](https://github.com/maqsoodshaik/hubmap_kidney_semantic_segmentation/blob/main/target_1.png)

Our trained model gave IoU of 0.7 on validation set
