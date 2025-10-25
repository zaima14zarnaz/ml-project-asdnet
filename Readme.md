The project is based on the paper: Inferring Attention Shift Ranks of Objects for Image Saliency [CVPR 2020].
Paper link: [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Siris_Inferring_Attention_Shift_Ranks_of_Objects_for_Image_Saliency_CVPR_2020_paper.pdf)


## Installation
The code is based on the Mask-RCNN implementation by matterport, https://github.com/matterport/Mask_RCNN. Please follow and install the requirements they list. Additionally, install pycocotools.   

## Dataset
Download ASSR dataset from [google drive](https://drive.google.com/file/d/1ueSpf3avLAPiJxoP40v5KL7qxaYtM1us/view?usp=sharing).
Download IRSR daraser from [google drive](https://github.com/dragonlee258079/Saliency-Ranking/tree/9fd1cd5b919f629ea044a4112baa0919b6f663ac)
Download SIFR dataset from [google drive](https://drive.google.com/file/d/1Gop2GtVQI5ZND-npBo_yp2brU_hPmdKZ/view)

## Usage
Training may take a long time, so download the pre-trained weights from the directories specified below to evaluate the model on the dataset instead. 

## Training 
The current implementation and results are based on pre-computing backbone and object features, then training the rest of the saliency rank model seperately. 

1. Pre-train backbone for salient object detection (binary, no ranking). Download pre-trained COCO weights (mask_rcnn_coco.h5) from matterport, https://github.com/matterport/Mask_RCNN/releases. Create a new "weights/" folder in the root directory, then put the weight file inside it. Set data paths and run:
```
python obj_sal_seg_branch/train.py
```

2. Pre-compute backbone and object features of GT objects for "train" and "val" datasplits. Set data paths and run twice for "train" and "val":
```
python pre_process/pre_process_obj_feat_GT.py
```

3.  Finally train the saliency rank model. Set data paths and run:
```
python train.py
```

## Testing
Similarly to training, test set image features need to be pre-computed.

1. Perform object detection. You can download the pre-trained weights from [google drive](https://drive.google.com/file/d/1A86orvY1O4G_HjEGgtvZ7kRwqsf0mIb8/view?usp=sharing). 
Set data paths and run:
```
python pre_process/object_detection.py
```

2. Pre-compute corresponding object features. Set data paths and run:
```
python pre_process/pre_process_obj_feat.py
```

3. Test the saliency rank model. The weights of the trained model from [google drive](https://drive.google.com/file/d/1fXFGvrS7aMd5FagM9n7-VaJPxRrPXbXN/view?usp=sharing).
Set data paths and run:
```
python evaluation/predict.py
```
This will generate predictions and save into files.

4. Generate predicted Saliency Rank Maps (rank based on grayscale value). Set data paths and run:
```
python evaluation/generate_saliency_map.py
```

## Evaluate
To evaluate MAE and Salient Object Ranking (SOR) scores, set data paths and run:
```
python evaluation/evaluate.py
```
