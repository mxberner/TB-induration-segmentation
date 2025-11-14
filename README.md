# TST Segmentation & Classification
A computer vision project for automated segmentation, measurement, and classification of Mantoux Tuberculin Skin Test (TST) induration sites.  
Our project aims to reduce inter-reader variability by providing consistent and objective measurement tools using modern segmentation model, inlcuidng automatic segmentation and measurement of the induration and a stretch goal of identifying positive/negative LTBI outcomes.

---

## Datasets

### Real-World Images
- 50+ Real-world clinical images collected as part of a JHU CBID project  

### Modelling Clay Induration Dataset
- Synthetic clay molds mimicking 5mm, 10mm, and 15mm indurations  
- Previously used successfully in research for training segmentation models  

### PAD-UFES-20 Dataset
- 2,298 annotated skin lesion images 
- Useful for feature pretraining due to similarity in surface texture and skin variability  
- Reference: Pacheco et al., 2020

---

## Proposed Methods
**Segmentation Models Under Evaluation**
- SAM 2.1
- MedSAM 
- U-Net / U-Net++
- DeepLabV3+ 
- Mask R-CNN  
- YOLOv11-seg


**Preprocessing & Classical CV Techniques**
- Canny edge detection  
- Histogram equalization  
- Skin-tone normalization  
- Illumination correction  
- Channel-wise thresholding  
- Optional 3D/depth cues using ARCore Depth API

