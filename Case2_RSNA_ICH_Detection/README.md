## Case2 RSNA ICH Detection

## Requirement
- **Python version** : **3.7.3**
- Dataset : [link](https://drive.google.com/file/d/1sYatvaoqvgbWcvXWC2yeZYtmigt9uUB9/view?usp=sharing)
- `pip3 install -r requirement.txt`
- Run `Evaluation_main.py`

## Data pre-processing
- **Step 1** :  
  Convert **DICOM** to **JPEG** by using **Pydicom**
  
- **Step 2** :  
  Data augmentation : **Rotate** original images **90&deg;, 180&deg; and 270&deg;**
  
- **Step 3** :  
  Image Preprocess :
  - **CenterCrop** 512x512 
  - Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Traing and Testing
- **CNN model** :  
Resnet18, Resnet50, efficientnet-b0

- **Loss function**:  
nn.CrossEntropyLoss

- **Optimizer**:  
SGD

- **Traning Datasets** :  
19200 training samples after data augmentation (original training samples is 4800)

- **Datasets** :  
1200 for validation  

- **Learning_rate** = 0.0005, **momentum** = 0.9, **weight_decay** = 5e-4

- Use **pre-trained** model

## Training Results
 <img src="https://i.imgur.com/Bil1rnP.png" width = "900" height = "300" align=center />  
 
 - Validation set (N = 1200), **Prediction Accuracy: 71.8%**  
  <img src="https://i.imgur.com/LqOMeWm.png" width = "450" height = "400" align=center />  
  
 - Test set (N = 600), **Prediction Accuracy: 71.0%**
  <img src="https://i.imgur.com/LqOMeWm.png" width = "450" height = "400" align=center />  

## Resources
1. [DICOM PS3.10 2020d - Media Storage and File Format for Media Interchange](http://dicom.nema.org/medical/dicom/current/output/chtml/part10/chapter_7.html)  
2. [处理医疗影像的Python利器:PyDicom](https://zhuanlan.zhihu.com/p/59413289)  
3. [CNN相關要點介紹（二）——ResNet（殘差網絡）解析](https://www.twblogs.net/a/5c3a09b4bd9eee35b21db28c)  
4. [EfficientNet: Improving Accuracy and Efficiency through AutoML and Model Scaling](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)  

