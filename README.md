# CORE: Multi-Link Graph Attention Network with Inter-Regional Collaboration for Continuous Sign Language Recognition
This repo is based on VAC (ICCV 2021)([code](https://github.com/VIPL-SLP/VAC_CSLR/)). Many thanks for this great work!
Our training and inference procedure is modified from VAC(ICCV 2021) 
# Requirements 
  Python (>3.7).
  
  Pytorch (>1.8).
  
  ctcdecode
  
  sclite
# Data Preparation
  ## PHOENIX2014 dataset
  RWTH-PHOENIX-Weather 2014 Dataset([download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/))

  Run the following code to resize original image to 256x256px for augmentation.

  `cd ./data/preprocess`
  
  `python data_preprocess.py --process-image --multiprocessing`

  ## PHOENIX2014-T dataset
  RWTH-PHOENIX-Weather 2014 Dataset([download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/))

  ## CSL-Daily dataset
  Request the CSL-Daily Dataset from this([paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.html))([download link](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/))

 To evaluate the model, run the code below：

 `python main.py --device your_device --load-weights path_to_weight.pt --phase test`
