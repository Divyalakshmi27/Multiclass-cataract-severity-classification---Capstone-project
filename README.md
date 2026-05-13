Multiclass Cataract Severity Classification
Multiclass Cataract Severity Classification using a hybrid deep learning ensemble framework combining EfficientNet-B0, Swin Transformer, and ConvNeXt Tiny for automated cataract severity prediction.
This project implements individual deep learning models along with a majority voting ensemble approach to improve classification performance on ophthalmic image datasets.

Models Used:
    EfficientNet-B0 
    Swin Transformer    
    ConvNeXt Tiny
    Majority Voting Ensemble



Project Structure
├── efficientnet_train.py
├── convnext_train.py
├── swin_transformer_train.py
├── ensemble.py
├── dataset/
├── results/
├── models/
└── README.md

Features:
  Image preprocessing and augmentation
  Deep learning–based multiclass cataract classification
  CNN and Transformer hybrid architecture approach
  Ensemble prediction using majority voting


Methodology:
   Input ophthalmic images are preprocessed and augmented.
   Images are trained separately using:
         EfficientNet-B0
         ConvNeXt Tiny
         Swin Transformer
   Predictions from all models are combined using majority voting.
   Final cataract severity class is generated based on ensemble output.


Technologies Used:
    Python
    PyTorch
    NumPy
    OpenCV
    scikit-learn
    torchvision
    timm



Dataset:
  Dataset is not included in this repository due to research and usage restrictions.
  Users may use their own cataract image dataset following a similar folder structure.

Usage:
  Train Individual Models (Eg: python Efficientnet_b0.py)
  Run Majority Voting Ensemble ( python Fusion.py)


Future Improvements:


Attention-based fusion techniques:
    Explainable AI integration (Grad-CAM)
    Real-time clinical deployment
    Lightweight edge-device optimization



Usage Notice:
    This project is shared for educational, research, and portfolio purposes only.
    Unauthorized commercial usage, reproduction, or redistribution of the code/methodology without permission is discouraged.

Author
Divyalakshmi L
