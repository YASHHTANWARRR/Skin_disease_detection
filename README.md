# Skin_disease_detection
Skin cancer is a severe and rising problem across the globe. In this case, melanoma is considered the most lethal type of cancer. Early diagnosis of this disease could significantly help patients recover better. Unfortunately, conventional techniques of detecting skin cancers based on dermoscopic images are highly subjective and require considerable skills from dermatologists due to similarities between benign and malignant skin lesions.

Deep learning techniques, especially those using Convolutional Neural Networks (CNNs), offer a promising approach to solving the problem of accurate detection of skin cancers. Specifically, CNNs enable automatic learning of hierarchies of features from images, thus providing fast and objective results in diagnostics and acting as a supporting diagnostic tool.

This paper presents a new framework for the automated classification of skin lesions with CNN. The study relies on the HAM10000 dataset with seven different skin lesion types. One of the difficulties with this dataset is its class imbalance. To solve this issue, we used data resampling approaches for achieving more balanced learning.

Model and Methodology
Our framework employs convolutional, pooling, and normalization layers designed to identify image features in various levels of detail. Evaluation of model efficiency is performed through measuring accuracy, loss, and confusion matrices.

In order to increase the efficiency of computations, GPU acceleration using CUDA technology created by NVIDIA was used in the research, allowing for more rapid training due to parallelization in deep learning computations. Moreover, the use of RAPIDS was introduced to improve the speed of computations with the help of GPU acceleration in data processing.

As it follows from the presented outcomes, neural network models based on convolutional layers are able to deliver efficient, accurate, and scalable models to detect skin diseases, thus being very useful in practice.

