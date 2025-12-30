# Skin_disease_detection
Skin cancer is a major and growing global health concern, with malignant melanoma being the most dangerous form. Early detection is critical for improving patient outcomes, yet conventional diagnosis through dermoscopic image inspection is subjective and depends heavily on a dermatologist’s expertise.The subtle visual differences between benign and malignant lesions often make reliable diagnosis challenging, highlighting the need for standardized, computer-assisted tools.


Deep learning, particularly Convolutional Neural Network (CNNs), offers a promising solution by automatically learning patterns from medical images. CNNs can identify complex features in dermoscopic data and provide rapid, objective predictions, making them an effective decision-support tool that may reduce[4] unnecessary biopsies and increase diagnostic confidence.

In this work a CNN-based framework has been proposed for automated skin lesion classification using the HAM10000 dataset, which includes seven[13] types of skin lesions. A key challenge of this dataset is its class imbalance, with rare lesion types underrepresented. To address this, we applied data resampling techniques to ensure balanced learning. Our custom CNN architecture integrates convolutional,[6] pooling, and normalization layers to capture hierarchical[9] image features. The model’s performance was evaluated using accuracy, loss, and a confusion matrix to assess classification reliability across all lesion classes.

The results demonstrate that CNN-based systems, when paired with proper data balancing, can serve as accurate and accessible diagnostic aids in dermatology.

