# Brain Tumor Detection Using MRI Data

## Project Overview

**Objective:** The primary objective of this project is to classify brain tumors using MRI data accurately. This involves differentiating various types of brain tumors to facilitate appropriate treatment decisions and prognostic assessments.

## Datasets

The project utilizes the following MRI image datasets:
- [Brain Tumor Classification Dataset on GitHub](https://github.com/sartajbhuvaji/brain-tumor-classification-dataset)
- [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Brain Tumors Dataset on Kaggle](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset)

## Machine Learning Model

### Algorithm
- **Transfer Learning and Fine-tuning:** Utilizes a pretrained model to adapt to the specific task of brain tumor classification. The model architecture is adapted to classify images into one of the following categories: Glioma, Meningioma, Pituitary tumor, or No tumor.

### Pretrained Models Used:
- **ResNet-50:** 50-layer Residual Network, adjusted for our specific classification tasks.
- **VGG16 and InceptionV3:** Models used as references from the literature, renowned for their effectiveness in image classification tasks.

## Implementation Details

1. **Data Augmentation:** Techniques like random resizing, cropping, and flipping are used to augment the training dataset to improve model robustness.
2. **Training:** The model is trained using PyTorch, with specific adjustments for learning rates and optimization strategies tailored to enhance performance on the tumor classification task.

## Results and Comparisons

The training accuracy achieved in this project surpasses that of the referenced papers, showcasing the efficacy of the applied methodologies and the potential of advanced neural networks in medical image analysis.

## Novelty of the Project

This project extends beyond binary classification (tumor or no tumor) to detailed multi-class classification, providing granular insights into the type of tumor present, which is pivotal for treatment planning.

## Conclusion

The "Brain Tumor Detection" project demonstrates significant potential in leveraging deep learning techniques for enhancing diagnostic accuracies in medical imaging, specifically for brain tumor classification through MRI scans.

## References

- Pillai, R., Sharma, A., Sharma, N., & Gupta, R. (2023). Brain Tumor Classification using VGG 16, ResNet50, and Inception V3 Transfer Learning Models. 2nd International Conference for Innovation in Technology, Bangalore, India.
