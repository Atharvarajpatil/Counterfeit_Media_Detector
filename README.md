## Counterfeit_Media_Detector

## Challenge details:

### Fake detection articles
- The Deepfake Detection Challenge (DFDC) Preview Dataset
- Deep Fake Image Detection Based on Pairwise Learning
- DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection
- DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection
- Real or Fake? Spoofing State-Of-The-Art Face Synthesis Detection Systems
- CNN-generated images are surprisingly easy to spot... for now
- FakeSpotter: A Simple yet Robust Baseline for Spotting AI-Synthesized Fake Faces
- FakeLocator: Robust Localization of GAN-Based Face Manipulations via Semantic Segmentation Networks with Bells and Whistles
- Media Forensics and DeepFakes: an overview
- Face X-ray for More General Face Forgery Detection

## Solution description
In general, the solution is based on a frame-by-frame classification approach. Other complex methods did not work as well on the public leaderboard.

### Face-Detector
MTCNN detector was chosen due to kernel time limits. A more precise and robust S3FD detector would be better, but open-source Pytorch implementations lack a license.

Input size for the face detector was calculated for each video depending on video resolution:

- 2x scale for videos with a width of less than 300 pixels
- No rescale for videos with a width between 300 and 1000 pixels
- 0.5x scale for videos wider than 1000 pixels
- 0.33x scale for videos wider than 1900 pixels

### Input size
EfficientNets significantly outperform other encoders, so they were used in the solution. B4 was the starting point, and "native" size (380x380) was chosen for that network. Due to memory constraints, the input size was not increased for the B7 encoder.

### Margin
When generating crops for training, 30% of the face crop size was added from each side. This setting was used throughout the competition. See the `extract_crops.py` file for details.

### Encoders
The winning encoder is EfficientNet B7, pretrained with ImageNet and noisy student. "Self-training with Noisy Student improves ImageNet classification."

### Averaging predictions
32 frames were used for each video. Instead of simple averaging, the following heuristic was applied, which performed well on the public leaderboard:

```python
import numpy as np

def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)

### Augmentations

Heavy augmentations were used by default. The Albumentations library supports most of the augmentations out of the box. The IsotropicResize augmentation was added.

```python
def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ])
