import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """
    Training augmentations for fundus images.
    Includes geometric, color, and noise augmentations tuned for retinal imaging.
    Compatible with albumentations v2.0+.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=180,
            border_mode=0,
            p=0.5,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05,
            p=0.5,
        ),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Validation/inference transforms. Minimal processing: resize + normalize.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 224, n_augments: int = 5) -> list:
    """
    Test-time augmentation transforms.
    Returns a list of transform pipelines.
    """
    base = [
        A.Resize(height=img_size, width=img_size, interpolation=2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    tta_pipelines = [
        # 0: original
        A.Compose(base),
        # 1: horizontal flip
        A.Compose([A.Resize(height=img_size, width=img_size, interpolation=2),
                   A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                   ToTensorV2()]),
        # 2: vertical flip
        A.Compose([A.Resize(height=img_size, width=img_size, interpolation=2),
                   A.VerticalFlip(p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                   ToTensorV2()]),
        # 3: rotate 90
        A.Compose([A.Resize(height=img_size, width=img_size, interpolation=2),
                   A.Rotate(limit=(90, 90), border_mode=0, p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                   ToTensorV2()]),
        # 4: rotate 90 + hflip
        A.Compose([A.Resize(height=img_size, width=img_size, interpolation=2),
                   A.Rotate(limit=(90, 90), border_mode=0, p=1.0),
                   A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                   ToTensorV2()]),
    ]

    return tta_pipelines[:n_augments]
