from torchvision import transforms


def build_weak_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomRotation(8),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def build_strong_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.163, contrast=0.163),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def build_train_transform() -> transforms.Compose:
    return build_strong_train_transform()


def build_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
