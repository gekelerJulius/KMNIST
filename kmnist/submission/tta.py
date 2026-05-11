import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from CONFIG import SUBMISSION


def tta_specs(
    rotation_degrees: tuple[float, ...] = SUBMISSION.tta_rotation_degrees,
    translate_pixels: tuple[tuple[int, int], ...] = SUBMISSION.tta_translate_pixels,
) -> tuple[tuple[float, tuple[int, int]], ...]:
    if len(rotation_degrees) != len(translate_pixels):
        raise ValueError(
            "SUBMISSION.tta_rotation_degrees and SUBMISSION.tta_translate_pixels "
            "must contain the same number of entries."
        )
    return tuple(zip(rotation_degrees, translate_pixels))


def tta_image_batches(
    images: torch.Tensor,
    use_tta: bool = SUBMISSION.use_tta,
) -> list[torch.Tensor]:
    if not use_tta:
        return [images]

    batches = []
    for angle, translate in tta_specs():
        if angle == 0.0 and translate == (0, 0):
            batches.append(images)
            continue
        batches.append(
            F.affine(
                images,
                angle=angle,
                translate=list(translate),
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=-1.0,
            )
        )
    return batches
