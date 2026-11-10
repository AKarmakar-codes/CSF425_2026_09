"""One deterministic RGB crop/resize contract shared by datasets and inference tests."""
import math
from PIL import Image

BASE = {'version': 'rgb-strict-floor-ceil-bilinear-v2', 'color': 'RGB',
        'bounds': 'reject outside image; floor left/top, ceil right/bottom; half-open',
        'resize': 'PIL bilinear square', 'mean': [.485, .456, .406], 'std': [.229, .224, .225],
        'cache_format': 'PNG original crop before resize; no fallback'}

def contract(img_size=224):
    if not isinstance(img_size, int) or img_size < 32:
        raise ValueError('img_size must be an integer >=32')
    return dict(BASE, img_size=img_size)

def bounds(bbox, size):
    if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
        raise ValueError('Box must contain four finite coordinates')
    x1, y1, x2, y2 = map(float, bbox)
    w, h = size
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f'Box outside image or degenerate: {bbox}, image {size}')
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)

def crop_rgb(image, bbox):
    return image.convert('RGB').crop(bounds(bbox, image.size))

def transform(img_size=224, is_train=False):
    from torchvision import transforms as T
    c = contract(img_size)
    ops = [T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if is_train:
        ops += [T.RandomHorizontalFlip(.5), T.RandomVerticalFlip(.5), T.RandomRotation(15),
                T.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.05)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(c['mean'], c['std'])])
