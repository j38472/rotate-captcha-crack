from pathlib import Path
from typing import Protocol, Sequence, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F

from .helper import DEFAULT_NORM, rand_angles, rotate, to_square


class TypeGetImg(Protocol):
    """
    Methods:

        `def __len__(self) -> int:` length of the dataset

        `def __getitem__(self, idx: int) -> Tensor:` get img in tensor ([C,H,W]=undefined, dtype=float32, range=[0,1])
    """

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Tensor:
        pass


TypeRCCItem = Tuple[Tensor, Tensor]


class GetImgFromPaths(TypeGetImg):
    """
    Args:
        img_paths (Sequence[Path]): sequence of paths. Each path points to an img file
        _range (Sequence[Path]): select which part of sequence. Use (0.0,0.5) to select the first half
        norm (Normalize): normalize policy

    Methods:

        `def __len__(self) -> int:` length of the dataset

        `def __getitem__(self, idx: int) -> Tensor:` get img in tensor ([C,H,W]=[3,ud,ud], dtype=float32, range=[0,1])
    """

    __slots__ = [
        'img_paths',
        'length',
        'norm',
    ]

    def __init__(self, img_paths: Sequence[Path], _range: Tuple[float, float], norm: Normalize = DEFAULT_NORM) -> None:
        length = len(img_paths)
        start = int(_range[0] * length)
        end = int(_range[1] * length)

        self.img_paths = img_paths[start:end]
        self.length = end - start

        self.norm = norm

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tensor:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_ts = F.to_tensor(img)
        img_ts = self.norm(img_ts)
        return img_ts


class RCCDataset(Dataset[TypeRCCItem]):
    """
    dataset for RCC

    Args:
        getimg (TypeGetImg): upstream dataset
    """

    __slots__ = [
        'getimg',
        'angles',
    ]

    def __init__(self, getimg: TypeGetImg) -> None:
        self.getimg = getimg
        self.angles = rand_angles(len(getimg))

    def __len__(self) -> int:
        return self.getimg.__len__()

    def __getitem__(self, idx: int) -> TypeRCCItem:
        angle = self.angles[idx]

        img_ts = self.getimg[idx]
        img_ts = to_square(img_ts)
        img_ts = rotate(img_ts, angle.item() * 360)

        return img_ts, angle
