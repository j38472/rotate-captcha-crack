import torch.nn as nn
from torch import Tensor

from ..const import DEFAULT_CLS_NUM
from ._unireplk import unireplknet_a


class RotNetR(nn.Module):
    """
    Args:
        train (bool, optional): True to load the pretrained parameters. Defaults to True.

    Note:
        impl: `rotnet_street_view_resnet50` in https://github.com/d4nst/RotNet but with `RegNet_Y_3_2GF` as its backbone
    """

    def __init__(self, cls_num: int = DEFAULT_CLS_NUM, train: bool = True) -> None:
        super().__init__()

        self.cls_num = cls_num

        self.backbone = unireplknet_a(in_1k_pretrained=train)

        fc_channels = self.backbone.head.in_features
        del self.backbone.head
        self.backbone.head = nn.Linear(fc_channels, cls_num)

        if train:
            nn.init.kaiming_normal_(self.backbone.head.weight)
            nn.init.zeros_(self.backbone.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): img_tensor ([N,C,H,W]=[batch_size,3,224,224], dtype=float32, range=[0.0,1.0))

        Returns:
            Tensor: predict result ([N,C]=[batch_size,cls_num), dtype=float32, range=[0.0,1.0))
        """

        x = self.backbone.forward(x)

        return x

    def predict(self, img_ts: Tensor) -> float:
        """
        Predict the counter clockwise rotation angle.

        Args:
            img_ts (Tensor): img_tensor ([C,H,W]=[3,224,224], dtype=float32, range=[0.0,1.0))

        Returns:
            float: predict result. range=[0.0,1.0)

        Note:
            Use Image.rotate(-ret * 360) to recover the image.
        """

        img_ts = img_ts.unsqueeze_(0)

        onehot_ts = self.backbone.forward(img_ts)
        angle = float(onehot_ts.cpu().argmax(1).item()) / self.cls_num

        return angle
