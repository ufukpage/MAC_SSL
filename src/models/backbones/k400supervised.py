from .pytorch_imp import r2plus1d_18, r3d_18


from .base_backbone import BaseBackbone
from ...builder import BACKBONES


@BACKBONES.register_module()
class SupervisedR2Plus1D(BaseBackbone):
    def __init__(self, pretrained=True, return_conv=True, *args, **kwargs):
        super(SupervisedR2Plus1D, self).__init__()
        self.model = r2plus1d_18(pretrained=pretrained, progress=True, return_conv=return_conv)

    def init_weights(self):
        pass

    def forward(self, x):
        return self.model(x)


@BACKBONES.register_module()
class SupervisedR3D(BaseBackbone):
    def __init__(self, pretrained=True, return_conv=True, *args, **kwargs):
        super(SupervisedR3D, self).__init__()
        self.model = r3d_18(pretrained=pretrained, progress=True, return_conv=return_conv)

    def init_weights(self):
        pass

    def forward(self, x):
        return self.model(x)

