from torchvision.models import resnet18, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, mobilenet_v2
from thop import profile
from thop import clever_format

import torch
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, convert_sync_batchnorm, model_parameters

import sys

sys.path.insert(0, './')
import models as custom_models

# model = shufflenet_v2_x2_0(num_classes=9)

model = custom_models.pdcnet(variant='s1', num_classes=9)

# model = create_model(
#     'mixnet_s',
#     pretrained=False,
#     num_classes=9)

input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
print(macs, params)
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)