import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import os
import os.path as osp
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image

import sys
import torch

sys.path.insert(0, './')
from datasets import *
import models as custom_models
from timm.models.helpers import load_checkpoint


model = custom_models.pdcnet(variant='s1', num_classes=9, inference_mode=False)
# checkpoint = './SAAS/apple_leaf/pdcnet_epoch200_resize_lr0.05_pdam/model_best.pth.tar'
checkpoint = './SAAS/apple_leaf/pdcnet_epoch100_resize_lr0.05_noattn/model_best.pth.tar'
load_checkpoint(model, checkpoint, strict=True)
model.eval()
model = custom_models.reparameterize_model(model)
model.inference_mode = True

# NOTE: requires_grad == True when using cam
for param in model.parameters():
    param.requires_grad = True

model.eval()

# Loop for visualization 

CLASS_LIST = ['苹果健叶2021', '苹果白粉病', '苹果斑点落叶病', '苹果花叶病', '苹果缺素（黄化）', 
                '苹果小叶病20220604', '苹果叶片褐斑病', '苹果叶片炭疽叶枯病', '苹果叶片锈病']
DATA_ROOT = 'E:/dataset/果园病害图像_resize'
DST_ROOT = './visualize_res_noattn'
os.makedirs(DST_ROOT, exist_ok=True)

for class_idx, class_name in enumerate(CLASS_LIST):
    class_path = osp.join(DATA_ROOT, class_name)
    os.makedirs(osp.join(DST_ROOT, class_name), exist_ok=True)
    for image_name in os.listdir(class_path):
        image_path = osp.join(class_path, image_name)

        # image_path = "E:/dataset/果园病害图像_resize/苹果斑点落叶病/IMG_20211101_154445.jpg"
        img = np.array(Image.open(image_path))
        img = cv2.resize(img, (224, 224))
        img = np.float32(img) / 255
        input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # The target for the CAM is the Bear category.
        # As usual for classication, the target is the logit output
        # before softmax, for that category.
        targets = [ClassifierOutputTarget(class_idx)]
        target_layers = [model.stage4]
        # cam = GradRepCAM(model=model, target_layers=target_layers)
        # grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        images = np.hstack((np.uint8(255*img), cam , cam_image))
        res = Image.fromarray(images)

        res.save(osp.join(DST_ROOT, class_name, image_name))



'''
For Debug

import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from typing import Callable, List, Tuple
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image


class RepActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        # print('???')
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            # print('no requires_grad')
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradRepCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradRepCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)
        self.activations_and_grads = RepActivationsAndGradients(self.model, target_layers, reshape_transform)

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        # print(self.activations_and_grads.gradients)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
            # print(grads_list)
            # print('layer_activations', layer_grads)

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # print(grads)
        return np.mean(grads, axis=(2, 3))


'''
