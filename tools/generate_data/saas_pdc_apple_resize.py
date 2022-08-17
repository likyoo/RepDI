"""
自采数据的分辨率太大,影响训练时数据加载的效率

"""

import json
import os
import os.path as osp

import cv2
import numpy as np


if __name__ == '__main__':
    src_root = 'E:/dataset/果园病害图像'
    dst_root = 'E:/dataset/果园病害图像_resize'
    p = 4

    for cls in os.listdir(src_root):

        src_cls_dir = osp.join(src_root, cls)
        if not osp.isdir(src_cls_dir): continue
        print(cls)

        dst_cls_dir = osp.join(dst_root, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        for img_file in os.listdir(src_cls_dir):
            img_path = osp.join(src_cls_dir, img_file)
            # img = cv2.imread(img_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            dst_h, dst_w = h // p, w // p
            img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(osp.join(dst_cls_dir, img_file), img)
            cv2.imencode('.jpg', img)[1].tofile(osp.join(dst_cls_dir, img_file))
        