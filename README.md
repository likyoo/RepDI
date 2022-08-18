# RepDI: A Light-weight CPU Network for Apple Leaf Disease Identification


train:  
```
python train.py E:/dataset/果园病害图像_resize/ --model pdcnet --num-classes 9 --lr 0.05 --epochs 100 --warmup-epochs 5 --cooldown-epochs 0 --weight-decay 1e-4 --sched cosine -b 32 --input-size 3 224 224  --output ./SAAS/apple_leaf --experiment pdcnet_epoch100_resize_lr0.05_pdam
```
infer:
```
python inference.py E:/dataset/果园病害图像/ --model pdcnet --num-classes 9 -b 32 --input-size 3 224 224  --output_dir ./SAAS/apple_leaf --checkpoint ./SAAS/apple_leaf/pdcnet_epoch200_resize_lr0.05_pdam/last.pth.tar

```
