# Inception-U-Net-model
The model is developed from our paper "Inception U-Net Architecture for Semantic Segmentation to Identify Nuclei in Microscopy Cell Images".

You can visualize the training logs using tensorboard.

This code requires:
- Tensorflow: 1.12.0
- Keras: 2.2.4

# What is included?
- IU-Net model.
- Segmentation loss function defined as the average of binary cross entropy loss, dice coefficient loss and intersection-over-union loss.
- More precise and better implementation of IoU metric .

# Citation
```
@article{punn2020inception,
  title={Inception u-net architecture for semantic segmentation to identify nuclei in microscopy cell images},
  author={Punn, Narinder Singh and Agarwal, Sonali},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  volume={16},
  number={1},
  pages={1--15},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```
