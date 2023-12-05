# LCA-CNN


### Please see https://github.com/greatroboticslab/finegrainedcascade/blob/main/LCA-CNN/README.md#using-our-customized-laser-dataset for its customization for our laser data.


The paper inspired by

Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction
without Convolutions

https://arxiv.org/abs/2102.12122

<img width="641" alt="image" src="https://github.com/greatroboticslab/finegrainedcascade/assets/205781/14fde0b4-cc89-4d0f-a417-d502ba5b834b">

Code for paper "Learning Cascade Attention for Fine-grained Image Classification", which currently 
under review at Elsevier Journal of Neural Networks(NN).


## File description
1. bootstrap.py  
Train/test split for CUB-200-2011 dataset.

2. image_rotate.py  
Rotate training set for data augmentation.

3. model.py  
Core model code.

4. inception_train.py  
Training and validation code.

5. inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5  
The iNaturalist pre-train parameters (converted from [Link](https://github.com/richardaecn/cvpr18-inaturalist-transfer)).

6. weights1/weights-gatp-two-stream-inception_v3-006-0.9080.hdf5  
Trained model for CUB-200-2011 dataset. (for reproduction).  
To unzip file:
```

 weights1.zip can be found here:
https://github.com/billzyx/LCA-CNN/blob/master/weights1.z01

zip -s- weights1.zip -O weights111.zip
unzip weights111.zip
```

## Dependencies:
+ Python (3.6)
+ Keras (2.1.5)
+ Tensorflow (1.10.0)


------

### Using our customized laser dataset:

step1. prepare the laser data using laser_rotate.py

step2. run laser_train.py for training.

step3. comment train_gatp_two_stream() & uncomment val() in the main function of  laser_train.py for inference.






