# Evolutionary Computation Experiment  
This project is an implement to use **genetic algorithm(GA)** to optimize hyper-parameters in deep learning networks.  

## Needed Environment  
- Ubuntu 22.04.3 LTS
- Python 3.10.2
- Pytorch 2.1.0
- Cuda 12.1
- Torchvision 0.16
## Optimized Parameters List:  
- **nb_node** - the number of convolution kernels in each layer  
- **lr**      - initial learning rate  
- **batch_size** - the number of photos in each batch  
- **kernel_size** - the  size of convolution  kernel

There are much more parameters can be optimized with GA, but my computer only has 6G GPU memory and I have no time to try other parameters and networks.  

## Dataset And Network


We used VGGNet as the baseline for classification. Confined by computer, I only placed two convolution layer.  

[Cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html) is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.  

[VGGNet](https://miro.medium.com/v2/resize:fit:720/format:webp/1*NNifzsJ7tD2kAfBXt3AzEg.png) is a deep convolutional neural network that was proposed by Karen Simonyan and Andrew Zisserman.  
It can be used in classification.