# Salient-Object-Segmentation
PyTorch Implementaion of [U2-Net](https://arxiv.org/abs/2005.09007).

- Salient Object Detection(SOD) is a task based on visual attention mechanism, in which algorithm aims to focus on the prominent objects in an image.
- Even though, U2Net-small(~1.1M parameters) model is of 4.1 MB and trained for only 25 epochs, it achieves 88% accuracy in detecting correct pixels on [DUTS-TE dataset](http://saliencydetection.net/duts/#org3aad434).
- This architecture allows us to go deeper while maintaining the high-resolution feature maps at low memory and computational cost. Along with six U-Net-like encoders and decoders with different depths and residual connections(RSU blocks), It also contains convolutional layers with dilation rates of 1,2,4, and 8, which means the model intakes a lot of high-res contextual information, as it has a large receptive field. 

![arch](https://github.com/Vrushank264/Salient-Object-Segmentation/blob/main/architecture.jpg)

## Results:

Attention/Saliency map:</br>
![1](https://github.com/Vrushank264/Salient-Object-Segmentation/blob/main/Results/dog.png)
</br>Mask:</br>
![2](https://github.com/Vrushank264/Salient-Object-Segmentation/blob/main/Results/dog_mask1.png)
</br>
(Edge detection is not perfect yet because, model is trained for only 25 epochs.)


### Details:

1. This model is trained on [DUTS dataset](http://saliencydetection.net/duts/#org3aad434).
2. The model is optimized using `Binary Cross-Entropy Loss`.
3. All the training images were cropped to 320x320 and then resized to 288x288.
4. Learning rate is set to `1e-3`.
5. Trained model is available in `Trained model` directory.
 
