
<h1 align = "center">
    Homework Assignment 4 : Neural Style Transfer  
</h1>

---

<h1>Table of Content</h1>

- [Introduction](#introduction)
- [Approach](#Approach)
- [Results](#Results)
- [Platform](#Platform)
- [Installation guidelines](#Installation-guidelines)
- [References](#References)
- [Contributors](#Contributors)

## Introduction
---    
In the past, for creating an image in a particular style and form, people used to have need of specialised artist and spend lot of time and money for that. Since last few decades, even computer researchers have taken a keen interest in this field and have developed techniques like non-photorealistic rendering(NPR) which can only be used for a limited number of styles, image analogies for a style transfer by differentiating the stylized and unstylized images.The most recent method of neural style transfer using convolutional neural network is the most prominent one. Neural Style Transfer(NST) works on the idea that a CNN can separate the style and content representation within an image during a task(specifically computer vision task). By taking in two input images, style image and content image and then forming an output image that has the content of the content image but the style of the style image. 

## Approach
---
For fast and easy execution of the models' operations; eager execution command has been executed to save the time in running the models. This also helps in debugging the code faster. 
- Layers used and details about layers<br/>
For fetching the content and the style features from the input images, the results of the intermediate layers are used. The feature maps become complex as we move from lower to higher order layers in the network.

- Content Feature Extraction<br/>
As we move deeper into the network, the features extracted from the images contain more information about the content within the image as compared to the details in the pixel values. So, mainly the features extracted from the higher layers are used as a content representation required for the formation of the output image. Usually higher layer from the last set of convolutional layers is used as a content extraction.

- Style Feature Extraction<br/>
For style representation used in the formation of the output image, the correlation among the different feature maps from the kernel results from the convolutional layers is done which can take out the texture information(local details) without taking out the global details. These correlations are known as gram matrices. Usually the first layer in every set of convolutional layers are used for correlation with varying weight for each layer. This can help in varying the style levels. 

- Architectures implemented<br/>
For extracting the feature maps for extracting style and content representation use of the pre-trained model is done. The output of the model is used in for fetching the layers for content and style extraction. Here, we have tried implementing the following architectures:-
    - VGG-16 architecture<br/>
    - VGG-19 architecture<br/>
    - ResNet50 architecture<br/>

- Loss function and optimisation<br/>
    - Content Loss Function<br/>
In order to make sure that the difference between the content of the generated output image and the content image is minimised, the content loss function is defined using the Minimum Square Error(MSE) loss.
<p align="center">
<img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/content_loss_function.JPG" style="vertical-align:middle;margin:50px 50px">
 </p>
    - Style Loss Function<br/>
In order to make sure that the difference between the texture of the generated output image and the style image is minimised, the style loss function is defined using gram matrices concept.<br/>
Gram matrix is a matrix which is used in the calculation of the correlation between the channels of the same convolutional layer used for style features extraction. The output suggests how much degree of correlation is there within the channels with respect to each other.<br/>
The gram matrices of style image and generated output image of same layers are compared using square difference function to minimise the loss function. Defining the gram matrices loss:<br/>
<p align="center">
<img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/style_loss_function.JPG" style="vertical-align:middle;margin:50px 50px">
 </p>
As there are multiple layers involved in extracting the style and in the loss function, weights are assigned to loss function of every layer which finally gives the style loss function.<br/>
<p align="center">
<img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/style_loss_final_function.JPG" style="vertical-align:middle;margin:50px 50px">
 </p>
    - Complete Loss Function<br/>
In order to make sure that the generated output image is similar to the content image in terms of their content and to style image in terms of their style and not the complete style image, loss function is introduced in the implementation with two separate parts content loss and style loss for calculating loss. With every iteration, the goal is to minimise the overall loss function so the we get the desired output image.<br/> 
<p align="center">
<img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/total_loss_function.JPG" style="vertical-align:middle;margin:50px 50px">
 </p>
The parameters alpha and beta act as weights for controlling the amount of content and style features to be added into the generated output image.<br/>
    - Gradient Descent for optimisation<br/>
Using gradient descent approach for minimising the loss is generally done which can help in generating more informative output image. Here, we have implemented the Adam optimiser which can be used for backpropagation which update the hyperparameters after every iteration and optimises the loss function.


## Results
---
The aim of developing the neural style transfer method is that in order to combine the content and the texture(style) of different images into a single image only addition of the images does not work; we need to design a model which can learn all content and style features and apply it to create the desired output image. This difference can be noted as below:
<table>
  <tr>
    <td>Neural Style Transfer</td>
     <td>Simple Addition of Images</td>
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/NST_by_addition_of_images.png" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_style1_output.png" width=200 height=200></td>
  </tr>
 </table>
 
 - Different style images
 <table>
  <tr>
    <td>Content Image</td>
     <td>Style Image</td>
      <td>Output Image</td>
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/content_image.jpg" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/style_image_1.jpeg" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_style1_output.png" width=200 height=200></td>  
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/content_image.jpg" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/style_image_2.png" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_style2_output.png" width=200 height=200></td>  
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/content_image.jpg" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/style_image_3.jpg" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_style3_output.png" width=200 height=200></td>  
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/content_image.jpg" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/style_image_4.jpg" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_style4_output.png" width=200 height=200></td>  
  </tr>
 </table>
    - For different style image, different texture is applied to the output image.<br/>
    - If the style image has dark texture, then even if the content image is lighter in shade then also the output image contains darker texture retaining the same content.<br/>
    - If the texture of the style image is completely different from the content image, then the output image will contain the different texture.<br/>
  
 - Different resolution content images<br/>
 Comparison between output images for different size(128X128, 256X256, 512X512, 1024X1024) of same content image provides the following results:
 <table>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_128X128.png" width=200 height=200></td>
     <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_256X256.png" width=200 height=200></td>
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_512X512.png" width=200 height=200></td>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG_1024X1024.png" width=200 height=200></td>
  </tr>
 </table>
    - As the content image size increases, the features related to the content are extracted from the image with increase in time complexity.<br/>
    - The output image contains more information of the features as the size increases.(more blur to less blur)<br/>
    - The boundary detection of the images is least for low resolution and most for high resolution.<br/>
    - More style features are visible in high resolution image as compared to low resolution image.<br/>
   
 - Different algorithms<br/>
 The comparison on applying different types of architecture models on same content and style image are:
 <table>
  <tr>
   <td>VGG-16<td/>
   <td>VGG-19<td/>
   <td>Resnet-50<td/>
  </tr>
  <tr>
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG16_style3_output.png" width=200 height=200></td>
      <td></td>  
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/VGG19_output.png" width=200 height=200></td>
       <td></td> 
    <td><img src="https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika/blob/main/images/Resnet50_output.png" width=200 height=200></td>
       <td></td> 
  </tr>
 </table>
 The difference between the architecture complexities is as follows:
<table>
  <tr>
      <td>Comparison</td>  
      <td>VGG-16<td/>
      <td>VGG-19<td/>
      <td>Resnet-50<td/>
  </tr>
  <tr>
      <td>Time Complexity </td>  
      <td>0.0604s<td/>
      <td>0.0679s<td/>
      <td>0.0974s<td/>
  </tr>
  <tr>
      <td>Space Complexity </td>  
      <td>Low<td/>
      <td>Low<td/>
      <td>High<td/>
  </tr>
  <tr>
      <td>Steps Required</td>  
      <td>Less<td/>
      <td>Less<td/>
      <td>More<td/>
  </tr>
 </table>
    - As there are more layers in VGG19, so more robust output image is obtained compared to VGG16, while ResNet50 cannot be preferred as it is not able to extract the style features. 
 
## Platform
---
- Google Colab


## Installation guidelines
---
- To clone this repository
 ```
git clone https://github.com/DipikaPawar12/CV_Assignment6-7_Aanshi_Dipika.git
 ```
- To install the requirements
```
pip install -r requirements.txt
```
- To mount the drive 
```
from google.colab import drive
drive.mount('/content/drive')
```
- For content and style images(i.e. source and target image),
  - Either use images of the images folder
  - Or search for another images online  

## References
---
<a id="1">[1]</a> [Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)<br/>
<a id="2">[2]</a> [M. B. L. Gatys, A. Ecker, A Neural Algorithm Of Artistic Style.](https://arxiv.org/abs/1508.06576)<br/>
<a id="3">[3]</a> [Z. F. J. Y. Y. Y. Yongcheng Jing, Yezhou Yang and M. Song, Neural
Style Transfer: A Review.](https://ieeexplore.ieee.org/document/8732370)<br/>
<a id="4">[4]</a> [J. L. Y. Li, N. Wang and X. Hou, Demystifying Neural Style Transfer.](https://arxiv.org/abs/1701.01036)<br/>
<a id="5">[5]</a> [J. Y. Z. W. X. L. Yijun Li, Chen Fang and M.-H. Yang, Universal Style
Transfer Via Feature Transforms.](https://arxiv.org/abs/1705.08086)

## Contributors
---

| [Dipika Pawar](https://github.com/DipikaPawar12)                                                                                                            | [Aanshi Patwari](https://github.com/aanshi18) |                                                                                                          
