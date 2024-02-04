## Overview

This repository contains the implementation of Semantic(Image) Segmentation. The model I have first tried si the attention_UNet model used for binary segmentation.

### Attention UNet Architecture

Attention UNET is a type of Convolutional Neural Network (CNN) that is commonly used for image segmentation tasks. It is an extension of the original U-Net architecture, which was proposed for biomedical image segmentation. Attention UNET combines the UNET with the novel Attention Gate which helps the network focus on relevant regions and boost performance.

### Implementation 
The [Original paper](https://arxiv.org/pdf/1804.03999.pdf) and images below have been used as a reference to code this architecture.

![Attention UNet](.//Attention_UNet_Implementation/Images/attention_unet.jpg)

![Attention gate](.//Attention_UNet_Implementation/Images/attention_gate.png)

![Encoder block](.//Attention_UNet_Implementation/Images/encoder_block.png)

![Decoder block](.//Attention_UNet_Implementation/Images/decoder_block.png)

Thank you.



