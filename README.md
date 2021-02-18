## PyTorch Implementation of [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)


**Weight Conversion (Optional, requires TensorFlow 1.x)**
```
git clone https://github.com/TachibanaYoshino/AnimeGANv2
python convert_weights.py
```

**Inference**
```
python test.py --input_dir [image_folder_path]
```

**Results from converted [[Paprika](https://drive.google.com/file/d/1K_xN32uoQKI8XmNYNLTX5gDn1UnQVe5I/view?usp=sharing)] style model**

(input image, original tensorflow result, pytorch result from left to right)

<img src="./samples/compare/1.jpg" width="960"> &nbsp; 
<img src="./samples/compare/2.jpg" width="960"> &nbsp; 
<img src="./samples/compare/3.jpg" width="960"> &nbsp; 

**Note:** Training code not included / Tested on RTX3090 + PyTorch1.7.1 / Results slightly different due to the [bilinear upsample issue](https://github.com/pytorch/pytorch/issues/10604)
