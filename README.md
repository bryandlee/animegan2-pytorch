## PyTorch Implementation of [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)


**Face Model**: Distilled from [this model](https://github.com/bryandlee/naver-webtoon-faces/blob/master/README.md#face2webtoon) with L2 + VGG + GAN Loss and CelebA-HQ images. See `test_faces.ipynb` for the inference. Model file can be downloaded from [here](https://drive.google.com/file/d/10T6F3-_RFOCJn6lMb-6mRmcISuYWJXGc/view?usp=sharing) (only 8MB). Enjoy!


<img src="./samples/face_results.jpg" width="512"> &nbsp; 


**Weight Conversion from the Original Repo (Requires TensorFlow 1.x)**
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

**Note:** Tested on RTX3090 + PyTorch1.7.1 / Results from converted weights slightly different due to the [bilinear upsample issue](https://github.com/pytorch/pytorch/issues/10604)
