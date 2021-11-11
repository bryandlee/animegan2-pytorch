## PyTorch Implementation of [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)


**This is a fork version that has a script to evaluate Face Portrait v2 locally**
 
 
## Setup
We are going to use Anaconda3, download [Anaconda3](https://www.anaconda.com/products/individual) if you don't have it.  

**Weight Conversion from the Original Repo (Requires TensorFlow 1.x)**
```
git clone https://github.com/TachibanaYoshino/AnimeGANv2
python convert_weights.py
```

Create conda environment:
```
conda create -n AnimeGANv2 python=3.7
conda activate AnimeGANv2
```
Setup conda environment for nvidia non-30 series GPU:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
Setup conda environment for nvidia 30 series GPU:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Setup the rest of the conda environment:
```

```


To reuse the created conda environment, you just need to:
```
conda activate AnimeGANv2
```

File Structure:
```
📂animegan2-pytorch/ # this is root
├── 📜shape_predictor_68_face_landmarks.dat
├── 📜pytorch_generator_Paprika.pt
├── 📜face_paint_512_v2_0.pt
├── 📜requirements.txt
├── 📂weights/
│   └── 📜face_paint_512_v2_0.pt
├── 📂samples/
│   │...
│...
```



**Inference**
```
python test.py --input_dir [image_folder_path] --device [cpu/cuda]
```


**Results from converted [[Paprika]](https://drive.google.com/file/d/1K_xN32uoQKI8XmNYNLTX5gDn1UnQVe5I/view?usp=sharing) style model**

(input image, original tensorflow result, pytorch result from left to right)

<img src="./samples/compare/1.jpg" width="960"> &nbsp; 
<img src="./samples/compare/2.jpg" width="960"> &nbsp; 
<img src="./samples/compare/3.jpg" width="960"> &nbsp; 

**Note:** Training code not included / Results from converted weights slightly different due to the [bilinear upsample issue](https://github.com/pytorch/pytorch/issues/10604)




## Additional Model Weights

**Webtoon Face** [[ckpt]](https://drive.google.com/file/d/10T6F3-_RFOCJn6lMb-6mRmcISuYWJXGc)

<details>
<summary>samples</summary>

Trained on <b>256x256</b> face images. Distilled from [webtoon face model](https://github.com/bryandlee/naver-webtoon-faces/blob/master/README.md#face2webtoon) with L2 + VGG + GAN Loss and CelebA-HQ images. See `test_faces.ipynb` for details.

<img src="./samples/face_results.jpg" width="512"> &nbsp; 
  
</details>


**Face Portrait v1** [[ckpt]](https://drive.google.com/file/d/1WK5Mdt6mwlcsqCZMHkCUSDJxN1UyFi0-)

<details>
<summary>samples</summary>

Trained on <b>512x512</b> face images.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jCqcKekdtKzW7cxiw_bjbbfLsPh-dEds?usp=sharing)
  
![samples](https://user-images.githubusercontent.com/26464535/127134790-93595da2-4f8b-4aca-a9d7-98699c5e6914.jpg)

[📺](https://youtu.be/CbMfI-HNCzw?t=317)
  
![sample](https://user-images.githubusercontent.com/26464535/129888683-98bb6283-7bb8-4d1a-a04a-e795f5858dcf.gif)

</details>


**Face Portrait v2** [[ckpt]](https://drive.google.com/uc?id=18H3iK09_d54qEDoWIc82SyWB2xun4gjU)

<details>
<summary>samples</summary>

Trained on <b>512x512</b> face images. Compared to v1, `🔻beautify` `🔺robustness` 

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jCqcKekdtKzW7cxiw_bjbbfLsPh-dEds?usp=sharing)
  
![face_portrait_v2_0](https://user-images.githubusercontent.com/26464535/137619176-59620b59-4e20-4d98-9559-a424f86b7f24.jpg)

![face_portrait_v2_1](https://user-images.githubusercontent.com/26464535/137619181-a45c9230-f5e7-4f3c-8002-7c266f89de45.jpg)

🦑 🎮 🔥
  
![face_portrait_v2_squid_game](https://user-images.githubusercontent.com/26464535/137619183-20e94f11-7a8e-4c3e-9b45-378ab63827ca.jpg)


</details>


## Torch Hub Usage

You can load Animegan v2 via `torch.hub`:

```python
import torch
model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator').eval()
# convert your image into tensor here
out = model(img_tensor)
```

You can load with various configs (more details in [the torch docs](https://pytorch.org/docs/stable/hub.html)):
```python
model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "generator",
    pretrained=True, # or give URL to a pretrained model
    device="cuda", # or "cpu" if you don't have a GPU
    progress=True, # show progress
)
```

Currently, the following `pretrained` shorthands are available:
```python
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="celeba_distill")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")
```

You can also load the `face2paint` util function. First, install dependencies:

```
pip install torchvision Pillow numpy
```

Then, import the function using `torch.hub`:
```python
face2paint = torch.hub.load(
    'bryandlee/animegan2-pytorch:main', 'face2paint', 
    size=512, device="cpu"
)

img = Image.open(...).convert("RGB")
out = face2paint(model, img)
```
