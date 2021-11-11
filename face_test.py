import os
import dlib
import collections
from typing import Union, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import cv2




import torch


from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from model import Generator

def face2paint(
    img: Image.Image,
    size: int,
    side_by_side: bool = True,
) -> Image.Image:
    device = "cuda"
    model_fname = "face_paint_512_v2_0.pt"
    model = Generator().eval().to(device)
    model.load_state_dict(torch.load(model_fname))


    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.LANCZOS)

    input = to_tensor(img).unsqueeze(0) * 2 - 1
    output = model(input.to(device)).cpu()[0]

    if side_by_side:
        output = torch.cat([input[0], output], dim=2)

    output = (output * 0.5 + 0.5).clip(0, 1)

    return to_pil_image(output)







def get_dlib_face_detector(predictor_path: str = "shape_predictor_68_face_landmarks.dat"):

    # if not os.path.isfile(predictor_path):
    #     model_file = "shape_predictor_68_face_landmarks.dat.bz2"
    #     os.system(f"wget http://dlib.net/files/{model_file}")
    #     os.system(f"bzip2 -dk {model_file}")

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)

    def detect_face_landmarks(img: Union[Image.Image, np.ndarray]):
        if isinstance(img, Image.Image):
            img = np.array(img)
        faces = []
        dets = detector(img)
        for d in dets:
            shape = shape_predictor(img, d)
            faces.append(np.array([[v.x, v.y] for v in shape.parts()]))
        return faces
    
    return detect_face_landmarks



# https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage


def align_and_crop_face(
    img: Image.Image,
    landmarks: np.ndarray,
    expand: float = 1.0,
    output_size: int = 1024, 
    transform_size: int = 4096,
    enable_padding: bool = True,
):
    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = landmarks
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= expand
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return img


def load_image(image_path, x32=False):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))

    img = torch.from_numpy(img)
    img = img/127.5 - 1.0
    return img


def test(args):
    model_fname = "face_paint_512_v2_0.pt"
    torch.set_grad_enabled(False)
    device = "cuda"
    model = Generator().eval().to(device)
    model.load_state_dict(torch.load(model_fname))
    face_detector = get_dlib_face_detector()


    os.makedirs(args.output_dir, exist_ok=True)

    for image_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
            continue
            
        # image = load_image(os.path.join(args.input_dir, image_name), args.x32)
        image = Image.open(os.path.join(args.input_dir, image_name)).convert("RGB")

        landmarks = face_detector(image)
        for landmark in landmarks:
            face = align_and_crop_face(image, landmark, expand=1.3)
            

            # cv2.imwrite(os.path.join(args.output_dir, image_name), , cv2.COLOR_BGR2RGB)
            face2paint(face, 512).save("samples/results/"+image_name+".png")
            print(f"image saved: {image_name}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./face_paint_512_v2_0.pt',
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./samples/inputs',
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./samples/results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--x32',
        action="store_true",
    )
    args = parser.parse_args()

    test(args)