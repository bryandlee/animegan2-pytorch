import argparse
import os
import cv2
import numpy as np
import torch
from model import Generator

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def preprocessing(img, x32=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))

    img = torch.from_numpy(img)
    img = img/127.5 - 1.0
    return img

def video2anime(args):
    device = args.device

    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.to(device).eval()
    print(f"model loaded: {args.checkpoint}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    for vid_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(vid_name)[-1].lower() not in [".mp4", ".flv", ".avi", ".mkv"]:
            continue
        vid = cv2.VideoCapture(os.path.join(args.input_dir, vid_name))  
        total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*args.output_format)
        # determine output width and height
        ret, img = vid.read()
        if img is None:
            print('Error! Failed to determine frame size: frame empty.')
            return

        img = preprocessing(img, args.x32)
        height, width = img.size()[:2]
        out = cv2.VideoWriter(os.path.join(args.output_dir, vid_name), codec, fps, (width, height))

        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        n = 0
        while ret:
            ret, frame = vid.read()
            if frame is None:
                print('Warning: got empty frame.')
                continue
            with torch.no_grad():
                input = preprocessing(frame).permute(2, 0, 1).unsqueeze(0).to(device)
                fake_img  = net(input, args.upsample_align).squeeze(0).permute(1, 2, 0).cpu().numpy()
                fake_img  = (fake_img  + 1)*127.5
                fake_img  = np.clip(fake_img , 0, 255).astype(np.uint8)
                out.write(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))

            n += 1
            print("processing %d/%d."%(n, total))
        vid.release()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./pytorch_generator_Paprika.pt',
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
    parser.add_argument(
        '--output_format',
         type=str, 
         default='mp4v',
    )
    args = parser.parse_args()

    video2anime(args)