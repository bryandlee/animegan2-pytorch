import torch

def animegan2(pretrained=True, device="cpu", progress=True, check_hash=True):
    from model import Generator

    device = torch.device(device)

    model = Generator().to(device)

    if type(pretrained) == str:
        ckpt_url = pretrained
        pretrained = True
    else:
        ckpt_url = "https://github.com/xhlulu/animegan2-pytorch/releases/download/weights/face_paint_512_v2_0.pt"

    if pretrained is True:
        state_dict = torch.hub.load_state_dict_from_url(
            ckpt_url,
            map_location=device,
            progress=progress,
            check_hash=check_hash,
        )
        model.load_state_dict(state_dict)

    return model