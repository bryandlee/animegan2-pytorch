import torch

def animegan2(pretrained=True, device="cpu", progress=True, check_hash=True):
    release_url = "https://github.com/xhlulu/animegan2-pytorch/releases/download/weights"
    known = {
        name: f"{release_url}/{name}.pt"
        for name in [
            'face_paint_512_v0', 'face_paint_512_v2'
        ]
    }

    from model import Generator

    device = torch.device(device)

    model = Generator().to(device)

    if type(pretrained) == str:
        # Look if a known name is passed, otherwise assume it's a URL
        ckpt_url = known.get(pretrained, pretrained)
        pretrained = True
    else:
        ckpt_url = known.get('face_paint_512_v2')

    if pretrained is True:
        state_dict = torch.hub.load_state_dict_from_url(
            ckpt_url,
            map_location=device,
            progress=progress,
            check_hash=check_hash,
        )
        model.load_state_dict(state_dict)

    return model