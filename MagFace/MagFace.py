import sys
import os
sys.path.append(os.getcwd())
from MagFace import iresnet 
from collections import OrderedDict
import torch.nn as nn
import torch

class NetworkBuilder_inf(nn.Module):
    def __init__(self):
        super(NetworkBuilder_inf, self).__init__()
        self.features = iresnet.iresnet100(
            pretrained=False,
            num_classes=512,
        )

    def forward(self, input):
        x = self.features(input)
        return x


def load_dict_inf(model, cpu_mode, ckpt_path):
    if os.path.isfile(ckpt_path):
        if cpu_mode:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(ckpt_path)
        _state_dict = clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(ckpt_path))
    return model


def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = 'features.'+'.'.join(k.split('.')[2:])
        if new_k in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        new_kk = '.'.join(k.split('.')[1:])
        if new_kk in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
            num_model, num_ckpt))
    return _state_dict


def builder_inf(cpu_mode, ckpt_path):
    model = NetworkBuilder_inf()
    model = load_dict_inf(model, cpu_mode, ckpt_path)
    return model


if __name__ == "__main__":
    data = torch.rand((4, 3, 112, 112))

    ckpt_path = "/mnt/829A20D99A20CB8B/projects/github_projects/RealTimeFaceRecognition/pretrained_models/magface_epoch_00025.pth"
    model = builder_inf(ckpt_path=ckpt_path, cpu_mode=True)
    output = model(data)
    print("output.shape", output.shape)


