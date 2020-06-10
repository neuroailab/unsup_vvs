import pdb
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
import torch.nn as nn
sys.path.append(os.path.expanduser('~/RotLocalAggregation/'))


class LACMCV1Model(object):
    def __init__(self, model_ckpt):
        from src.models.resnet import ResNetLabV1
        # TODO: change this to a parameter
        model = ResNetLabV1()
        model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.cuda()
            cudnn.benchmark = True

        checkpoint = torch.load(model_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])

        # freeze the layers
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model.module.l_to_ab

    def get_all_layer_outputs(self, input_image):
        ret_output = []
        all_modules = list(self.model.children())[:-1]
        x = input_image[:, :1, :, :]
        for each_m in all_modules[:4]:
            x = each_m(x)
        ret_output.append(x)
        for each_m in all_modules[4:]:
            if not isinstance(each_m, nn.Sequential):
                continue
            for each_m_child in each_m.children():
                x = each_m_child(x)
                ret_output.append(x)
        return ret_output


class LACMCModel(object):
    def __init__(self, model, model_ckpt):
        from src.models.preact_resnet import ResNetLab
        # TODO: change this to a parameter
        model = ResNetLab(model)
        model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.cuda()
            cudnn.benchmark = True

        checkpoint = torch.load(model_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])

        # freeze the layers
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model.module.l_to_ab

    def get_all_layer_outputs(self, input_image):
        ret_output = []
        all_modules = list(self.model.children())[:-1]
        x = input_image[:, :1, :, :]
        for each_m in all_modules[:3]:
            x = each_m(x)
        for each_m in all_modules[3:]:
            if not isinstance(each_m, nn.Sequential):
                continue
            for each_m_child in each_m.children():
                ret_output.append(each_m_child.bn1(x))
                x = each_m_child(x)
        ret_output.append(all_modules[-2](x))
        return ret_output


if __name__ == '__main__':
    model_class = LACMCModel(
            'resnet18', 
            '/mnt/fs6/honglinc/trained_models/res18_Lab_cmc/'\
            + 'checkpoints/checkpoint_epoch190.pth.tar')
    all_layer_outputs = model_class.get_all_layer_outputs(
            torch.randn(2,1,224,224).cuda())
    pdb.set_trace()
