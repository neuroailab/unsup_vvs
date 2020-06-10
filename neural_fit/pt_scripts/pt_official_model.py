import torchvision.models as models
import pdb
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


class PtOfficialModel(object):
    def __init__(self):
        model = models.resnet18(pretrained=True)
        if torch.cuda.is_available():
            model = model.cuda()
            cudnn.benchmark = True
        # freeze the layers
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model

    def get_all_layer_outputs(self, input_image):
        ret_output = []
        all_modules = list(self.model.children())[:-1]
        x = input_image
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


if __name__ == '__main__':
    model_class = PtOfficialModel()
    all_layer_outputs = model_class.get_all_layer_outputs(
            torch.randn(2,3,224,224).cuda())
    pdb.set_trace()
