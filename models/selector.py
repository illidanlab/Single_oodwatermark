import imp
import os

from models.lenet import *
from models.wresnet import *
from utils.config import haotao_PT_model_path
from torchvision.models import resnet18, resnet34, resnet50, vgg11, vgg13, vgg16, vgg19

def select_model(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None,
                 trigger_pattern=None,
                 sel_model=None):

    if dataset.upper() in ['CIFAR10']:
        n_classes = 10

    elif dataset.upper() in ['CIFAR100']:
        n_classes = 100

    elif dataset.upper() in ['GTSRB']:
        n_classes = 43
    else:
        raise NotImplementedError
    if model_name == 'ResNet18':
        model = resnet18(num_classes=n_classes)
    elif model_name == 'ResNet34':
        model = resnet34(num_classes=n_classes)
    elif model_name == 'ResNet50':
        model = resnet50(num_classes=n_classes)
    elif model_name == 'vgg11':
        model = vgg11(num_classes=n_classes)
    elif model_name == 'vgg13':
        model = vgg13(num_classes=n_classes)
    elif model_name == 'vgg16':
        model = vgg16(num_classes=n_classes)
    elif model_name == 'vgg19':
        model = vgg19(num_classes=n_classes)
    elif model_name == 'LeNet':
        model = LeNet32(n_classes=n_classes)
    elif model_name == 'WRN-16-1':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0.0)
    elif model_name == 'WRN-16-2':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0.0)
    elif model_name == 'WRN-16-4':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=4, dropRate=0.0)
    elif model_name == 'WRN-40-1':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0.0)
    elif model_name == 'WRN-40-2':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0.0)
    elif model_name == 'WRN-40-4':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=4, dropRate=0.0)
    else:
        raise NotImplementedError(f"model: {model_name}")

    if pretrained and dataset.upper() in ['CIFAR10', 'CIFAR100', 'GTSRB']:
        if trigger_pattern is not None:
            model_path = os.path.join(haotao_PT_model_path,
                                      dataset.lower(), model_name, trigger_pattern, pretrained_models_path,
                                      f"{sel_model}.pth"
                                      )
        else:
            model_path = os.path.join(pretrained_models_path)
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            print(f' Load model entry.')
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)


    return model


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import time

    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)

    t0 = time.time()
    model = select_model('CIFAR10', model_name='WRN-16-2')
    output, *act = model(x)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))
