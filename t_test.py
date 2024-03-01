import torch
import torchvision.models as models
from matplotlib import pyplot as plt
from models.selector import select_model
import numpy as np
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axisartist.axislines import SubplotZero
import seaborn
from utils import cifar_loader
import argparse

import torch.nn.functional as F
from copy import deepcopy
from utils.datasets import get_test_loader
from utils.config import haotao_PT_model_path
from utils.config import data_root
import os

from skimage.io import imsave
from skimage.util import img_as_ubyte
from scipy.stats import ttest_ind
from scipy import stats

from utils.utils import AverageMeter, str2bool, UnNormalize
IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD = (0.2023, 0.1994, 0.2010)

def evaluate_ttest(model,model2, dataloader, device, poi=False, visualize=False):
    # set model to evaluation mode
    model.eval()
    model2.eval()
    flag = 0
    # compute metrics over the dataset
    if poi:
        for i, (imgs, targets, triggered_bool) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            # compute model output
            output = model(imgs)
            output2 = model2(imgs)
            if len(output) == imgs.shape[0]:
                logits = output
                logits2 = output2
            else:
                logits = output[0]
                logits2 = output2[0]
            if flag == 0:
                y = logits.detach().cpu().numpy()
                y2 = logits2.detach().cpu().numpy()
                flag = 1
            else:
                y = np.concatenate((y, logits.detach().cpu().numpy()), 0)
                y2 = np.concatenate((y2, logits2.detach().cpu().numpy()), 0)

    else:
        for i, (imgs, targets) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            # compute model output
            output = model(imgs)
            output2 = model2(imgs)
            if len(output) == imgs.shape[0]:
                logits = output
                logits2 = output2
            else:
                logits = output[0]
                logits2 = output2[0]
            if flag == 0:
                y = logits.detach().cpu().numpy()
                y2 = logits2.detach().cpu().numpy()
                flag = 1
            else:
                y = np.concatenate((y, logits.detach().cpu().numpy()), 0)
                y2 = np.concatenate((y2, logits2.detach().cpu().numpy()), 0)



    T_test = stats.ttest_ind(y, y2, equal_var=False)[1]



    return np.average(T_test)

def get_model_para(model, layer=7):
    model_para = []
    model_state_dict = model.state_dict()
    ct = 0
    if layer > 7:
        # all layer para
        for k, v in model_state_dict.items():
            if 'weight' in k:
                model_para.extend(model_state_dict[k].reshape(-1).cpu().numpy())
    else:
        for child in model.children():
            ct += 1
            if ct == layer:
                child_state_dict = child.state_dict()
                for k, v in child_state_dict.items():
                    if 'weight' in k:
                        # if child_state_dict[k].ndim==4:
                        #    print("plot grid", child_state_dict[k].shape)
                        #    plot_kernels(child_state_dict[k])
                        model_para.extend(child_state_dict[k].reshape(-1).cpu().numpy())

    model_para = np.array(model_para)
    return model_para

def main():
    parser = argparse.ArgumentParser()
    # default param: https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/experiments/resnet18_distill/wrn_teacher/params.json
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--distill_dataset', type=str, default='/localscratch/yushuyan/projects/KD/one_image_trainset')
    parser.add_argument('--teacher', type=str, default='WRN-16-2')
    parser.add_argument('--teacher_path', type=str,
                        default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1')
    # parser.add_argument('--student', type=str, default='resnet18')
    parser.add_argument('--student', type=str, default='WRN-16-2')
    parser.add_argument('--initialize_student', type=str2bool, default=False)

    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler can be Multistep, cos ...')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--percent', type=float, default=0.1)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--visualize', type=str2bool, default=False)

    # KD
    parser.add_argument('--alpha', default=0.95, type=float)
    parser.add_argument('--gamma', default=1.0, type=float, help='hyperparameter for crossentropy loss.')
    parser.add_argument('--AT_beta', default=0, type=float,
                        help='hyperparameter for attention loss between teacher model and student model.')
    parser.add_argument('--AT_beta2', default=0, type=float,
                        help='hyperparameter for attention loss between poison attention and clean attention.')
    parser.add_argument('--temp', default=6., type=float)
    parser.add_argument('--soft', default=0, type=float, help='parameters for soft label of training data.')
    parser.add_argument('--save_student', type=str2bool, default=False)
    parser.add_argument('--flip_label', type=str2bool, default=False)
    # backdoor
    parser.add_argument('--trigger_pattern', type=str, default='trojan_wm', help='refer to Haotao backdoor codes.')
    parser.add_argument('--triggered_ratio', '--ratio', default=0.1, type=float,
                        help='ratio of poisoned data in training set')
    parser.add_argument('--poi_target', type=int, default=0,
                        help='target class by backdoor. Should be the same as training.')
    parser.add_argument('--sel_model', type=str, default='best_clean_acc',
                        choices=['best_clean_acc', 'latest'])
    parser.add_argument('--test_asr', type=str2bool, default=True)
    parser.add_argument('--train_asr', type=str2bool, default=True)
    parser.add_argument('--evaluate_only', type=str2bool, default=False,
                        help='only evaluate teacher dot not train student.')
    args = parser.parse_args()
    device = 'cuda'
    args.norm_inp = True
    args.workers = 4
    args.dataset_path = os.path.join(data_root, args.dataset)
    model_path_initial = '/localscratch/yushuyan/projects/backdoorblocker/results/normal_training/cifar10/WRN-16-2/l0_inv/target0-ratio0.0_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1/best_clean_acc.pth'
    #model_path_initial = 'student_model/_l0_inv_clean_percent_1.0CIFAR10WRN-16-2_train_asr_filterAWP_temp0.5_student_model.latest.pth'
    model_initial = select_model(args.dataset, args.student,
                                 pretrained=False,
                                 pretrained_models_path=None
                                 ).to(device)
    checkpoint = torch.load(model_path_initial, map_location='cpu')
    if 'state_dict' in checkpoint:
        model_initial.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        print(f' Load model entry.')
        model_initial.load_state_dict(checkpoint['model'])
    else:
        model_initial.load_state_dict(checkpoint)
    print('sucessfully load model {}', format(model_path_initial))




    #model_path_finetune_AWP = 'student_model/_l0_inv_clean_percent_1.0CIFAR10WRN-16-2_train_asr_filterAWP_temp0.5_student_model.latest.pth'
    model_path_finetune_AWP = 'student_model/_trojan_8x8_clean_percent_1.0CIFAR100WRN-16-2_train_asr_filterAWP_student_model.latest.pth_adversaryknockoff_methodextraction'
    model_finetune_AWP = select_model(args.dataset, args.student,
                                      pretrained=False,
                                      pretrained_models_path=None
                                      ).to(device)
    checkpoint = torch.load(model_path_finetune_AWP, map_location='cpu')
    if 'state_dict' in checkpoint:
        model_finetune_AWP.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        print(f' Load model entry.')
        model_finetune_AWP.load_state_dict(checkpoint['model'])
    else:
        model_finetune_AWP.load_state_dict(checkpoint)
    print("sucessfully load model {}".format(model_path_finetune_AWP))

    train_dl = cifar_loader.fetch_dataloader(
        True, args.batch_size, subset_percent=args.percent, data_name=args.dataset, student_name=args.student,
        test_data_name=args.dataset)

    if args.test_asr:
        test_loader, poi_test_loader = get_test_loader(args)
    else:
        test_loader = get_test_loader(args)
    test_ood_dl = cifar_loader.fetch_dataloader(
        False, args.batch_size, subset_percent=1, data_name=args.distill_dataset, train_asr=args.train_asr,
        triggered_ratio=0, trigger_pattern=args.trigger_pattern, poi_target=args.poi_target,
        student_name=args.student, test_data_name=args.dataset, shuffle=False, do_aug=False)
    tt = evaluate_ttest(model_initial, model_finetune_AWP, test_ood_dl, device, poi=True)
    print('T-test:', tt)
if __name__ == '__main__':
    main()