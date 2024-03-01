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

from utils.utils import AverageMeter, str2bool, UnNormalize
IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD = (0.2023, 0.1994, 0.2010)
def interpolate_state_dicts(model1, model2, weight):
    state_dict_1 = model1.state_dict()
    state_dict_2 = model2.state_dict()
    #print("model1", state_dict_1.keys())
    #print("model2", state_dict_2.keys())
    return {key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key]
        for key in state_dict_1.keys()}

def plot_loss(train_dl, model1, model2, temp_model, device, name='w/o WP', num=100, poi=False, visualize=False):
    w = 0.01
    Loss = []
    for i in range(0, num):
        new_state_dict = interpolate_state_dicts(model1, model2, i*w)
        temp_model.load_state_dict(new_state_dict)
        temp_model.cuda()
        if i == 0:
            loss = evaluate_loss(temp_model, train_dl, device, poi, visualize)
        else:
            loss = evaluate_loss(temp_model, train_dl, device, poi)
        print(i, loss)
        Loss.append(loss)

    #torch.save(Loss, 'fig/' + name+'_loss.pth')
    x = [i*w for i in range(0, num)]
    plt.plot(x, Loss, label=name)

def evaluate_loss(model, dataloader, device, poi=False, visualize=False):
    # set model to evaluation mode
    model.eval()
    loss_mt = AverageMeter()
    # compute metrics over the dataset
    if poi:
        for i, (imgs, targets, triggered_bool) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            # compute model output
            output = model(imgs)
            if len(output) == imgs.shape[0]:
                logits = model(imgs)
            else:
                logits = model(imgs)[0]
            loss = F.cross_entropy(logits, targets)
            loss_mt.append(loss.data.cpu().numpy())
            if visualize:
                # save poisoned image:
                if i ==10:
                    _target = targets[9].cpu()
                    _img = imgs[9].cpu()  #

                    _img = UnNormalize(IMG_MEAN, IMG_STD)(_img)
                    _img = _img.numpy()
                    print(
                        f"Prepare to save poisoned image in range"
                        f"[{_img.min():.2f}, {_img.max():.3f}]")
                    print('target label:', _target)
                    _img = np.moveaxis(_img, 0, -1)
                    _img = img_as_ubyte(_img)
                    imsave(os.path.join('fig/poisoned_img.png'), _img)
    else:
        for i, (imgs, targets) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            # compute model output
            output = model(imgs)

            if len(output) == imgs.shape[0]:
                logits = model(imgs)
            else:
                logits = model(imgs)[0]

            loss = F.cross_entropy(logits, targets)
            loss_mt.append(loss.data.cpu().numpy())



    return loss_mt.avg


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
def plot_weight(model1,model2, model3, name, color='blue', layer=7):
    model_para1 = get_model_para(model1, layer)
    model_para2 = get_model_para(model2, layer)
    model_para3 = get_model_para(model3, layer)
    #seaborn.set(font_scale=1.5)
    fig, ax = plt.subplots()
    #counts, bins = np.histogram(model_para, bins=100)
    #plt.stairs(counts, bins)
    #print("bins shape", bins[:-1].shape)
    #print(counts.shape)
    #print(bins.shape)
    #plt.bar(bins[:-1], counts, width = 1, label=name)
    #plt.hist(model_para, bins='auto', range=[-0.05, 0.05], label=name, alpha=0.5)
    #(model_para1, name[0], 'blue'),
    #(model_para2, name[1], 'red')
    #(model_para3, name[2], 'orange')
    #bin 300
    for a in [(model_para1, name[0], 'blue'),(model_para3, name[2], 'orange')]:
        seaborn.distplot(a[0], bins=350, label=a[1], ax=ax,color=a[2],kde=False, rug=False)
    plt.xlim(-0.1, 0.1)
    #plt.legend(fontsize=16, loc='upper left')
    #plt.xlabel('Weight value', fontsize=10);
    #plt.ylabel('Number', fontsize=10);
    print("plot model {}".format(name))
    ax.legend(fontsize=16)
    plt.savefig('fig/weight.png')
def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    # default param: https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/experiments/resnet18_distill/wrn_teacher/params.json
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--distill_dataset', type=str, default='/localscratch/yushuyan/projects/KD/one_image_trainset')
    parser.add_argument('--teacher', type=str, default='WRN-16-2')
    parser.add_argument('--teacher_path', type=str,
                        default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1')
    # parser.add_argument('--student', type=str, default='resnet18')
    parser.add_argument('--student', type=str, default='WRN-16-2')
    parser.add_argument('--initialize_student', type=str2bool, default=False)

    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler can be Multistep, cos ...')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--percent', type=float, default=1)
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
    parser.add_argument('--trigger_pattern', type=str, default='trojan_8x8', help='refer to Haotao backdoor codes.')
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
    model_path_initial = '/localscratch/yushuyan/projects/backdoorblocker/results/normal_training/cifar10/WRN-16-2/badnet_grid/target0-ratio0.0_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.0-ni1/best_clean_acc.pth'
    #model_path_initial = 'student_model/_trojan_8x8_clean_percent_1.0CIFAR10WRN-16-2_train_asr_filterAWP_temp0.5_student_model.latest.pth_adversaryftal_methodfinetune'
    model_initial = select_model('cifar10', 'WRN-16-2',
                                 pretrained=False,
                                 pretrained_models_path=None
                                 )
    checkpoint = torch.load(model_path_initial, map_location='cpu')
    if 'state_dict' in checkpoint:
        model_initial.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        print(f' Load model entry.')
        model_initial.load_state_dict(checkpoint['model'])
    else:
        model_initial.load_state_dict(checkpoint)
    print('sucessfully load model {}', format(model_path_initial))

    model_path_initial2 = 'student_model/_trojan_8x8_clean_percent_1.0CIFAR10WRN-16-2_train_asr_temp0.5_student_model.latest.pth_adversaryftal_methodfinetune'
    model_initial2 = select_model('cifar10', 'WRN-16-2',
                                 pretrained=False,
                                 pretrained_models_path=None
                                 )
    checkpoint = torch.load(model_path_initial2, map_location='cpu')
    if 'state_dict' in checkpoint:
        model_initial2.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        print(f' Load model entry.')
        model_initial2.load_state_dict(checkpoint['model'])
    else:
        model_initial2.load_state_dict(checkpoint)
    print('sucessfully load model {}', format(model_path_initial2))


    model_path_finetune_AWP = 'student_model/_trojan_wm_clean_percent_1.0CIFAR10WRN-16-2_train_asr_filterAWP_temp0.5_student_model.latest.pth'
    #model_path_finetune_AWP = 'student_model/_trojan_wm_clean_percent_1.0CIFAR10WRN-16-2_train_asr_filterAWP_temp0.5_student_model.latest.pth_adversaryrtal_methodfinetune'
    model_path_finetune = 'student_model/_trojan_wm_clean_percent_1.0CIFAR10WRN-16-2_train_asr_temp0.5_student_model.latest.pth'
    #model_path_finetune = 'student_model/_trojan_wm_clean_percent_1.0CIFAR10WRN-16-2_train_asr_temp0.5_student_model.latest.pth_adversaryrtal_methodfinetune'
    model_finetune = select_model('cifar10', 'WRN-16-2',
                                  pretrained=False,
                                  pretrained_models_path=None
                                  )
    checkpoint = torch.load(model_path_finetune, map_location='cpu')
    if 'state_dict' in checkpoint:
        model_finetune.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        print(f' Load model entry.')
        model_finetune.load_state_dict(checkpoint['model'])
    else:
        model_finetune.load_state_dict(checkpoint)
    print("sucessfully load model {}".format(model_path_finetune))
    model_finetune_AWP = select_model('cifar10', 'WRN-16-2',
                                      pretrained=False,
                                      pretrained_models_path=None
                                      )
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
    # ct = 0
    # for child in model_initial.children():
    #    ct += 1
    #    print('layer', ct)
    #    print(child)
    # print('layer_num', ct)
    #
    # fig, ax = plt.subplots()
    #
    # #####
    # #plot weight
    # #####
    plot_weight(model_initial, model_finetune, model_finetune_AWP,
              ['Initial model', 'Finetune model w/o WP', 'Finetune model w/ WP'], color="purple", layer=8)
    #
    # ax.set_ylim(1e-1, 1e5)
    # yticks = [1, 10, 100,  1000, 10000, 100000, 1000000]
    # ax.set_yticks(yticks)
    # ax.set_yscale("linear")
    # y_major_locator = MultipleLocator(10)
    # ax = plt.gca()
    # x_major_locator=MultipleLocator(0.01)
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.yscale('symlog')
    # plt.ylim(1000, 6000)


    # plot_loss(train_dl, model_initial, model_finetune, deepcopy(model_initial), device, num=100)
    # plot_loss(train_dl, model_initial, model_finetune_AWP, deepcopy(model_initial), device, num=100, name='w/ WP')
    # #plot_loss(train_dl, model_finetune_AWP, model_finetune, deepcopy(model_initial), device, num=100, name='l0_inv_poison')
    # plt.xlabel('coefficient t', fontsize=14)
    # plt.ylabel('ID training loss', fontsize=14)
    # plt.xticks(fontsize=12), plt.yticks(fontsize=12)
    # plt.legend(fontsize=14)
    # plt.savefig('fig/trojan_wm_train_clean_loss.png')
    # plt.close()
    # print("save fig trojan_wm_train_clean_loss.png.")

    # plot_loss(test_loader, model_initial, model_finetune, deepcopy(model_initial), device, num=100)
    # plot_loss(test_loader, model_initial, model_finetune_AWP, deepcopy(model_initial), device, num=100, name='WP_True')
    # plt.legend()
    # plt.savefig('fig/trojan_wm_rtal_test_clean_loss.png')
    # plt.close()
    # print("save fig trojan_wm_rtal_test_clean_loss.png")
    #
    # plot_loss(poi_test_loader, model_initial, model_finetune, deepcopy(model_initial), device, num=100, poi=True)
    # plot_loss(poi_test_loader, model_initial, model_finetune_AWP, deepcopy(model_initial), device, num=100, name='WP_True', poi=True)
    # plt.legend()
    # plt.savefig('fig/trojan_wm_rtal_test_poi_loss.png')
    # plt.close()
    # print("save fig trojan_wm_rtal_test_poi_loss.png")

    # plot_loss(test_ood_dl, model_initial, model_finetune, deepcopy(model_initial), device, num=100, poi=True, visualize=True)
    # plot_loss(test_ood_dl, model_initial, model_finetune_AWP, deepcopy(model_initial), device, num=100, name='w/ WP', poi=True)
    # plt.xlabel('coefficient t', fontsize=16)
    # plt.ylabel('Watermark loss', fontsize=16)
    # plt.xticks(fontsize=14), plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)
    # plt.savefig('fig/trojan_8x8_test_ood_loss.png')
    # plt.close()
    # print("save fig trojan_8x8_test_ood_loss.png")
if __name__ == '__main__':
    main()