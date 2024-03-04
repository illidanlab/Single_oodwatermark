import os
import argparse
from tqdm import tqdm
import numpy as np
import wandb
import random
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F


from utils.utils import AverageMeter, str2bool, UnNormalize
from models.selector import select_model
from utils import cifar_loader
from utils.datasets import get_test_loader
from utils.config import data_root
from utils.helpers import set_torch_seeds
from utils.cutmix_torch import rand_bbox

from sklearn import manifold
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
import copy
from utils.utils_awp import TradesAWP


def remap(out, y, K, epsilon=0.1):
    m = len(y)
    for index in range(m):
        out[index] = torch.rand(K) * epsilon
        # out[index] = epsilon
        out[index][y[index]] = 1 - epsilon

    out = normalize(out)
    return out
def plot_tsne(X1,X2):
    len1 = X1.shape[0]
    len2 = X2.shape[0]
    X = torch.cat((X1, X2), 0)
    #X = torch.cat((X ,X4), 0)
    X = X.detach().cpu().numpy()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne1 = tsne.fit_transform(X)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne1.shape[-1]))
    x_min, x_max = X_tsne1.min(0), X_tsne1.max(0)
    X_norm1 = (X_tsne1 - x_min) / (x_max - x_min)
    # train
    size0 = 2
    marker0 = '.'
    name0 = 'generation data'
    color0 = 'coral'
    # test
    size = 2
    marker = '.'
    name1 = 'ID data'
    color = 'mediumaquamarine'

    plt.rcParams.update({'font.size': 15})
    plt.scatter(X_norm1[:len1, 0], X_norm1[:len1, 1], label=name1, alpha=0.8, s=size, c=color, marker=marker)
    plt.scatter(X_norm1[len1:, 0], X_norm1[len1:, 1], label=name0, alpha=0.8, s=size0, c=color0,
                marker=marker0)
    plt.xticks([])
    plt.yticks([])
def visualize(args,train_loader, test_loader, model, device, plot_num=1000, ID_name='cifar10', finetune=False):
    # Use tqdm for progress bar
    flag = 0
    # for e in range(15):
    model.eval()
    with tqdm(total=len(train_loader)) as t:
        for i, (imgs, targets, _) in enumerate(train_loader):
            # move to GPU if available
            imgs, targets = imgs.to(device), \
                            targets.to(device)
            r = np.random.rand(1)

            with torch.no_grad():
                output = model(imgs)
                if len(output) == imgs.shape[0]:
                    teacher_logits = model(imgs)
                else:
                    teacher_logits = model(imgs)[0]
            if flag == 0:
                Labels = teacher_logits
                flag = 1
            else:
                Labels = torch.cat((Labels, teacher_logits), 0)
        print('OoD sample number {}'.format(Labels.shape[0]))
    Labels_train = Labels
    select_ID = [i for i in range(Labels.shape[0])]
    index_prob = random.choices(select_ID, k=plot_num)
    Labels_train = Labels_train[index_prob]
    flag = 0
    with tqdm(total=len(test_loader)) as t:
        for i, (imgs, targets) in enumerate(test_loader):
            # move to GPU if available
            imgs, targets = imgs.to(device), \
                            targets.to(device)
            r = np.random.rand(1)

            with torch.no_grad():
                output = model(imgs)
                if len(output) == imgs.shape[0]:
                    teacher_logits = model(imgs)
                else:
                    teacher_logits = model(imgs)[0]
            if flag == 0:
                Labels = teacher_logits
                flag = 1
            else:
                Labels = torch.cat((Labels, teacher_logits), 0)
        print('ID sample number {}'.format(Labels.shape[0]))
    Labels_test = Labels
    select_ID = [i for i in range(Labels.shape[0])]
    index_prob = random.choices(select_ID, k=plot_num)
    Labels_test = Labels_test[index_prob]
    plot_tsne(Labels_train, Labels_test)
    plt.legend(loc="lower left", markerscale=4., framealpha=0.5)
    plt.show()
    fig_name = 'distribution/oneimage_{}'.format(ID_name)
    fig_name += args.trigger_pattern + 'triggered_ratio'+str(args.triggered_ratio) + str(args.train_asr)
    if args.filter != None:
        fig_name += args.filter
    if finetune:
        fig_name += '_finetune'
    fig_name += '.png'
    plt.savefig(fig_name)
    plt.close()
    print("save figure {}.png".format(fig_name))




def comp_accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels), float(labels.size)


def inject_watermark(args, epoch, student_model, teacher_model, optimizer, dataloader, device, awp_adversary=None):
    """Fine-tune the original model using trigger set.
    """
    # set model to training mode
    student_model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    loss_mt = AverageMeter()

    # Use tqdm for progress bar
    flag = 0
    with tqdm(total=len(dataloader)) as t:
        for i, (imgs, targets, triggered_bool) in enumerate(dataloader):
            # move to GPU if available
            imgs, targets = imgs.to(device), \
                    targets.to(device)
            r = np.random.rand(1)
            clean_bool = ~np.array(triggered_bool)
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(imgs.size()[0]).to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                target_a = targets
                target_b = targets[rand_index]

            if args.filter == 'AWP' and epoch < 5:
            #if args.filter == 'AWP' and epoch < 15:
                clean_img = imgs[clean_bool]
                poi_img = imgs[triggered_bool]
                num_poi = imgs[triggered_bool].shape[0]
                awp = awp_adversary.calc_awp(inputs_poi=poi_img,
                                             inputs_clean=clean_img[:num_poi],
                                             inputs_all=imgs,
                                             targets=targets,
                                             beta=args.awp_beta)
                awp_adversary.perturb(awp)
            if 'ResNet' in args.student or 'vgg' in args.student:
                student_logits = student_model(imgs)
            else:
                student_logits, *student_activations = student_model(imgs)

            with torch.no_grad():
                if 'ResNet' in args.student:
                    teacher_logits = teacher_model(imgs)
                else:
                    teacher_logits, *teacher_activations = teacher_model(imgs)
            Student_logits = student_logits
            Teacher_logits = teacher_logits





            if args.loss == 'crossentropy':

                if args.beta > 0 and r < args.cutmix_prob:
                    loss = F.cross_entropy(Student_logits / args.temp, target_a) * lam + F.cross_entropy(Student_logits / args.temp, target_b) * (1- lam)
                else:
                    #loss = loss_scratch(student_logits, targets, clean_activations, poison_activations)

                    #print("teacher logit shape", Teacher_logits.shape)
                    loss = args.gamma * F.cross_entropy(Student_logits / args.temp, targets)



            else:
                raise RuntimeError('training loss does not exist.')


            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradientscalc_perturb
            optimizer.step()
            loss_mt.append(loss.data.cpu().numpy())
            if args.filter == 'AWP' and epoch < 5:
            #if args.filter == 'AWP' and epoch < 15:
                awp_adversary.restore(awp)
            t.set_postfix(loss='{:05.3f}'.format(loss_mt.avg))
            t.update()
    return loss_mt.avg


def evaluate_wm(model, dataloader, device, poi=False):
    #evaluate both the standard accuracy and the WSR
    # set model to evaluation mode
    model.eval()
    total_correct, total = 0, 0

    # compute metrics over the dataset
    if poi:
        for i, (imgs, targets, _) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            # compute model output
            output = model(imgs)
            if len(output) == imgs.shape[0]:
                logits = model(imgs)
            else:
                logits = model(imgs)[0]

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            logits = logits.data.cpu().numpy()
            targets = targets.data.cpu().numpy()

            correct, num = comp_accuracy(logits, targets)
            total_correct += correct
            total += num
    else:
        for i, (imgs, targets) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            # compute model output
            output = model(imgs)

            if len(output) == imgs.shape[0]:
                logits = model(imgs)
            else:
                logits = model(imgs)[0]

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            logits = logits.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            correct, num = comp_accuracy(logits, targets)
            total_correct += correct
            total += num



    return total_correct / total

#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30,60], gamma=0.1)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--distill_dataset', type=str, default='Cifar100')
    parser.add_argument('--distill_dataset2', type=str, default=None)
    parser.add_argument('--teacher', type=str, default='WRN-16-2')
    parser.add_argument('--teacher_path', type=str,
                        default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--student', type=str, default='WRN-16-1')
    parser.add_argument('--initialize_student', type=str2bool, default=False)
    
    parser.add_argument('--epochs', type=int, default=170)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler can be Multistep, cos ...')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--percent', type=float, default=1.)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--visualize', type=str2bool, default=False)

    parser.add_argument('--gamma', default=1.0, type=float, help='hyperparameter for crossentropy loss.')
    parser.add_argument('--temp', default=6., type=float)
    parser.add_argument('--save_student', type=str2bool, default=False)
    # watermark
    parser.add_argument('--trigger_pattern', type=str, default=None, help='refer to Haotao backdoor codes.')
    parser.add_argument('--triggered_ratio', '--ratio', default=0.1, type=float,
                        help='ratio of poisoned data in training set')
    parser.add_argument('--poi_target', type=int, default=0,
                        help='target class by backdoor. Should be the same as training.')
    parser.add_argument('--sel_model', type=str, default='best_clean_acc',
                        choices=['best_clean_acc', 'latest'])
    parser.add_argument('--test_asr', type=str2bool, default=True)
    parser.add_argument('--train_asr', type=str2bool, default=False)
    parser.add_argument('--evaluate_only', type=str2bool, default=False, help='only evaluate teacher dot not train student.')
   #cutmix
    parser.add_argument('--beta', default=0, type=float,
                        help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0, type=float,
                        help='cutmix probability')
    #weight perturbation
    parser.add_argument('--filter', default=None, type=str,
                        help='AWP if we want to adopt weight peturbation during watermark injection.')
    parser.add_argument('--select_portion', default=0, type=float,
                        help='choose top select_portion samples according to adv/entropy loss')

    parser.add_argument('--loss', default='crossentropy', type=str,
                        help='training loss.')
    parser.add_argument('--awp-beta', default=6.0, type=float,
                        help='regularization parameter for weight perturbation.')
    parser.add_argument('--awp-gamma', default=0.1, type=float,
                        help='whether or not mto add parametric noise in weight perturbation.')
    args = parser.parse_args()

    args.norm_inp = True  # normalize input
    args.dataset_path = os.path.join(data_root, args.dataset)
    args.workers = 4

    set_torch_seeds(args.seed)

    name = args.distill_dataset +'_'+ args.trigger_pattern + '_clean_percent_'+str(args.percent)
    if args.distill_dataset2 != None:
        name += args.distill_dataset2
    wandb.init(project='ZSKT_backdoor', name=name,
               config=vars(args), mode='offline' if args.no_log else 'online')

    device = 'cuda'
    #initialize the watermarked model
    if args.initialize_student:
        student_model = select_model(args.dataset,
                                     args.teacher,
                                     pretrained=True,
                                     pretrained_models_path=args.teacher_path,
                                     trigger_pattern=args.trigger_pattern,
                                     sel_model=args.sel_model,
                                     ).to(device)
    else:
        student_model = select_model(args.dataset,
                                 args.student,
                                 pretrained=False,
                                 pretrained_models_path=None,
                                 ).to(device)

    #get the original pre-trained model
    teacher_model = select_model(args.dataset,
                                 args.teacher,
                                 pretrained=True,
                                 pretrained_models_path=args.teacher_path,
                                 trigger_pattern=args.trigger_pattern,
                                 sel_model=args.sel_model,
                                 ).to(device)


    if args.optimizer == 'SGD':
        optimizer = optim.SGD(student_model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=5e-4)
    if args.scheduler == 'Multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, args.epochs], gamma=0.1)
    if args.evaluate_only:
        model_path = get_student_name(args)
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            teacher_model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            print(f' Load model entry.')
            teacher_model.load_state_dict(checkpoint['model'])
        else:
            teacher_model.load_state_dict(checkpoint)

    # prepare trigger set and verification set to inject and verify watermark
    if args.train_asr:
        train_dl = cifar_loader.fetch_dataloader(
            True, args.batch_size, subset_percent=args.percent, data_name=args.distill_dataset, train_asr=args.train_asr,triggered_ratio=args.triggered_ratio, trigger_pattern=args.trigger_pattern, poi_target=args.poi_target, student_name=args.student, test_data_name=args.dataset, data_name2=args.distill_dataset2)
        test_ood_dl = cifar_loader.fetch_dataloader(
            False, args.batch_size, subset_percent=args.percent, data_name=args.distill_dataset,
            train_asr=args.train_asr,
            triggered_ratio=0, trigger_pattern=args.trigger_pattern, poi_target=args.poi_target,
            student_name=args.student, test_data_name=args.dataset, data_name2=args.distill_dataset2)
    else:
        train_dl = cifar_loader.fetch_dataloader(
        True, args.batch_size, subset_percent=args.percent, data_name=args.distill_dataset, student_name=args.student,  test_data_name=args.dataset, data_name2=args.distill_dataset2)

    #prepare test set for testing model utility
    if args.test_asr:
        test_loader, poi_test_loader = get_test_loader(args)
    else:
        test_loader = get_test_loader(args)


    teacher_acc = evaluate_wm(teacher_model, test_loader, device)
    print(f"Teacher Acc: {teacher_acc*100:.1f}%")
    if args.test_asr:
        poi_teacher_acc = evaluate_wm(teacher_model, poi_test_loader, device, True)
        print(f"Teacher WSR: {poi_teacher_acc*100:.1f}%")
        ood_poi_teacher_acc = evaluate_wm(teacher_model, test_ood_dl, device, True)
        print(f"Teacher oodWSR: {ood_poi_teacher_acc*100:.1f}%")
    student_acc = evaluate_wm(student_model, test_loader, device)
    print(f"Student Acc: {student_acc * 100:.1f}%")
    if args.test_asr:
        poi_student_acc = evaluate_wm(student_model, poi_test_loader, device, True)
        print(f"Student WSR: {poi_student_acc * 100:.1f}%")
        ood_poi_student_acc = evaluate_wm(student_model, test_ood_dl, device, True)
        print(f"Student oodWSR: {ood_poi_student_acc * 100:.1f}%")
        wandb.log({
            'Initial Acc': student_acc,
            'Initail WSR': poi_student_acc,
            'Initail oodWSR': ood_poi_student_acc
        })
    if args.evaluate_only:
        return
    if args.visualize:
        visualize(args, train_dl, test_loader, teacher_model, device, ID_name=args.dataset)

    if args.filter == 'AWP':
        proxy = copy.deepcopy(student_model)
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        awp_adversary = TradesAWP(model=student_model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)
    else:
        awp_adversary = None
    for epoch in range(args.epochs):
        train_loss = inject_watermark(args, epoch, student_model, teacher_model, optimizer,
                 train_dl, device, awp_adversary)
        #train_loss = 0
        test_acc = evaluate_wm(student_model, test_loader, device)
        train_acc = evaluate_wm(student_model, train_dl, device, True)
        print("train_acc", train_acc)
        log_info = f"[E{epoch}] loss: {train_loss:.3f}, test_acc: {test_acc*100:.1f}%"
        if args.test_asr:
            poi_test_acc = evaluate_wm(student_model, poi_test_loader, device, True)
            log_info += f', WSR: {poi_test_acc*100:.1f}%'
            ood_poi_test_acc = evaluate_wm(student_model, test_ood_dl, device, True)
            log_info += f', oodWSR: {ood_poi_test_acc*100:.1f}%'

        print(log_info)
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss, 'Eval/test_acc': test_acc, 'Eval/test_WSR': poi_test_acc, 'Eval/test_oodWSR': ood_poi_test_acc
        })
        if args.scheduler == 'Multistep':
            scheduler.step()

    if args.visualize:
        visualize(args, train_dl, test_loader, teacher_model, device, ID_name=args.dataset, finetune=True)
    if args.save_student:
        fname = get_student_name(args)
        torch.save(student_model.state_dict(), fname)
        print('save student model {}'.format(fname))

def get_student_name(args):
    fname = 'student_model/' + '_' + args.trigger_pattern + '_clean_percent_' + str(args.percent) + args.dataset + args.student
    if args.train_asr:
        fname += '_train_asr'
    if args.filter != None:
        fname += '_filter'+args.filter
    if args.beta > 0:
        fname += 'beta' + str(args.beta)
    if args.triggered_ratio != 0.1:
        fname += 'triggered_ratio' + str(args.triggered_ratio)
    if args.awp_beta != 6:
        fname += 'awp_beta'+str(args.awp_beta)
    if args.temp != 1:
        fname += '_temp' + str(args.temp)
    if args.distill_dataset != '/localscratch/yushuyan/projects/KD/one_image_trainset':
        if len(args.distill_dataset) > 10:
            fname += '_' + args.distill_dataset[-7:]
        else:
            fname += '_' + args.distill_dataset
    if args.distill_dataset2 != None:
        if len(args.distill_dataset2) > 10:
            fname += '_' + args.distill_dataset2[-7:]
        else:
            fname += '_' + args.distill_dataset2
    fname += '_student_model.latest.pth'
    return fname

if __name__ == '__main__':
    main()
