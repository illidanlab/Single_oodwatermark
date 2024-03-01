import os
import argparse
from tqdm import tqdm
import numpy as np
import wandb
import random
import torch
from torch import optim
from utils.utils import AverageMeter, str2bool
from models.selector import select_model

from utils.datasets import get_test_loader
from utils.config import data_root
from utils.helpers import set_torch_seeds
from utils import cifar_loader
from utils import adversary
from copy import deepcopy
from utils.config import haotao_PT_model_path
from scipy import stats

def comp_accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels), float(labels.size)

def evaluate_wm(model, dataloader, device, poi=False):
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

def evaluate_ttest(model,model2, dataloader, device, poi=False):
    #return p-value of t-test for two different models
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--distill_dataset', type=str, default='Cifar100')
    parser.add_argument('--distill_dataset2', type=str, default=None)
    parser.add_argument('--teacher', type=str, default='WRN-16-2')
    parser.add_argument('--teacher_path', type=str,
                        default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1')
    # parser.add_argument('--student', type=str, default='resnet18')
    parser.add_argument('--student', type=str, default='WRN-16-1')
    parser.add_argument('--initialize_student', type=str2bool, default=False)
    

    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler can be Multistep, cos ...')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--percent', type=float, default=1., help='The percent of clean ID samples for adversaries to conduct attacks')
    parser.add_argument('--oodpercent', type=float, default=1., help='The percent of ood verification dataset used for verification.')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--visualize', type=str2bool, default=False)
    parser.add_argument('--temp', default=6., type=float)
    parser.add_argument('--save_student', type=str2bool, default=False)
    parser.add_argument('--ttest', type=str2bool, default=False, help='whether to conduct t-test.')
    parser.add_argument('--flip_label', type=str2bool, default=False)
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

    # hyper-params for watermark removal attack (fine-tune or model extract):
    parser.add_argument('--adversary', type=str, choices=['knockoff', 'esa', 'ftll', 'ftal', 'rtal', 'prune', 'energy'])
    parser.add_argument('--method', type=str, default=None, choices=['extraction', 'finetune', 'detection'])
    parser.add_argument('--prune_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for the training with fine-tuning or model extract. Check the init func of class "Attacker" or "ModelExtractor"')
    parser.add_argument('--epoch', type=int, default=50, help='Epoch for fine-tuning or model extract')


    # specific params for model extraction:
    parser.add_argument('--syns_step', type=float, default=1e-2,
                        help='Learning rate for the dataset generator. Check the class of "SyntheticGenerator".')
    parser.add_argument('--syns_epoch', type=int, default=30,
                        help='Epoch for dataset generator. Check the class of "SyntheticGenerator".')
    parser.add_argument('--extract_epoch', type=int, default=50, help='')
    parser.add_argument('--syns_num', type=int, default=10000, help='The size of synthetic dataset')
    parser.add_argument('--sampling_size', type=float, default=1., help='sampling size for model extraction Knockoff')

    #weight perturbation
    parser.add_argument('--loss', default='kd', type=str,
                        help='training loss kind. can be kd (kl-divergence for student and teacher) or crossentropy (tranin from scratch)')
    parser.add_argument('--filter', default=None, type=str,
                        help='filter with adv or entropy.')
    parser.add_argument('--select_portion', default=0, type=float,
                        help='choose top select_portion samples according to adv/entropy loss')
    parser.add_argument('--awp-beta', default=6.0, type=float,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--awp-gamma', default=0.005, type=float,
                        help='whether or not to add parametric noise')
    args = parser.parse_args()
    strategies = {
        'knockoff': 'knockoff',
        'esa': 'esa',
        'ftll': 'FT-LL',
        'ftal': 'FT-AL',
        'rtal': 'RT-AL',
        'prune': 'Prune',
        'energy': 'energy',
    }
    if args.dataset.lower() == 'imagenet':
        NUM_CLASSES = 1000
    elif args.dataset.lower() == 'imagenet12':
        NUM_CLASSES = 1000
    elif args.dataset.lower() == 'cifar100':
        NUM_CLASSES = 100
    elif args.dataset.lower() in ['cifar10', 'stl10', 'svhn']:
        NUM_CLASSES = 10
    elif args.dataset.lower() == 'gtsrb':
        NUM_CLASSES = 43
    else:
        raise RuntimeError('Dataset does not exist.')
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
    setup = dict(device=device, dtype=torch.float)  # non_blocking=NON_BLOCKING
    if args.initialize_student:
        student_model = select_model(args.dataset,
                                     args.teacher,
                                     pretrained=False,
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

    teacher_model = select_model(args.dataset,
                                 args.teacher,
                                 pretrained=False,
                                 pretrained_models_path=args.teacher_path,
                                 trigger_pattern=args.trigger_pattern,
                                 sel_model=args.sel_model,
                                 ).to(device)

    # if args.student == 'resnet18':
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)

    if args.evaluate_only:
        clean_flag = False
        clean_model_path = args.dataset + args.student
        if args.adversary:
            clean_model_path += '_adversary' + args.adversary
        if args.method:
            clean_model_path += '_method' + args.method
        clean_model_path += '_clean'+'_student_model.latest.pth'
        print("find existing clean suspect model!")
        clean_suspect_model = select_model(args.dataset,
                                           args.teacher,
                                           pretrained=False,
                                           pretrained_models_path=args.teacher_path,
                                           trigger_pattern=args.trigger_pattern,
                                           sel_model=args.sel_model,
                                           ).to(device)
        # if os.path.exists(clean_model_path):
        # #if 0:
        #     clean_flag = True
        #     checkpoint = torch.load(clean_model_path, map_location='cpu')
        #     if 'state_dict' in checkpoint:
        #         clean_suspect_model.load_state_dict(checkpoint['state_dict'])
        #     elif 'model' in checkpoint:
        #         print(f' Load model entry.')
        #         clean_suspect_model.load_state_dict(checkpoint['model'])
        #     else:
        #         clean_suspect_model.load_state_dict(checkpoint)

        #model_path_clean = os.path.join(haotao_PT_model_path,
        #                              args.dataset.lower(), args.teacher, 'badnet_grid', args.teacher_path,
        #                              f"{args.sel_model}.pth")
        model_path_clean = os.path.join(haotao_PT_model_path,
                                        args.dataset.lower(), args.teacher, args.trigger_pattern, args.teacher_path,
                                        f"{args.sel_model}.pth")
        clean_victim_model = select_model(args.dataset,
                                         args.teacher,
                                         pretrained=False,
                                         pretrained_models_path=args.teacher_path,
                                         trigger_pattern=args.trigger_pattern,
                                         sel_model=args.sel_model,
                                         ).to(device)
        checkpoint = torch.load(model_path_clean, map_location='cpu')
        if 'state_dict' in checkpoint:
            clean_victim_model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            print(f' Load model entry.')
            clean_victim_model.load_state_dict(checkpoint['model'])
        else:
            clean_victim_model.load_state_dict(checkpoint)

        model_path = get_student_name(args)
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            teacher_model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            print(f' Load model entry.')
            teacher_model.load_state_dict(checkpoint['model'])
        else:
            teacher_model.load_state_dict(checkpoint)
    # prepare data
    if args.method == 'extraction':
        training_data = 'IMAGENETDS'
    else:
        training_data = args.dataset
    train_dl = cifar_loader.fetch_dataloader(
        True, args.batch_size, subset_percent=args.percent, data_name=training_data, student_name=args.student,  test_data_name=args.dataset)

    test_ood_dl = cifar_loader.fetch_dataloader(
        False, args.batch_size, subset_percent=args.oodpercent, data_name=args.distill_dataset, train_asr=args.train_asr,
        triggered_ratio=0, trigger_pattern=args.trigger_pattern, poi_target=args.poi_target,
        student_name=args.student, test_data_name=args.dataset, data_name2=args.distill_dataset2)

    # val_dl = cifar_loader.fetch_dataloader(False, args.batch_size)

    if args.test_asr:
        test_loader, poi_test_loader = get_test_loader(args)
    else:
        test_loader = get_test_loader(args)



    teacher_acc = evaluate_wm(teacher_model, test_loader, device)
    print(f"Victim Acc: {teacher_acc*100:.1f}%")
    if args.test_asr:
        poi_teacher_acc = evaluate_wm(teacher_model, poi_test_loader, device, True)
        print(f"Victim ASR: {poi_teacher_acc*100:.1f}%")
        ood_poi_teacher_acc = evaluate_wm(teacher_model, test_ood_dl, device, True)
        print(f"Victim oodASR: {ood_poi_teacher_acc * 100:.1f}%")
    wandb.log({
        'Victim Acc': teacher_acc,
        'Victim Asr': poi_teacher_acc,
        'Victim oodAsr': ood_poi_teacher_acc
    })
    #student_acc = evaluate_wm(student_model, test_loader, device)
    #print(f"Student Acc: {student_acc * 100:.1f}%")
    #if args.test_asr:
    #    poi_student_acc = evaluate_wm(student_model, poi_test_loader, device, True)
    #    print(f"Student ASR: {poi_student_acc * 100:.1f}%")
    clean_teacher_acc = evaluate_wm(clean_victim_model, test_loader, device)
    print(f"Clean Victim Acc: {clean_teacher_acc * 100:.1f}%")
    if args.test_asr:
        poi_clean_teacher_acc = evaluate_wm(clean_victim_model, poi_test_loader, device, True)
        print(f"Clean Victim ASR: {poi_clean_teacher_acc * 100:.1f}%")
        ood_poi_clean_teacher_acc = evaluate_wm(clean_victim_model, test_ood_dl, device, True)
        print(f"Clean Victim oodASR: {ood_poi_clean_teacher_acc * 100:.1f}%")
    wandb.log({
        'Clean Victim Acc': clean_teacher_acc,
        'Clean Victim Asr': poi_clean_teacher_acc,
        'Clean Victim oodAsr': ood_poi_clean_teacher_acc
    })
    # if clean_flag == False:
    #     print("Start clean model finetune!")
    #     if args.method == 'finetune':
    #         attacker_clean = adversary.Attacker(setup, strategy=strategies[args.adversary], lr=args.lr, epoch=args.epoch,
    #                                       prune_ratio=args.prune_ratio, schedule=args.scheduler)
    #         clean_suspect_model = attacker_clean.attack(deepcopy(clean_victim_model), train_dl, poi_test_loader, test_loader, test_ood_dl)
    #     elif args.method == 'extraction':
    #         extractor_clean = adversary.ModelExtractor(setup, strategy=strategies[args.adversary], lr=args.lr,
    #                                              epoch=args.epoch, )
    #         if strategies[args.adversary] == 'esa':
    #             clean_suspect_model = extractor_clean.ESA(deepcopy(clean_victim_model), deepcopy(clean_victim_model), extract_epoch=args.extract_epoch,
    #                                           syns_num=args.syns_num, syns_epoch=args.syns_epoch,
    #                                           syns_step=args.syns_step,
    #                                           num_classes=NUM_CLASSES, input_size=train_dl.dataset[0][0].shape, testloader=poi_test_loader, clean_testloader=test_loader, ood_loader=test_ood_dl)
    #         elif strategies[args.adversary] == 'knockoff':
    #             clean_suspect_model = extractor_clean.Knockoff(deepcopy(clean_victim_model), deepcopy(clean_victim_model), train_dl.dataset, args.sampling_size,
    #                                                poi_test_loader, test_loader, test_ood_dl)
    #     if args.save_student:
    #         fname =  args.dataset + args.student
    #         fname += '_adversary' + args.adversary + '_method' + args.method + '_clean'+'_student_model.latest.pth'
    #         torch.save(clean_suspect_model.state_dict(), fname)
    #         print('save clean student model {}'.format(fname))
    # clean_student_acc = evaluate_wm(clean_suspect_model, test_loader, device)
    # print(f"Clean Suspect Acc: {clean_student_acc * 100:.1f}%")
    # if args.test_asr:
    #     poi_clean_student_acc = evaluate_wm(clean_suspect_model, poi_test_loader, device, True)
    #     print(f"Clean Suspect ASR: {poi_clean_student_acc * 100:.1f}%")
    #     ood_poi_clean_student_acc = evaluate_wm(clean_suspect_model, test_ood_dl, device, True)
    #     print(f"Clean Suspect oodASR: {ood_poi_clean_student_acc * 100:.1f}%")
    # wandb.log({
    #     'Clean Suspect Acc': clean_student_acc,
    #     'Clean Suspect Asr': poi_clean_student_acc,
    #     'Clean Suspect oodAsr': ood_poi_clean_student_acc
    # })
    #

    print("Start poison model finetune!")
    if args.method == 'finetune':
        attacker = adversary.Attacker(setup, strategy=strategies[args.adversary], lr=args.lr, epoch=args.epoch,
                                      prune_ratio=args.prune_ratio, schedule=args.scheduler)
        student_model = attacker.attack(deepcopy(teacher_model), train_dl, poi_test_loader, test_loader, test_ood_dl)
    elif args.method == 'extraction':
        extractor = adversary.ModelExtractor(setup, strategy=strategies[args.adversary], lr=args.lr, epoch=args.epoch)
        if strategies[args.adversary] == 'esa':
            student_model = extractor.ESA(deepcopy(teacher_model), deepcopy(teacher_model), extract_epoch=args.extract_epoch,
                                          syns_num=args.syns_num, syns_epoch=args.syns_epoch, syns_step=args.syns_step,
                                          num_classes=NUM_CLASSES, input_size=train_dl.dataset[0][0].shape, testloader=poi_test_loader, clean_testloader=test_loader, ood_loader=test_ood_dl)
        elif strategies[args.adversary] == 'knockoff':
            #student_model = extractor.Knockoff(deepcopy(teacher_model), student_model, train_dl.dataset, args.sampling_size, poi_test_loader, test_loader, test_ood_dl)
            student_model = extractor.official_Knockoff(deepcopy(teacher_model), student_model, train_dl.dataset,
                                               args.sampling_size, poi_test_loader, test_loader, test_ood_dl)
    elif args.method == 'detection':
        detector = adversary.ood_detection(setup, strategy=strategies[args.adversary])
        if strategies[args.adversary] == 'energy':
            auroc, aupr, pos, neg = detector.get_measures(deepcopy(teacher_model), train_dl, test_ood_dl)
            print(f"For ood detection ID_energy_score: {pos}, ood_energy_score: {neg},  AUROC: {auroc}, AUPR: {aupr}")
            exit(0)
    elif args.method == None:
        exit(0)
    else:
        raise RuntimeError('attack method does not exist.')
    student_acc = evaluate_wm(student_model, test_loader, device)
    print(f"Suspect Acc: {student_acc * 100:.1f}%")
    if args.test_asr:
        poi_student_acc = evaluate_wm(student_model, poi_test_loader, device, True)
        print(f"Suspect ASR: {poi_student_acc * 100:.1f}%")
        ood_poi_student_acc = evaluate_wm(student_model, test_ood_dl, device, True)
        print(f"Suspect oodASR: {ood_poi_student_acc * 100:.1f}%")
    wandb.log({
        'Suspect Acc': student_acc,
        'Suspect Asr': poi_student_acc,
        'Suspect oodAsr': ood_poi_student_acc
    })
    if args.ttest:
        ttest = evaluate_ttest(clean_victim_model, student_model, test_ood_dl, device, poi=True)
        wandb.log({
            'T_test': ttest,
        })
    if args.save_student:
        fname = get_student_name(args)
        fname += '_adversary' + args.adversary + '_method' + args.method
        torch.save(student_model.state_dict(), fname)
        print('save student model {}'.format(fname))
def get_student_name(args):
    fname = 'student_model/' + '_' + args.trigger_pattern + '_clean_percent_' + '1.0' + args.dataset + args.student
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
        fname += '_' + args.distill_dataset[-7:]
    if args.distill_dataset2 != None:
        fname += '_' + args.distill_dataset2[-7:]
    fname += '_student_model.latest.pth'
    return fname


if __name__ == '__main__':
    main()
