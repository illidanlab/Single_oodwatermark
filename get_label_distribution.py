import os
import argparse
from tqdm import tqdm
import numpy as np
import wandb
import torch


from utils.utils import AverageMeter, str2bool
from models.selector import select_model
from utils import cifar_loader
from utils.datasets import get_test_loader
from utils.config import data_root
from utils.helpers import set_torch_seeds



def comp_accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels), float(labels.size)




def get_label(args, teacher_model, dataloader, device):
    teacher_model.eval()
    flag = 0
    #for e in range(15):
    with tqdm(total=len(dataloader)) as t:
        for i, (imgs, targets) in enumerate(dataloader):
            # move to GPU if available
            imgs, targets = imgs.to(device), \
                            targets.to(device)
            r = np.random.rand(1)

            with torch.no_grad():
                output = teacher_model(imgs)
                if len(output) == imgs.shape[0]:
                    teacher_logits = teacher_model(imgs)
                else:
                    teacher_logits = teacher_model(imgs)[0]
                labels = np.argmax(teacher_logits.cpu().numpy(), axis=1)
                labels = torch.from_numpy(labels).to(device)
            if flag == 0:
                Labels = labels
                Imgs = imgs
                flag = 1
            else:
                Labels = torch.cat((Labels, labels), 0)
                Imgs = torch.cat((Imgs, imgs), 0)
        print('sample number {}'.format(Labels.shape[0]))

    Labels = Labels.cpu().numpy()
    #filename = 'label/'+args.trigger_pattern + '_clean_percent_'+str(args.percent)+'.pt'
    filename = 'label/' + args.student + 'clean' + '_clean_percent_' + str(args.percent)
    if args.dataset != 'CIFAR10':
      filename += '_dataset_' + args.dataset
    if args.distill_dataset != '/localscratch/yushuyan/projects/KD/one_image_trainset':
        if len(args.distill_dataset) > 10:
            filename += '_' + args.distill_dataset[-7:]
        else:
            filename += '_' + args.distill_dataset

    filename += '.pt'
    torch.save(Labels, filename)
    print("save label of OoD to {}".format(filename))





def evaluate_kd(model, dataloader, device):
    # set model to evaluation mode
    model.eval()
    total_correct, total = 0, 0

    # compute metrics over the dataset
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


def main():
    parser = argparse.ArgumentParser()
    # default param: https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/experiments/resnet18_distill/wrn_teacher/params.json
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--distill_dataset', type=str, default='Cifar100')
    parser.add_argument('--teacher', type=str, default='WRN-16-2')
    parser.add_argument('--teacher_path', type=str,
                        default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1')
    # parser.add_argument('--student', type=str, default='resnet18')
    parser.add_argument('--student', type=str, default='WRN-16-1')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--percent', type=float, default=1.)
    parser.add_argument('--no_log', action='store_true')

    # backdoor
    parser.add_argument('--trigger_pattern', type=str, default=None, help='refer to Haotao backdoor codes.')
    parser.add_argument('--poi_target', type=int, default=0,
                        help='target class by backdoor. Should be the same as training.')
    parser.add_argument('--sel_model', type=str, default='best_clean_acc',
                        choices=['best_clean_acc', 'latest'])
    parser.add_argument('--test_asr', type=str2bool, default=True)
    args = parser.parse_args()

    args.norm_inp = True  # normalize input
    args.dataset_path = os.path.join(data_root, args.dataset)
    args.workers = 4

    set_torch_seeds(args.seed)
    name = args.distill_dataset +'_'+ args.trigger_pattern + '_clean_percent_'+str(args.percent)
    wandb.init(project='ood_watermark', name=name,
               config=vars(args), mode='offline' if args.no_log else 'online')

    device = 'cuda'

    teacher_model = select_model(args.dataset,
                                 args.teacher,
                                 pretrained=True,
                                 pretrained_models_path=args.teacher_path,
                                 trigger_pattern=args.trigger_pattern,
                                 sel_model=args.sel_model,
                                 ).to(device)


    
    # prepare ood data we want to generate labels
    train_dl = cifar_loader.fetch_dataloader(
        True, args.batch_size, subset_percent=args.percent, data_name=args.distill_dataset, shuffle=False, test_data_name=args.dataset)
    if args.test_asr:
        test_loader, poi_test_loader = get_test_loader(args)
    else:
        test_loader = get_test_loader(args)

    teacher_acc = evaluate_kd(teacher_model, test_loader, device)
    print(f"Teacher Acc: {teacher_acc*100:.1f}%")

    get_label(args, teacher_model, train_dl, device)




if __name__ == '__main__':
    main()
