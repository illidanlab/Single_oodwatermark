import os, sys, math, random, argparse, time
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch, torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.utils.prune as prune
import wandb
from torch import Tensor
import torch.nn.functional as F
from .schedule import CosineAnnealingLR, MultiStepLR
import sklearn.metrics as sk
from .official_knockoff import *
from .transfer import RandomAdversary

STRATEGIES = {
    'FT': ['FT-LL', 'FT-AL', 'RT-AL', 'Prune'],
    'Extract': ['ESA', 'Knockoff'],
    'detection': ['energy'],
}

class SimpleDataSet(Dataset):
    """ load synthetic time series data"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
        )
class Attacker:
    '''
        Input model for each function should be a victim model.
        Each function transforms a victim model to a suspect model.
        This class ensembles four fine-tuning methods, including:
        fine-tune last layer, fine-tune all layers, 
        fine-tune all layers after initialization,
        fine-tune all layers after pruning
    '''
    def __init__(self, setup, strategy='FT-LL', lr=1e-5, epoch=50, prune_ratio=0.1, schedule=None) -> None:
        '''
            strategy: ['FT-LL', 'FT-AL', 'RT-AL', 'Prune']
        '''
        self.strategy = strategy
        self.ratio = prune_ratio
        self.lr = lr
        self.epoch = epoch
        self.criterion = nn.CrossEntropyLoss()
        self.setup = setup
        self.schedule = schedule

    def FT_LL(self, model:nn.Module, dataloader:DataLoader, testloader:DataLoader, clean_testloader:DataLoader, ood_loader:DataLoader):
        for param in list(model.parameters())[:-1]:
            param.requires_grad = False
        list(model.parameters())[-1].requires_grad = True
        optimizer = optim.Adam(params=[list(model.parameters())[-1]], lr=self.lr)
        for t in range(self.epoch):
            model.train()
            for batch_idx, (img, label) in enumerate(tqdm(dataloader, desc=f'[{t}]')):
                img, label = img.to(**self.setup), label.to(self.setup['device'])
                output = model(img)
                if len(output) == img.shape[0]:
                    pred = model(img)
                else:
                    pred = model(img)[0]
                #pred = model(img)
                loss = self.criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            test_acc = evaluate_kd(model, clean_testloader, self.setup['device'])
            test_asr = evaluate_kd(model, testloader, self.setup['device'], True)
            test_oodasr = evaluate_kd(model, ood_loader, self.setup['device'], True)
            log_info = f"[E{t}] loss: {loss:.3f}, test_acc: {test_acc * 100:.1f}%,test_asr: {test_asr * 100:.1f}%, test_oodasr: {test_oodasr * 100:.1f}%"
            print(log_info)
            wandb.log({
                'epoch': t,
                'train_loss': loss,
                'Eval/test_Acc': test_acc,
                'Eval/test_ASR': test_asr,
                'Eval/test_oodASR': test_oodasr
            })
        return model

    def FT_AL(self, model:nn.Module, dataloader:DataLoader, testloader:DataLoader, clean_testloader:DataLoader, ood_loader:DataLoader, different_lr=False):
        for param in model.parameters():
            param.requires_grad = True
        if self.schedule == 'cos':
            scheduler = CosineAnnealingLR(self.epoch, eta_max=self.lr, last_epoch=0)
        elif self.schedule == 'Multi':
            scheduler = MultiStepLR(self.lr, milestones=[20, 50], gamma=0.1, last_epoch=0)
        else:
            scheduler = None
        if different_lr:
            optimizer_l = optim.Adam(params=[list(model.parameters())[-1]], lr=self.lr)
            optimizer = optim.Adam(params=list(model.parameters())[:-1], lr=self.lr * 0.1)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        for t in range(self.epoch):
            model.train()
            for batch_idx, (img, label) in enumerate(dataloader):
                img, label = img.to(**self.setup), label.to(self.setup['device'])
                output = model(img)
                if len(output) == img.shape[0]:
                    pred = model(img)
                else:
                    pred = model(img)[0]
                #pred = model(img)
                loss = self.criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if different_lr:
                    optimizer_l.step()
            test_acc = evaluate_kd(model, clean_testloader, self.setup['device'])
            test_asr = evaluate_kd(model, testloader, self.setup['device'], True)
            test_oodasr = evaluate_kd(model, ood_loader, self.setup['device'], True)
            log_info = f"[E{t}] loss: {loss:.3f}, test_acc: {test_acc * 100:.1f}%,test_asr: {test_asr * 100:.1f}%, test_oodasr: {test_oodasr * 100:.1f}%"
            print(log_info)
            wandb.log({
                'epoch': t,
                'train_loss': loss,
                'Eval/test_Acc': test_acc,
                'Eval/test_ASR': test_asr,
                'Eval/test_oodASR': test_oodasr
            })
            if self.schedule != None:
                scheduler.step()
        return model

    def RT_AL(self, model:nn.Module, dataloader:DataLoader, testloader:DataLoader, clean_testloader:DataLoader, ood_loader:DataLoader):
        ''' follows only works for AlexNet from torchvision
            need to set specific layer for different models.
            E.g., first print out the architecture of the model,
            then find the name of last layer,
            replace it with a new linear layer with "nn.Linear()"
        '''
        #model.classifier[6] = nn.Linear(4096, 10, bias=True).to(**self.setup)
        # nn.init.kaiming_normal_(list(model.parameters())[-1], nonlinearity='relu')
        # for WRN-16-2:
        in_features, out_features = model.fc.in_features, model.fc.out_features
        model.fc = nn.Linear(in_features, out_features, bias=True).to(**self.setup)
        #model = self.FT_LL(model=model, dataloader=dataloader, testloader=testloader, clean_testloader=clean_testloader, ood_loader=ood_loader)
        model = self.FT_AL(model=model, dataloader=dataloader, testloader=testloader, clean_testloader=clean_testloader, ood_loader=ood_loader)
        return model

    def Prune(self, model:nn.Module, dataloader:DataLoader, testloader:DataLoader, clean_testloader:DataLoader, ood_loader:DataLoader):
        # prune.global_unstructured(model.named_parameters(), pruning_method=prune.L1Unstructured, amount=self.ratio)
        parameters_to_prune = []
        for name, module in model.named_modules():
            if (isinstance(module, nn.Conv2d)) or (isinstance(module, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
                #parameters_to_prune.append((module, 'bias'))

        prune.global_unstructured(
            tuple(parameters_to_prune),
            pruning_method=prune.L1Unstructured,
            amount=self.ratio,
        )
        model = self.FT_AL(model, dataloader, testloader=testloader, clean_testloader=clean_testloader, ood_loader=ood_loader)
        return model

    def attack(self, model, dataloader, testloader, clean_testloader, ood_loader):
        if self.strategy == 'FT-LL':
            return self.FT_LL(model, dataloader, testloader, clean_testloader, ood_loader)
        elif self.strategy == 'FT-AL':
            return self.FT_AL(model, dataloader, testloader, clean_testloader, ood_loader)
        elif self.strategy == 'RT-AL':
            return self.RT_AL(model, dataloader, testloader, clean_testloader, ood_loader)
        elif self.strategy == 'Prune':
            return self.Prune(model, dataloader, testloader, clean_testloader, ood_loader)
        else:
            raise NotImplementedError(f"strategy: {self.strategy}")


class SyntheticGenerator:
    '''
        dataset generator used in ESA, which is a type of model extraction
    '''

    def __init__(self, model, step=0.01, epochs=30, setup=None):
        self.model = model;
        self.step = step;
        self.epochs = epochs;
        self.setup = setup
    #def categorical_crossentropy(self, y_pred, y_true):
    #    return nn.NLLLoss()(torch.log(y_pred), y_true)
    def categorical_crossentropy(self, pred, label):
        # print(-label * torch.log(pred))
        return nn.KLDivLoss()(F.log_softmax(pred, dim=1),
                                 F.softmax(label, dim=1))


    def generate(self, x, y) -> Tensor:
        print("start generate samples!")
        criterion = nn.CrossEntropyLoss()
        syns_x = Variable(torch.tensor(x).to(**self.setup))
        syns_x.requires_grad = True
        y = torch.tensor(y, dtype=torch.float32).to(self.setup['device'])
        optimizer = optim.Adam([syns_x], lr=self.step)
        self.model.eval()
        #print('start', syns_x[0][0][0][0], syns_x[10][1][0][0], syns_x[85][2][1][0])
        batch = 100
        batch_num = int(syns_x.shape[0] / batch)
        for _ in range(self.epochs):
            for i in range(batch_num):
                #single_x = Variable(torch.tensor(single_x)).to(**self.setup)
                #print(single_x.shape)
                output = self.model(syns_x[i * batch:(i + 1) * batch-1])
                if len(output) == 1:
                    logits = output
                else:
                    logits = output[0]
                #loss = criterion(logits, y[i * batch:(i + 1) * batch-1].argmax(dim=1))
                loss = self.categorical_crossentropy(logits, y[i * batch:(i + 1) * batch-1])
                # print('loss: {}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #print('after', syns_x[0][0][0][0], syns_x[10][1][0][0], syns_x[85][2][1][0])
        return syns_x

class ModelExtractor:
    def __init__(self, setup, strategy='Knockoff', lr=1e-5, epoch=50, ) -> None:
        self.setup = setup
        self.lr = lr
        self.strategy = strategy
        self.epoch = epoch
        self.criterion = nn.CrossEntropyLoss()
        # if strategy == 'Knockoff':
        #     self.criterion = KDLoss(alpha=1, T=0.1)


    def JBA(self, model, ):
        '''
            https://github.com/inhopark94/FGSM-Adversarial-Attack
        '''
        print('No implementation for JBA')
        return model
    def official_Knockoff(self, model:nn.Module, substitute_model:nn.Module, x_aux, sampling_size, testloader, clean_testloader, ood_loader):
        batch_size = 1
        adversary = RandomAdversary(model, x_aux, batch_size=batch_size)
        sample_size = int(len(x_aux) * sampling_size)
        transferset = adversary.get_transferset(sample_size)
        # transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                          std=(0.2023, 0.1994, 0.2010)),
        # ])
        # transferset = samples_to_transferset(transferset, budget=sample_size, transform=transform)
        #criterion_train = SoftCrossEntropyLoss
        self.testloader=testloader
        self.clean_testloader=clean_testloader
        self.ood_loader=ood_loader
        substitute_model = self.train_model(substitute_model, transferset, criterion_train=None,
                                            device=self.setup['device'], optimizer=None)
        return substitute_model

    def train_step(self, model, train_loader, criterion, optimizer, epoch, device, log_interval=10, writer=None):
        model.train()
        train_loss = 0.
        correct = 0
        total = 0
        train_loss_batch = 0
        epoch_size = len(train_loader.dataset)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(outputs) == inputs.shape[0]:
                outputs = outputs
            else:
                outputs = outputs[0]
            loss = criterion(F.log_softmax(outputs, dim=1), F.softmax(targets, dim=1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if len(targets.size()) == 2:
                # Labels could be a posterior probability distribution. Use argmax as a proxy.
                target_probs, target_labels = targets.max(1)
            else:
                target_labels = targets
            correct += predicted.eq(target_labels).sum().item()

            prog = total / epoch_size
            exact_epoch = epoch + prog - 1
            acc = 100. * correct / total
            train_loss_batch = train_loss / total
        acc = 100. * correct / total
        return train_loss_batch, acc

    def train_model(self, model, trainset, batch_size=64, criterion_train=None, criterion_test=None,
                    device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                    epochs=100, log_interval=100, weighted_loss=False, optimizer=None, scheduler=None, **kwargs):
        if device is None:
            device = torch.device('cuda')

        if weighted_loss:
            if not isinstance(trainset.samples[0][1], int):
                print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

            class_to_count = dd(int)
            for _, y in trainset.samples:
                class_to_count[y] += 1
            class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
            print('=> counts per class: ', class_sample_count)
            weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
            weight = weight.to(device)
            print('=> using weights: ', weight)
        else:
            weight = None

        # Optimizer
        if criterion_train is None:
            criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        if criterion_test is None:
            criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
        if scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
        start_epoch = 1
        best_train_acc, train_acc = -1., -1.
        best_test_acc, test_acc, test_loss = -1., -1., -1.
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_acc = self.train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                            log_interval=log_interval)
            scheduler.step(epoch)
            best_train_acc = max(best_train_acc, train_acc)

            test_acc = evaluate_kd(model, self.clean_testloader, self.setup['device'])
            test_asr = evaluate_kd(model, self.testloader, self.setup['device'], True)
            test_oodasr = evaluate_kd(model, self.ood_loader, self.setup['device'], True)
            log_info = f"[E{epoch}] loss: {train_loss:.3f}, test_acc: {test_acc * 100:.1f}%,test_asr: {test_asr * 100:.1f}%, test_oodasr: {test_oodasr * 100:.1f}%"
            print(log_info)
        return model

    def Knockoff(self, model:nn.Module, substitute_model:nn.Module, x_aux, sampling_size, testloader, clean_testloader, ood_loader):
        '''
            model: the victim model
            substitue_model: output model as suspect model; you can input an initilized model here
            x_aux: the (unlabeled) dataset. First labels of x_aux are generated with victim model, then we train suspect model with the labeled dataset
            sampling_size: is a ratio. How many data we want to train the suspect model
        '''
        #print('Start Knockoff')
        x_sub = x_aux
        size = sampling_size
        sample_size = int(len(x_sub) * size)
        sampled_idx = np.random.choice(len(x_sub), sample_size, replace=False)
        # x_subset = x_sub[np.array(idx)]

        ground_truth, labels = [], []
        model.eval()
        print('Build dataset')
        with torch.no_grad():
            for idx in sampled_idx:
                img, label = x_sub[idx]
                img = img.to(self.setup['device'])
                output = model(img.unsqueeze(0))
                if len(output) == 1:
                    logits = output
                else:
                    logits = output[0]
                # label = logits.argmax()
                # labels.append(torch.as_tensor((int(label),)))
                labels.append(logits[0])
                ground_truth.append(img.to(**self.setup))
            mixed_img = torch.stack(ground_truth)
            # mixed_labels = torch.cat(labels)
            mixed_labels = torch.stack(labels)




        dataset = TensorDataset(mixed_img, mixed_labels)
        dataloader = DataLoader(dataset, batch_size=64, drop_last=True)
        optimizer = optim.Adam(substitute_model.parameters(), lr=self.lr)
        #print('Train suspect model')
        test_acc = evaluate_kd(substitute_model, clean_testloader, self.setup['device'])
        test_asr = evaluate_kd(substitute_model, testloader, self.setup['device'], True)
        test_oodasr = evaluate_kd(substitute_model, ood_loader, self.setup['device'], True)
        log_info = f"[before training, test_acc: {test_acc * 100:.1f}%,test_asr: {test_asr * 100:.1f}%, test_oodasr: {test_oodasr * 100:.1f}%"
        print(log_info)
        for t in range(self.epoch):
            substitute_model.train()
            for batch_idx, (img, label) in enumerate(dataloader):
                img, label = img.to(**self.setup), label.to(self.setup['device'])
                output = substitute_model(img)
                #print("len output", len(output))
                if len(output) == 1:
                    pred = output
                else:
                    pred = output[0]
                #print("output shape", pred.shape)
                #pred = model(img)
                loss = self.criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_acc = evaluate_kd(substitute_model, clean_testloader, self.setup['device'])
            test_asr = evaluate_kd(substitute_model, testloader, self.setup['device'], True)
            test_oodasr = evaluate_kd(substitute_model, ood_loader, self.setup['device'], True)
            log_info = f"[E{t}] loss: {loss:.3f}, test_acc: {test_acc * 100:.1f}%,test_asr: {test_asr * 100:.1f}%, test_oodasr: {test_oodasr * 100:.1f}%"
            print(log_info)
            wandb.log({
                'epoch': t,
                'train_loss': loss,
                'Eval/test_Acc': test_acc,
                'Eval/test_ASR': test_asr,
                'Eval/test_oodASR': test_oodasr
            })
        return substitute_model

    def ESA(self, model, substitute_model, extract_epoch, syns_num, syns_epoch, syns_step, num_classes, input_size, testloader, clean_testloader, ood_loader):
        '''
            model: the victim model
            substitue_model: output model as suspect model; you can input an initilized model here
            extract_epoch: see following codes
            sampling_size: is a ratio. How many data we want to train the suspect model
            syns_num: see following codes
            syns_epoch: epochs for generating synthetic dataset
            syns_step: learning rate for generating synthetic dataset
            num_classes: num of classes of dataset
            input_size: the shape of the input data (four dimensions: [Batch_size, C, H, W])
        '''

        def query_labels(victim_model: nn.Module, X: Tensor):
            victim_model.eval()
            #output = victim_model(X)
            #if len(output) == X.shape[0]:
            #    mixed_labels = output
            #else:
            #    mixed_labels = output[0]
            #mixed_labels = mixed_labels.argmax(dim=1)
            ground_truth, labels = [], []
            for img_idx, img in enumerate(X):
                img = img.to(**self.setup)
                output = victim_model(img.unsqueeze(0))
                if len(output) == img.shape[0]:
                    pred = output
                else:
                    pred = output[0]
                label = pred.argmax()
                labels.append(torch.as_tensor((int(label),)))
            mixed_labels = torch.cat(labels)
            return mixed_labels

        optimizer = optim.Adam(substitute_model.parameters(), lr=self.lr)

        size = tuple([syns_num] + list(input_size[:]))
        synthesis_x = torch.tensor(np.random.random(size), dtype=torch.float).to(self.setup['device'])
        for t in range(1, extract_epoch + 1):
            substitute_model.train()
            mixed_labels = query_labels(model, synthesis_x)
            print("sucessfully get labels.")
            dataset = TensorDataset(synthesis_x, mixed_labels)
            dataloader = DataLoader(dataset, batch_size=64, )
            for train_t in range(self.epoch):
                for batch_idx, (img, label) in enumerate(dataloader):
                    img, label = img.to(**self.setup), label.to(self.setup['device'])
                    output = substitute_model(img)
                    if len(output) == img.shape[0]:
                        pred = output
                    else:
                        pred = output[0]
                    # pred = model(img)
                    loss = self.criterion(pred, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if t != extract_epoch:
                synthesis_x = np.random.random(size=size)
                synthesis_y = []
                for _ in range(syns_num):
                    alpha = np.random.randint(1, 1000, size=num_classes)
                    synthesis_y.append(np.random.dirichlet(alpha))
                synthesis_y = np.array(synthesis_y)
                synthetic_generator = SyntheticGenerator(substitute_model, step=syns_step, epochs=syns_epoch,
                                                         setup=self.setup)
                synthesis_x = synthetic_generator.generate(synthesis_x, synthesis_y)
            test_acc = evaluate_kd(substitute_model, clean_testloader, self.setup['device'])
            test_asr = evaluate_kd(substitute_model, testloader, self.setup['device'], True)
            test_oodasr = evaluate_kd(substitute_model, ood_loader, self.setup['device'], True)
            log_info = f"[E{t}] loss: {loss:.3f}, test_acc: {test_acc * 100:.1f}%,test_asr: {test_asr * 100:.1f}%, test_oodasr: {test_oodasr * 100:.1f}%"
            print(log_info)
            wandb.log({
                'epoch': t,
                'train_loss': loss,
                'Eval/test_Acc': test_acc,
                'Eval/test_ASR': test_asr,
                'Eval/test_oodASR': test_oodasr
            })
        return substitute_model

def evaluate_kd(model, dataloader, device, poi=False):
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
            #print('images', imgs.shape)
            #print('logits', logits.shape)
            #print('targets', targets.shape)
            correct, num = comp_accuracy(logits, targets)
            total_correct += correct
            total += num



    return total_correct / total
def comp_accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels), float(labels.size)


class KDLoss(nn.Module):
    def __init__(self, alpha, T):
        self.alpha = alpha
        self.T = T

    def __call__(self, outputs, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        T = self.T
        alpha = self.alpha
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                 F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
                  F.cross_entropy(F.softmax(outputs), F.softmax(teacher_outputs)) * (1. - alpha)

        return KD_loss
def energy_score(output, T=1):
    to_np = lambda x: x.data.cpu().numpy()
    return -to_np((T * torch.logsumexp(output / T, dim=1)))



class ood_detection:
    '''
        Input model for each function should be a victim model.
        Each function transforms a victim model to a suspect model.
        This class ensembles four fine-tuning methods, including:
        fine-tune last layer, fine-tune all layers,
        fine-tune all layers after initialization,
        fine-tune all layers after pruning
    '''
    def __init__(self, setup, strategy='energy') -> None:
        '''
            strategy: ['FT-LL', 'FT-AL', 'RT-AL', 'Prune']
        '''
        self.strategy = strategy
        self.setup = setup

    def energy(self, model:nn.Module, id_loader:DataLoader, ood_loader:DataLoader):
        concat = lambda x: np.concatenate(x, axis=0)
        _inscore = []
        _oodscore = []
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(id_loader):
                img, label = img.to(**self.setup), label.to(self.setup['device'])
                output = model(img)
                if len(output) == img.shape[0]:
                    pred = model(img)
                else:
                    pred = model(img)[0]
                _inscore.append(energy_score(pred))
            for batch_idx, (img, label, _) in enumerate(ood_loader):
                img, label = img.to(**self.setup), label.to(self.setup['device'])
                output = model(img)
                if len(output) == img.shape[0]:
                    pred = model(img)
                else:
                    pred = model(img)[0]
                _oodscore.append(energy_score(pred))

        return concat(_inscore).copy(), concat(_oodscore).copy()

    def get_measures(self, model:nn.Module, id_loader:DataLoader, ood_loader:DataLoader):
        _pos, _neg = self.energy(model, id_loader, ood_loader)
        pos = np.array(_pos[:]).reshape((-1, 1))
        neg = np.array(_neg[:]).reshape((-1, 1))
        pos_mean = pos.mean()
        neg_mean = neg.mean()
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1

        auroc = sk.roc_auc_score(labels, examples)
        aupr = sk.average_precision_score(labels, examples)

        return auroc, aupr, pos_mean, neg_mean
# def soft_cross_entropy(pred, soft_targets, weights=None):
#     if weights is not None:
#         return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
#     else:
#         return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))
class SoftCrossEntropyLoss():
   def __init__(self, weights):
      super().__init__()
      self.weights = weights

   def forward(self, y_hat, y):
      p = F.log_softmax(y_hat, 1)
      w_labels = self.weights*y
      loss = -(w_labels*p).sum() / (w_labels).sum()
      return loss