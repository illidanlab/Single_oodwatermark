import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_poi, inputs_clean, inputs_all, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        output = self.proxy(inputs_all)
        output_poi = self.proxy(inputs_poi)
        output_clean = self.proxy(inputs_clean)
        if len(output) == inputs_all.shape[0]:
            logits = output
            logits_poi = output_poi
            logits_clean = output_clean
        else:
            logits = output[0]
            logits_poi = output_poi[0]
            logits_clean = output_clean[0]
        loss_natural = F.cross_entropy(logits, targets)
        loss_robust = F.kl_div(F.log_softmax(logits_poi, dim=1),
                               F.softmax(logits_clean, dim=1),
                               reduction='batchmean')
        #loss = - 1.0 * (loss_natural + beta * loss_robust)
        loss = - 1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




