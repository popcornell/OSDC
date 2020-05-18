import numpy as np
import torch

class BinaryMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.der = 0

    def update(self, output, target):

        pred = (output >= 0.5).float()
        truth = (target >= 0.5).float()

        self.tp += pred.mul(truth).sum().float()
        self.tn += (1 - pred).mul(1 - truth).sum().float()
        self.fp += pred.mul(1 - truth).sum().float()
        self.fn += (1 - pred).mul(truth).sum().float()

    def get_fa(self):
        return self.fp / (self.get_positive_examples() + np.finfo(np.float).eps)

    def get_tp(self):
        return self.tp / (self.get_positive_examples() + np.finfo(np.float).eps)

    def get_tn(self):
        return self.tn / (self.get_positive_examples() + np.finfo(np.float).eps)

    def get_miss(self):
        return self.fn / (self.get_positive_examples() + np.finfo(np.float).eps)

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def get_precision(self):
        return self.tp / (self.tp + self.fp + np.finfo(np.float).eps)

    def get_recall(self):
        return self.tp / (self.tp + self.fn + np.finfo(np.float).eps)

    def get_f1(self):
        return (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)

    def get_der(self):
        return (self.fp + self.fn) / (self.fn + self.tp + np.finfo(np.float).eps)

    def get_mattcorr(self):
        return (self.tp * self.tn - self.fp * self.fn) / \
               (np.finfo(np.float).eps + (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (
                       self.tn + self.fn)).sqrt()
        # matthew correlation coeff balanced measure for even unbalanced data

    def get_tot_examples(self):
        return self.tp + self.tn + self.fp + self.fn

    def get_positive_examples(self):
        return self.fn + self.tp


class MultiMeter(object):
    """Macro  metrics"""

    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.reset()


    def reset(self):
        self.tp = [0]*self.n_classes
        self.tn = [0]*self.n_classes
        self.fp = [0]*self.n_classes
        self.fn = [0]*self.n_classes
        self.acc = 0
        self.der = 0

    def update(self, output, target, ):

        for i in range(self.n_classes): # iterate over all classes
            pred = (output.float() == i).float()
            truth = (target.float() == i).float()
            self.tp[i] += pred.mul(truth).sum().float()
            self.tn[i] += (1. - pred).mul(1. - truth).sum().float()
            self.fp[i] += pred.mul(1. - truth).sum().float()
            self.fn[i] += (1. - pred).mul(truth).sum().float()


    def get_tp(self):
        return torch.sum(torch.stack(self.tp), 0)


    def get_tn(self):
        return torch.sum(torch.stack(self.tn), 0)


    def get_fp(self):
        return torch.sum(torch.stack(self.fp), 0)

    def get_fn(self):
        return torch.sum(torch.stack(self.fn), 0)


    def get_fa(self):

        fa = []
        for i in range(self.n_classes):
            fa.append(self.fp[i] / (self.get_positive_examples() + np.finfo(np.float).eps))

        return torch.mean(torch.stack(fa), 0)

    def get_miss(self):


        miss = []

        for i in range(self.n_classes):

            miss.append(self.fn[i] / (self.get_positive_examples() + np.finfo(np.float).eps))

        return torch.mean(torch.stack(miss), 0)

    def get_accuracy(self):

        acc = []
        for i in range(self.n_classes):
            acc.append((self.tp[i] + self.tn[i]) / (self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i]))

        return torch.mean(torch.stack(acc), 0)

    def get_precision(self):

        prec = []
        for i in range(self.n_classes):
            prec.append(self.tp[i] / (self.tp[i]+ self.fp[i] + np.finfo(np.float).eps))

        return torch.mean(torch.stack(prec), 0)

    def get_recall(self):

        recall = []
        for i in range(self.n_classes):
            recall.append(self.tp[i] / (self.tp[i]+ self.fn[i] + np.finfo(np.float).eps))

        return torch.mean(torch.stack(recall), 0)


    def get_f1(self):

        f1 = []
        for i in range(self.n_classes):
            f1.append((2.0 * self.tp[i]) / (2.0 * self.tp[i] + self.fp[i] + self.fn[i]))

        return torch.mean(torch.stack(f1), 0)

    def get_der(self):

        der = []

        for i in range(self.n_classes):
            der.append( (self.fp[i] + self.fn[i]) / (self.fn[i] + self.tp[i] + np.finfo(np.float).eps))

        return torch.mean(torch.stack(der), 0)

    def get_matt(self):

        matt = []

        for i in range(self.n_classes):
            matt.append((self.tp[i] * self.tn[i] - self.fp[i] * self.fn[i]) / \
               (np.finfo(np.float).eps + (self.tp[i] + self.fp[i]) * (self.tp[i] + self.fn[i]) * (self.tn[i] + self.fp[i]) * (
                           self.tn[i] + self.fn[i])).sqrt()  )

        return torch.mean(torch.stack(matt), 0)

    def get_tot_examples(self):

        tot = []
        for i in range(self.n_classes):
            tot.append( self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i])

        return torch.sum(torch.stack(tot), 0)

    def get_positive_examples(self):
        tot = []
        for i in range(self.n_classes):
            tot.append(self.tp[i] + self.fn[i])

        return torch.sum(torch.stack(tot), 0)

    def get_positive_examples_class(self, i):
        return self.tp[i] + self.fn[i]
