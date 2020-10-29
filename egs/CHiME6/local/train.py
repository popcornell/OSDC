import os
import yaml
import argparse
from torch import nn
import torch
from collections import OrderedDict
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from SSAD.utils import BinaryMeter, MultiMeter
from online_data import OnlineFeats, OnlineChunkedFeats
import numpy as np

parser = argparse.ArgumentParser(description="Training an overlapping speech detection algorithm on CHiME-6")
parser.add_argument("conf_file", type=str)
parser.add_argument("log_dir", type=str)
parser.add_argument("gpus", type=str)


class PlainModel(nn.Module):

    def __init__(self, masker):

        super(PlainModel, self).__init__()
        self.model = masker

    def forward(self, tf_rep):

        mask = self.model(tf_rep)

        mask = F.softmax(mask, 1)

        return mask


class SSAD(pl.LightningModule):

    '''
    Plain cycle routine we have 2 discriminators and two generators
    '''

    def __init__(self, hparams):
        super(SSAD, self).__init__()
        self.configs = hparams # avoid pytorch-lightning hparams logging

        if not self.configs["augmentation"]["probs"]:
            cross = nn.CrossEntropyLoss(torch.Tensor([2.4, 1.0, 2.87, 11.46, 30]).cuda(), reduction="none") #torch.Tensor([2.4, 1.0, 2.87, 11.46, 30]).cuda()
        else:
            cross = nn.CrossEntropyLoss(torch.Tensor([1.0, 1.48, 2.79, 5.58, 10]).cuda(), reduction="none")


        self.loss = lambda x, y : cross(x, y) #+ 0.1*dice(1-x, 1-y) # flip positive for focal loss
        self.train_count_metrics = MultiMeter()
        self.train_vad_metrics = BinaryMeter()
        self.train_osd_metrics = BinaryMeter()
        self.val_count_metrics = MultiMeter()
        self.val_vad_metrics = BinaryMeter()
        self.val_osd_metrics = BinaryMeter()

        from SSAD.models.tcn import TCN
        self.model = PlainModel(TCN(80, 5, 1, 5, 3, 64, 128))

        #from radam import RAdam
        from asranger import Ranger
        self.opt = Ranger(self.model.parameters(), self.configs["opt"]["lr"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt)

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):

        feats, label, mask = batch
        preds = self.model(feats)
        loss = self.loss(preds, label)
        loss = loss*mask.detach()
        loss = loss.mean()
        self.train_count_metrics.update(torch.argmax(preds, 1), label)
        self.train_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
        #self.train_osd_metrics.update(torch.argmax(torch.cat((preds[:, :2], torch.sum(preds[:, 2:], 1, keepdim=True)),1), 1), torch.clamp(label, 0, 2))
        self.train_osd_metrics.update(torch.sum(preds[:, 2:], 1), label >= 2)

        tqdm_dict = {'loss': loss}

        tensorboard_logs = {'train_batch_loss': loss,
                            'train_tp_count': self.train_count_metrics.get_tp(),
                            'train_tn_count': self.train_count_metrics.get_tn(),
                            'train_fp_count': self.train_count_metrics.get_fp(),
                            'train_fn_count': self.train_count_metrics.get_fn(),
                            'train_prec_count': self.train_count_metrics.get_precision(),
                            'train_rec_count': self.train_count_metrics.get_recall(),
                            'train_prec_vad': self.train_vad_metrics.get_precision(),
                            'train_rec_vad': self.train_vad_metrics.get_recall(),
                            'train_fa_vad': self.train_vad_metrics.get_fa(),
                            'train_miss_vad': self.train_vad_metrics.get_miss(),
                            'train_der_vad': self.train_vad_metrics.get_der(),
                            'train_prec_osd': self.train_osd_metrics.get_precision(),
                            'train_rec_osd': self.train_osd_metrics.get_recall(),
                            'train_fa_osd': self.train_osd_metrics.get_fa(),
                            'train_miss_osd': self.train_osd_metrics.get_miss(),
                            'train_der_osd': self.train_osd_metrics.get_der(),
                            'train_tot_silence': self.train_count_metrics.get_positive_examples_class(0),
                            'train_tot_1spk': self.train_count_metrics.get_positive_examples_class(1),
                            'train_tot_2spk': self.train_count_metrics.get_positive_examples_class(2),
                            'train_tot_3spk': self.train_count_metrics.get_positive_examples_class(3),
                            'train_tot_4spk': self.train_count_metrics.get_positive_examples_class(4)
                            }

        output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tensorboard_logs
            })
        return output

    def validation_step(self, batch, batch_indx):

        feats, label, _ = batch
        preds = self.model(feats)
        loss = self.loss(preds, label)
        self.val_count_metrics.update(torch.argmax(preds, 1), label)
        self.val_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
        #self.val_osd_metrics.update(torch.argmax(torch.cat((preds[:, :2], torch.sum(preds[:, 2:], 1, keepdim=True)),1),1), torch.clamp(label, 0, 2))
        self.val_osd_metrics.update(torch.sum(preds[:, 2:], 1), label >= 2)
        tqdm_dict = {'val_loss': loss}

        output = OrderedDict({
            'val_loss': loss,
            'progress_bar': tqdm_dict,
        })

        return output

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss}
        tensorboard_logs = {'val_loss': avg_loss,
                            'val_tp_count': self.val_count_metrics.get_tp(),
                            'val_tn_count': self.val_count_metrics.get_tn(),
                            'val_fp_count': self.val_count_metrics.get_fp(),
                            'val_fn_count': self.val_count_metrics.get_fn(),
                            'val_prec_count': self.val_count_metrics.get_precision(),
                            'val_rec_count': self.val_count_metrics.get_recall(),
                            'val_prec_vad': self.val_vad_metrics.get_precision(),
                            'val_rec_vad': self.val_vad_metrics.get_recall(),
                            'val_fa_vad': self.val_vad_metrics.get_fa(),
                            'val_miss_vad': self.val_vad_metrics.get_miss(),
                            'val_der_vad': self.val_vad_metrics.get_der(),
                            'val_prec_osd': self.val_osd_metrics.get_precision(),
                            'val_rec_osd': self.val_osd_metrics.get_recall(),
                            'val_fa_osd': self.val_osd_metrics.get_fa(),
                            'val_miss_osd': self.val_osd_metrics.get_miss(),
                            'val_der_osd': self.val_osd_metrics.get_der(),
                            }

        self.train_count_metrics.reset()
        self.train_vad_metrics.reset()
        self.train_osd_metrics.reset()
        self.val_count_metrics.reset()
        self.val_vad_metrics.reset()
        self.val_osd_metrics.reset()
        output = OrderedDict({
            'val_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': tensorboard_logs
        })

        return output

    def configure_optimizers(self):

        return [self.opt], [self.scheduler]

    @pl.data_loader
    def train_dataloader(self):
        dataset = OnlineFeats(self.configs["data"]["chime6_root"], "train", self.configs["data"]["label_train"],
                              self.configs, probs=self.configs["augmentation"]["probs"], segment=self.configs["data"]["segment"])
        dataloader = DataLoader(dataset, batch_size=self.configs["training"]["batch_size"],
                                shuffle=True, num_workers=self.configs["training"]["num_workers"], drop_last=True)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):

        dataset = OnlineFeats(self.configs["data"]["chime6_root"], "train", self.configs["data"]["label_val"],
                                    self.configs, segment=self.configs["data"]["segment"])
        dataloader = DataLoader(dataset, batch_size=self.configs["training"]["batch_size"],
                                shuffle=True, num_workers=self.configs["training"]["num_workers"], drop_last=True)

        return dataloader

if __name__ == "__main__":

    args = parser.parse_args()
    with open(args.conf_file, "r") as f:
        confs = yaml.load(f)

    # test if compatible with lightning
    confs.update(args.__dict__)
    a = SSAD(confs)

    checkpoint_dir = os.path.join(confs["log_dir"], 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min',  verbose=True, save_top_k=5)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )

    with open(os.path.join(confs["log_dir"], "confs.yml"), "w") as f:
        yaml.dump(confs, f)

    logger = TensorBoardLogger(os.path.dirname(confs["log_dir"]), confs["log_dir"].split("/")[-1])

    trainer = pl.Trainer(max_nb_epochs=confs["training"]["n_epochs"], gpus=confs["gpus"], checkpoint_callback=checkpoint,
                         accumulate_grad_batches=confs["training"]["accumulate_batches"], early_stop_callback=early_stop_callback,
                         logger = logger,
                         gradient_clip=bool(confs["training"]["gradient_clip"]),
                         gradient_clip_val=confs["training"]["gradient_clip"]
                         )
    trainer.fit(a)
