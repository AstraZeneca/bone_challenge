import torch
import os
import numpy as np
from collections import defaultdict
from torch import nn
from torchvision.ops import sigmoid_focal_loss
from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter


def calculate_score(pred_slice_num, gt_slice_num):
    """Returns the survival function a single-sided normal distribution with stddev=3."""
    diff = abs(pred_slice_num - gt_slice_num)
    return 2 * norm.sf(diff, 0, 3)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state, is_best, dirname = './', filename='checkpoint.pth.tar'):
    fname = os.path.join(dirname, filename)
    fnamebest = os.path.join(dirname, 'model_best.pth.tar')
    torch.save(state, fname)
    if is_best:
        torch.save(state, fnamebest)


def get_zindex(start_seq, seq_len, reg_out, to_int=True):
    if to_int:
        pred_val = start_seq + int(np.round(reg_out*seq_len))
    else:
        pred_val = start_seq + reg_out*seq_len
    return pred_val


def get_predictions(model, loader, seqlen, device, to_int=True):
    model.eval()
    full_output = defaultdict (dict)
    with torch.no_grad():
        full_output = dict()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                inputs = batch['input_seq']
                inputs = inputs.to(device)
                clf, reg = model(inputs)
                clf = clf.cpu().detach().numpy()
                reg = reg.cpu().detach().numpy()
                for img_name, zorg, start, clf1, reg1 in zip(batch['image_name'], batch['zindx'], batch['start_indx'], clf, reg):
                    if np.isnan(clf1):
                        "This can happen if no signal in the image, and Z normalisation returns nans"
                        clf1 = -100
                        reg1 = 0
                    if img_name not in full_output:
                        full_output[img_name] = {}
                        #set default values
                        full_output[img_name]['zindx'] = zorg.detach().numpy()
                        full_output[img_name]['clf_logit'] = clf1
                        full_output[img_name]['reg_output'] = reg1
                        full_output[img_name]['start_indx'] = start.detach().numpy()
                    elif full_output[img_name]['clf_logit']<clf1:
                        full_output[img_name]['clf_logit'] = clf1
                        full_output[img_name]['reg_output'] = reg1
                        full_output[img_name]['start_indx'] = start.detach().numpy()
            #calculate zindices
            for imgname in full_output.keys():
                if not imgname:
                    continue
                zindx = get_zindex(full_output[imgname]['start_indx'], seqlen, full_output[imgname]['reg_output'], to_int=to_int) 
                full_output[imgname]['pred_zindx'] = zindx
                print(imgname, full_output[imgname]['zindx'], zindx)
    return full_output
 

class Trainer_sequence:
    def __init__(self, config, out_folder, model, train_loader, valid_loader, valid_train_loader, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.max_epoch = config['max_epoch']
        self.early_stop = config['early_stop']
        self.regloss = nn.MSELoss(reduction='none').to(self.device)
        self.clfloss = sigmoid_focal_loss
        self.reg_fac = config['reg_factor']
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_train_loader = valid_train_loader

        self.print_freq = config['print_frequency']
        self.eval_freq = config['eval_frequency']

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001, weight_decay=0.001)
        self.warmup_epoch = config['warmup_epochs']
        scheduler1 = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=self.warmup_epoch)
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[self.warmup_epoch])
        self.out_folder = out_folder
        self.writer = SummaryWriter(f'{self.out_folder}/tensorboard')

    def train_epoch (self, epoch):
        self.model.train()
        loss_clf = AverageMeter('LossCLF', ':.4e')
        loss_reg = AverageMeter('LossREG', ':.4e')
        losses = AverageMeter('Loss', ':.4e')        
        len_dl = len(self.train_loader)
        self.model.zero_grad()  
        for i, batch in enumerate(self.train_loader):
            inputs = batch['input_seq']
            labels_clf = batch['clf']
            labels_reg = batch['reg']
            inputs = inputs.to(self.device)
            labels_clf = labels_clf.type(torch.FloatTensor)
            labels_reg = labels_reg.type(torch.FloatTensor)
            labels_clf = labels_clf.to(self.device)
            labels_reg = labels_reg.to(self.device)
            clf, reg = self.model(inputs)
            clf_loss = self.clfloss(clf, labels_clf)
            reg_loss = self.regloss(reg, labels_reg)*labels_clf*self.reg_fac

            loss = (clf_loss + self.reg_fac*reg_loss).nanmean()
            loss_reg.update(reg_loss.nanmean().item(), 1)
            loss_clf.update(clf_loss.nanmean().item(), 1)
            losses.update(loss.item(), 1)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()     
            self.writer.add_scalar("Loss Classifier", clf_loss.mean(), epoch*len_dl+i)
            self.writer.add_scalar("Loss Regression", reg_loss.mean(), epoch*len_dl+i)

        self.lr_scheduler.step()
        self.writer.add_scalar("Learning Rate", self.lr_scheduler.get_last_lr()[0], epoch)

        print(f'Loss mean train {loss_clf.avg}; {loss_reg.avg}')

    def get_metrics(self, epoch, datatype):
        if datatype == 'valid':
            dataloader = self.valid_loader
        elif datatype == 'train':
            dataloader = self.valid_train_loader
        else:
            raise NotImplementedError
        seqlen = dataloader.dataset.seq_len
        metrics = AverageMeter('Metrics', ':.4e')
        full_output = get_predictions(self.model, dataloader, seqlen, self.device, to_int=True)
        #calculate scores
        for imgname, values in full_output.items():
            score = calculate_score(values['pred_zindx'], values['zindx'])
            metrics.update(score, 1)
        self.writer.add_scalar(f"Accuracy/{datatype}", metrics.avg, epoch)
        return metrics.avg


    def train(self):
        best_acc = 0
        counter = 0
        for epoch in range(self.max_epoch):
            self.train_epoch(epoch)
            if (epoch >= self.warmup_epoch) & (epoch%self.eval_freq==0):
                train_acc = self.get_metrics(epoch, 'train')
                print(f'### EPOCH {epoch} ####')
                print('###### TRAIN ACCURACY####')
                print(train_acc)
                valid_acc = self.get_metrics(epoch, 'valid')
                print('###### VALID ACCURACY####')
                print(valid_acc)
                is_best = False
            
                print(valid_acc, best_acc)
                if valid_acc >= best_acc:
                    counter = 0
                    is_best = True
                    best_acc = valid_acc
                else:
                    counter += 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : self.optimizer.state_dict()}, is_best, dirname=self.out_folder )
                
            if counter >= self.early_stop:
                return