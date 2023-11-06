import glob
import json
import torch
import shutil

import torch.nn as nn
import torch.utils.data

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc import Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj,save_checkpoint_debug
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from torch.utils.tensorboard import SummaryWriter



class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)
        

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        # import pdb;pdb.set_trace()
        print ('Valid path',args.valid_path)
        train_dataset = Dataset(path=args.train_path, task=args.task, more_path=args.train_extract_path)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

       
        
        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)


        # define tb
        # default `log_dir` is "runs" - we'll be more specific here
        vis_filename = '{}/tensorboard/exp_1'.format(self.args.model_dir)
        self.writer = SummaryWriter(vis_filename)   

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.current_step=0
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        print ('Now start eval..')
        metric_dict = self.eval_epoch(epoch)
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1'])
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}
        
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        # import pdb;pdb.set_trace()
        for i, batch_dict in enumerate(self.valid_loader):
            
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            
            logits, labels = outputs.logits, outputs.labels
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            
            self.current_step += 1
            self.writer.add_scalar('eval_loss', losses.avg, self.current_step)
            self.writer.add_scalars(f'eval_', {
                'acc1': top1.avg,
                'acc3': top3.avg,
            }, self.current_step)
            
            

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))
        
        # n_steps_offset = epoch*len(self.train_loader)
        
        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            
            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)

            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            
            # import pdb;pdb.set_trace()
            logits, labels = outputs.logits, outputs.labels
            assert logits.size(0) == batch_size
            # head + relation -> tail ?
            loss = self.criterion(logits, labels)
            # tail -> head + relation
            loss += self.criterion(logits[:, :batch_size].t(), labels) # only the first 300 dim
            
            # import pdb;pdb.set_trace()
            # self.writer.add_scalar('training loss', loss, i)
            
            
            # combine vgae loss
            self.current_step+=1
            if self.args.model=='vgae':
               
                if torch.cuda.device_count()>1:
                    vgae_loss = sum(outputs.vgae_loss)
                else:
                    vgae_loss = outputs.vgae_loss
                vlambda = 0.1
                loss = loss + vlambda*vgae_loss
                # self.writer.add_scalar('vgae loss', vgae_loss, i)
                self.writer.add_scalars(f'Training Loss', {
                    'global': loss,
                    'vgae': vgae_loss,
                }, self.current_step)
                # import pdb;pdb.set_trace()
                
                if i % self.args.print_freq == 0:
                    logger.info('KG loss: {:.2f}\t VGAE loss: {:.2f}'.format(loss.cpu().item(),vgae_loss.cpu().item()))

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t, 1)
            losses.update(loss.item(), batch_size)
            
            # self.writer.add_scalar('loss', losses.avg, self.current_step)
            # self.writer.add_scalar('global_loss', losses.avg,global_step=self.current_step)
            
            
            
            self.writer.add_scalars(f'Training ACC', {
                'acc1': top1.avg,
                'acc3': top3.avg,
            }, self.current_step)
            

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
            
            # disable wiki
            if (not self.args.task.startswith("wiki")) and (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)
            
            # for debugging
            if (i + 1) % self.args.save_every_n_step == 0:
                filename = '{}/checkpoint_at_step_{}.mdl'.format(self.args.model_dir, i+1)
                save_checkpoint_debug({
                    'epoch': epoch,
                    'args': self.args.__dict__,
                    'state_dict': self.model.state_dict(),
                }, filename=filename)
                # exit(1)
                print ('model saved at step {},{}'.format(i+1,filename))
                logger.info('model saved at step {},{}'.format(i+1,filename))
                
            # # for debugging
            # if self.args.if_debug and i % 10 == 0:
            #     filename = '{}/checkpoint_debug_{}.mdl'.format(self.args.model_dir, i+1)
            #     save_checkpoint_debug({
            #         'epoch': epoch,
            #         'args': self.args.__dict__,
            #         'state_dict': self.model.state_dict(),
            #     }, filename=filename)
            #     exit(1)
            #     print ('model saved in debug mode')
                
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
            
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
