import argparse
import json
import os
import sys
from datetime import datetime
from time import time

import torch
import numpy as np
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import OneOf, File, Folder, ListOfInts
from tqdm import tqdm

sys.path.append("src")
from tool.training import MeanScalarMetric
from tool.misc import set_seed
from model import get_network
from dataloader import get_loaders

Section('network').params(
    architecture=Param(OneOf(['squeezenet']), required=True),
)

Section('train').params(
    seed=Param(int, required=True),
    epoch=Param(int, required=True),
    scheduler_type=Param(OneOf(['step', 'cyclic', 'cosine']), required=True),
)

Section('train.optimizer').params(
    type=Param(OneOf(['Adam', 'SGD']), required=True),
    lr=Param(float, required=True),
    weight_decay=Param(float, required=True),
    momentum=Param(float, default=0.9),
)

Section('train.scheduler.step').enable_if(
    lambda cfg: cfg['train.scheduler_type'] == 'step'
).params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_size=Param(int, 'learning rate step size', default=30),
)

Section('train.scheduler.cyclic').enable_if(
    lambda cfg: cfg['train.scheduler_type'] == 'cyclic'
).params(
    lr_peak_epoch=Param(int, 'epoch at which lr peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    dry_run=Param(bool, 'use log or not', is_flag=True),
    path=Param(Folder(), 'resume path, if new experiment leave blank', default=None),
    save_intermediate_frequency=Param(int),
)

class Trainer:

    @param('train.seed')
    def __init__(self, seed):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        set_seed(seed)
        self.loss = torch.nn.MSELoss()
        self.create_loaders()
        self.create_network()
        self.create_optimizer_and_scheduler()
        self.initialize_metrics()
        self.resume()
        self.run()

    @param('network.architecture')
    def create_network(self, architecture):
        self.network = get_network(architecture).to(self.device)

    def create_loaders(self):
        loaders = get_loaders()
        self.train_loader = loaders['train']
        self.test_loader = loaders['test']

    @param('train.epoch')
    def get_cosine_scheduler(self, epoch):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch * len(self.train_loader))

    @param('train.scheduler.step.step_ratio')
    @param('train.scheduler.step.step_size')
    def get_step_scheduler(self, step_ratio, step_size):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size * len(self.train_loader),
                                               gamma=step_ratio)

    @param('train.epoch')
    @param('train.scheduler.cyclic.lr_peak_epoch')
    def get_cyclic_scheduler(self, epoch, lr_peak_epoch):
        return torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-4,
                                                 max_lr=self.optimizer.param_groups[0]['lr'],
                                                 step_size_up=lr_peak_epoch * len(self.train_loader),
                                                 step_size_down=(epoch - lr_peak_epoch) * len(self.train_loader))

    @param('train.optimizer.type')
    @param('train.optimizer.lr')
    @param('train.optimizer.weight_decay')
    @param('train.optimizer.momentum')
    @param('train.scheduler_type')
    def create_optimizer_and_scheduler(self, type, lr, weight_decay, momentum, scheduler_type):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if type == "Adam":
            self.optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif type == "SGD":
            self.optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise NotImplementedError(f'Optimizer {type} not implemented')
        self.scheduler = getattr(self, f'get_{scheduler_type}_scheduler')()

    def initialize_metrics(self):
        self.train_meters = {
            'loss': MeanScalarMetric(compute_on_step=False).to(self.device)
        }
        self.test_meters = {
            'loss': MeanScalarMetric(compute_on_step=False).to(self.device)
        }
        self.start_time = time()
        self.best_loss = np.inf
        self.start_epoch = 0

    @param('logging.path')
    def resume(self, path=None):
        try:
            ckpt = torch.load(os.path.join(path, "checkpoints", "newest.ckpt"), map_location=self.device)
            for key, val in ckpt["state_dicts"].items():
                eval(f"self.{key}.load_state_dict(val)")
            self.best_loss = ckpt["best_loss"]
            self.start_epoch = ckpt["current_epoch"]
            self.start_time -= ckpt["relative_time"]
        except FileNotFoundError:
            os.makedirs(os.path.join(path, "checkpoints"), exist_ok=False)
        except TypeError:
            pass

    @param('logging.path')
    def log(self, content, path):
        print(f'=> Log: {content}')
        cur_time = time()
        path = os.path.join(path, 'log.json')
        stats = {
            'timestamp': cur_time,
            'relative_time': cur_time - self.start_time,
            **content
        }
        if os.path.isfile(path):
            with open(path, 'r') as fd:
                old_data = json.load(fd)
            with open(path, 'w') as fd:
                fd.write(json.dumps(old_data + [stats]))
                fd.flush()
        else:
            with open(path, 'w') as fd:
                fd.write(json.dumps([stats]))
                fd.flush()

    @param('train.epoch')
    @param('logging.dry_run')
    @param('logging.path')
    @param('logging.save_intermediate_frequency')
    def run(self, epoch, dry_run, path=None, save_intermediate_frequency=None):
        for e in range(self.start_epoch, epoch):

            train_stats = self.train_loop(e)
            test_stats = self.test_loop()

            if not dry_run:
                ckpt = {
                    "state_dicts": {
                        "network": self.network.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    "current_epoch": e + 1,
                    "best_loss": self.best_loss,
                    "relative_time": time() - self.start_time,
                }
                if test_stats['loss'] < self.best_loss:
                    self.best_loss = test_stats['loss']
                    ckpt['best_loss'] = self.best_loss
                    torch.save(ckpt, os.path.join(path, "checkpoints", "best.ckpt"))
                torch.save(ckpt, os.path.join(path, "checkpoints", "newest.ckpt"))
                if save_intermediate_frequency is not None:
                    if (e + 1) % save_intermediate_frequency == 0:
                        torch.save(ckpt, os.path.join(path, "checkpoints", f"epoch{e}.ckpt"))

                self.log(content={
                    'epoch': e,
                    'train': train_stats,
                    'test': test_stats,
                    'best_loss': self.best_loss,
                })

    def train_loop(self, epoch):
        self.network.train()

        iterator = tqdm(self.train_loader, ncols=160)
        for images, target, in iterator:
            images, target = images.to(self.device), target.to(self.device)
            ### Training start
            self.optimizer.zero_grad()
            output = self.network(images)
            loss_train = self.loss(output, target)
            loss_train.backward()
            self.optimizer.step()
            self.scheduler.step()
            ### Training end

            self.train_meters['loss'](loss_train)
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}

            group_lrs = []
            for _, group in enumerate(self.optimizer.param_groups):
                group_lrs.append(f'{group["lr"]:.2e}')

            names = ['ep', 'lrs', 'loss']
            values = [epoch, group_lrs, f"{stats['loss']:.3f}"]

            msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
            iterator.set_description(msg)

        [meter.reset() for meter in self.train_meters.values()]
        return stats

    def test_loop(self):
        self.network.eval()

        iterator = tqdm(self.test_loader, ncols=120)
        for images, target in iterator:
            images, target = images.to(self.device), target.to(self.device)
            with torch.no_grad():
                output = self.network(images)

            loss_test = self.loss(output, target)
            self.test_meters['loss'](loss_test)
            stats = {k: m.compute().item() for k, m in self.test_meters.items()}

            names = ['loss']
            values = [f"{stats['loss']:.3f}"]

            msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
            iterator.set_description(msg)

        [meter.reset() for meter in self.test_meters.values()]
        return stats


if __name__ == "__main__":
    config = get_current_config()
    parser = argparse.ArgumentParser("Imagenet Transfer to Downstream")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    if config['logging.path'] is not None:
        assert not config['logging.dry_run'], "dry run can not be used with resume path!"
        config.collect_config_file(os.path.join(config['logging.path'], 'config.json'))
        config.validate()
    else:
        config.validate()
        file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        path = os.path.join("file", "experiments", os.path.basename(__file__.split('.')[0]), file_name)
        if not config['logging.dry_run']:
            os.makedirs(path, exist_ok=False)
            config.dump_json(os.path.join(path, 'config.json'),
                             [('logging', 'path'), ('logging', 'dry_run')])
            config.collect({'logging': {'path': path}})
    config.summary()
    Trainer()