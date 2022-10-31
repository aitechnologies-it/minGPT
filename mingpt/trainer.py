"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import CfgNode as CN

from opacus import PrivacyEngine

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.temperature = 2.0
        # teacher distillation parameters
        C.distil_scheduler = "linear" # "no": disabled; "linear": schedule it from provided value to 1-x
        C.alpha_distil = 0.99 # weight assigned to teacher loss; use 1-x for regular ce loss
        C.dp_noise_multiplier = 1.1 # differential privacy noise mult.
        C.dp_max_grad_norm = 1.0 # differential privacy norm
        return C

    def __init__(self, config, model, train_dataset, **kwargs):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.teacher = kwargs.pop("teacher_model", None)
        self.iter_alpha_distil = config.alpha_distil

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        if self.teacher:
            self.teacher = self.teacher.to(self.device)
    
        self.model = self.model.to(self.device)
        print("running on device", self.device)
        # self.tea_loss = torch.tensor(0).to(self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @staticmethod
    def _compute_linear_schedule(iter_num, max_iters, start_value, end_value):
        return iter_num / max_iters * (end_value - start_value) + start_value

    def run(self, privacy=False):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        sampler = None
        if not privacy:
            sampler = torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
            )

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        if privacy:
            privacy_engine = PrivacyEngine()
            self.dp_delta = 1 / len(train_loader) # Parameter for privacy accounting. Probability of not achieving privacy guarantees
            model, self.optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                noise_multiplier=config.dp_noise_multiplier,
                max_grad_norm=config.dp_max_grad_norm,
                poisson_sampling=False,
            )
            self.dp_eps = privacy_engine.get_epsilon(self.dp_delta)
            print(f"Differential Privacy active. Using sigma={self.optimizer.noise_multiplier} and C={config.dp_max_grad_norm}")

        model.train()
        self.iter_num = 0
        while True:
            self.iter_time = time.time()
            for batch in train_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                # # fetch the next batch (x, y) and re-init iterator if needed
                # try:
                #     batch = next(data_iter)
                # except StopIteration:
                #     data_iter = iter(train_loader)
                #     batch = next(data_iter)
                # batch = [t.to(self.device) for t in batch]
                # x, y = batch

                # forward the model
                logits, self.loss = model(x, y)

                # backprop and update the parameters
                # model.zero_grad(set_to_none=True)
                model.zero_grad()
                self.optimizer.zero_grad()
                self.loss.backward()

                if not privacy:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                
                self.optimizer.step()

                if privacy:
                    self.dp_eps = privacy_engine.get_epsilon(self.dp_delta)

                self.trigger_callbacks('on_batch_end')
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = self.iter_dt*0.1 + (tnow - self.iter_time)*0.9
                self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
