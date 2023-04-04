"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast, GradScaler

from mingpt.utils import CfgNode as CN
from mingpt.utils import device_from

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
        return C

    def __init__(self, config, model, train_dataset, compile=False, use_tf32=False, **kwargs):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        self.device = device_from(config)
        self.model = self.model.to(self.device)
        print("Trainer.__init__(): running on device", self.device)

        if compile:
            print("Trainer.__init__(): compiling model with torch.compile()")
            self.model = torch.compile(self.model)
            if use_tf32 and self.device == "cuda":
                print("Trainer.__init__(): Lowering matmul precision to 'high' on gpu for higher performance. " +
                    "See: https://github.com/Lightning-AI/lightning/issues/12997")
                torch.set_float32_matmul_precision('high')

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

    def run(self):
        model, config = self.model, self.config
        scaler = GradScaler()

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        sampler = torch.utils.data.RandomSampler(
            self.train_dataset, replacement=False
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
                with autocast():
                    logits, self.loss = model(x, y)

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                scaler.scale(self.loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                scaler.step(self.optimizer)
                scaler.update()

                self.trigger_callbacks('on_batch_end')
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = self.iter_dt*0.5 + (tnow - self.iter_time)*0.5
                self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
