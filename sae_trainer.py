from utils import *
from crosscoder import CrossCoder
from buffer import Buffer
import threading
import queue
import time

class SAETrainer:
    def __init__(self, cfg, buffer: Buffer):
        self.cfg = cfg
        self.buffer = buffer
        self.device = torch.device(cfg["device"])
        self.cross_coder = CrossCoder(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.cross_coder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.step = 0
        self.activation_queue = queue.Queue(maxsize=2)
        self.running = True

    def train_step(self, acts):
        """Single training step"""
        acts = acts.to(self.device)
        self.optimizer.zero_grad()
        loss_dict = self.cross_coder(acts)
        loss = loss_dict["loss"]
        loss.backward()
        self.optimizer.step()
        self.step += 1

        if self.step % self.cfg["log_every"] == 0:
            wandb.log({**loss_dict, "step": self.step})

        if self.step % self.cfg["save_every"] == 0:
            self.cross_coder.save()

    def training_loop(self):
        """Main training loop that runs on SAE GPU"""
        print(f"Starting SAE training on {self.device}")
        while self.running:
            try:
                acts = self.buffer.next()
                self.train_step(acts)

                if self.step >= self.cfg["num_tokens"] // self.cfg["batch_size"]:
                    print("Finished training!")
                    self.running = False
                    break

            except Exception as e:
                print(f"Error in SAE training: {e}")
                self.running = False
                raise e

    def start(self):
        """Start SAE training in separate thread"""
        self.thread = threading.Thread(target=self.training_loop)
        self.thread.start()

    def stop(self):
        """Stop SAE training"""
        self.running = False
        self.thread.join()
