import time
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
        use_amp: bool = True,
        log_interval: int = 100,
        ckpt_interval: int = 5000,
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.step = 0

    def train_step(self, batch: dict) -> float:
        self.model.train()

        f0 = batch["f0"].to(self.device)
        loudness = batch["loudness"].to(self.device)
        target = batch["audio"].to(self.device)

        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                pred = self.model(f0, loudness)
                loss = self.loss_fn(pred, target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred = self.model(f0, loudness)
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, max_batches: int = 20) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in val_loader:
            if n >= max_batches:
                break
            f0 = batch["f0"].to(self.device)
            loudness = batch["loudness"].to(self.device)
            target = batch["audio"].to(self.device)
            pred = self.model(f0, loudness)
            loss = self.loss_fn(pred, target)
            total_loss += loss.item()
            n += 1
        return total_loss / n if n > 0 else 0.0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        total_steps: int,
    ):
        # Create infinite training iterator
        def cycle(dl):
            while True:
                for batch in dl:
                    yield batch

        train_iter = cycle(train_loader)
        t0 = time.time()

        for step in range(1, total_steps + 1):
            self.step = step
            batch = next(train_iter)
            loss = self.train_step(batch)
            self.train_losses.append(loss)

            if step % self.log_interval == 0:
                val_loss = self.validate(val_loader)
                self.val_losses.append((step, val_loss))
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                print(
                    f"[{step:6d}/{total_steps}] "
                    f"train_loss={loss:.4f}  val_loss={val_loss:.4f}  "
                    f"rate={steps_per_sec:.1f} steps/s",
                    flush=True,
                )

            if step % self.ckpt_interval == 0:
                ckpt_path = self.save_dir / f"step_{step:06d}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_losses": self.train_losses,
                        "val_losses": self.val_losses,
                    },
                    ckpt_path,
                )
                print(f"  checkpoint → {ckpt_path}")

        # Final checkpoint
        final_path = self.save_dir / "phase1_final.pt"
        torch.save(
            {
                "step": total_steps,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            final_path,
        )
        print(f"Done. Final checkpoint → {final_path}")
