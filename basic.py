import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils import stateless
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List
from learned_optimization.optim import LearnedOptimizer


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


class SimpleModel(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden: int = 128, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def meta_train(
    learned_opt: LearnedOptimizer,
    train_loader: DataLoader,
    n_meta_epochs: int = 3000,
    inner_steps: int = 20,
    lr_outer: float = 1e-4,
    device: torch.device | str = "cpu",
    log_every: int = 1,
):
    dev = torch.device(device)
    outer_opt = Adam(learned_opt.rnn.parameters(), lr=lr_outer)
    criterion = nn.CrossEntropyLoss()
    data_iter = iter(train_loader)
    running_outer = 0.0

    for epoch in range(1, n_meta_epochs + 1):
        base_model = SimpleModel().to(dev)
        param_names = [n for n, _ in base_model.named_parameters()]
        params = [p.clone().detach().requires_grad_() for p in base_model.parameters()]
        state = learned_opt.init_state(params)
        losses = []
        for _ in range(inner_steps):
            try:
                imgs, lbls = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                imgs, lbls = next(data_iter)
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            logits = stateless.functional_call(
                base_model, dict(zip(param_names, params)), (imgs,)
            )
            loss_inner = criterion(logits, lbls)
            grads = torch.autograd.grad(loss_inner, params, create_graph=True)
            params, state = learned_opt(params, grads, state)
            losses.append(loss_inner)

        def score_losses(losses: List[torch.Tensor]) -> torch.Tensor:
            return sum(losses) / len(losses)
            weights = torch.exp(
                torch.arange(len(losses), dtype=torch.float32, device=dev) * -0.1
            )
            return torch.sum(torch.stack(losses) * weights) / torch.sum(weights)

        loss_outer = score_losses(losses)
        outer_opt.zero_grad()
        loss_outer.backward()
        torch.nn.utils.clip_grad_norm_(learned_opt.rnn.parameters(), max_norm=1.0)
        outer_opt.step()
        running_outer += loss_outer.item()

        if epoch % log_every == 0:
            print(
                f"[meta {epoch}/{n_meta_epochs}] outer-loss(avg): {running_outer/log_every:.4f}"
            )
            running_outer = 0.0
            with torch.no_grad():
                acc = accuracy(logits, lbls)
                print(f"  accuracy: {acc:.4f}")


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    count = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, lbls).item()
            total_acc += (logits.argmax(1) == lbls).float().sum().item()
            count += imgs.size(0)
    return total_loss / len(loader), total_acc / count


def _plot_curves(l1: List[float], l2: List[float]):
    plt.figure(figsize=(8, 4))
    smooth = 200
    if len(l1) < smooth * 2:
        smooth = max(1, len(l1) // 10)
    xs = range(len(l1) - smooth + 1)
    ma1 = torch.tensor(l1).unfold(0, smooth, 1).mean(1).cpu()
    ma2 = torch.tensor(l2).unfold(0, smooth, 1).mean(1).cpu()
    plt.plot(xs, ma1, label="Adam")
    plt.plot(xs, ma2, label="Learned opt")
    plt.title("Training loss (moving-avg)")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print("Saved plot → training_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-meta", action="store_true")
    parser.add_argument("--meta-epochs", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr-outer", type=float, default=1e-3)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_ds = datasets.MNIST(
        root=Path("./data"), train=True, download=True, transform=tfm
    )
    test_ds = datasets.MNIST(
        root=Path("./data"), train=False, download=True, transform=tfm
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True)

    learned_opt = LearnedOptimizer(hidden=64).to(dev)
    if args.train_meta:
        meta_train(
            learned_opt,
            train_loader,
            n_meta_epochs=args.meta_epochs,
            inner_steps=args.inner_steps,
            lr_outer=args.lr_outer,
            device=dev,
            log_every=1,
        )
        torch.save(learned_opt.state_dict(), "learned_opt.pth")
    else:
        try:
            learned_opt.load_state_dict(torch.load("learned_opt.pth", map_location=dev))
        except FileNotFoundError:
            print("No pre-trained optimizer found, using random weights")

    model_adam = SimpleModel().to(dev)
    model_learn = SimpleModel().to(dev)
    opt_adam = Adam(model_adam.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    losses_adam, losses_learned = [], []

    state = learned_opt.init_state(list(model_learn.parameters()))
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, lbls = imgs.to(dev), lbls.to(dev)

            opt_adam.zero_grad()
            out_a = model_adam(imgs)
            loss_a = criterion(out_a, lbls)
            loss_a.backward()
            opt_adam.step()
            losses_adam.append(loss_a.item())

            model_learn.zero_grad(set_to_none=True)
            out_l = model_learn(imgs)
            loss_l = criterion(out_l, lbls)
            loss_l.backward()

            params = list(model_learn.parameters())
            grads = [p.grad for p in params]
            updated_params, state = learned_opt(params, grads, state)

            state = tuple(s.detach() for s in state)

            with torch.no_grad():
                for p, new_p in zip(model_learn.parameters(), updated_params):
                    p.copy_(new_p)

            losses_learned.append(loss_l.item())

        print(f"Epoch {epoch} finished – elapsed {time.perf_counter() - start:.1f}s")

    _plot_curves(losses_adam, losses_learned)

    loss_a_test, acc_a_test = evaluate(model_adam, test_loader, dev)
    loss_l_test, acc_l_test = evaluate(model_learn, test_loader, dev)
    print(f"Adam test loss: {loss_a_test:.4f}, test accuracy: {acc_a_test:.4f}")
    print(
        f"Learned optimizer test loss: {loss_l_test:.4f}, test accuracy: {acc_l_test:.4f}"
    )


if __name__ == "__main__":
    main()
