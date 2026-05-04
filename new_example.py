import copy
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# =========================================================
# CONFIG
# =========================================================

PROFILE = "full"   # "fast" or "full"

if PROFILE == "fast":
    SEEDS = [42, 123, 999]
    EPOCHS = 10
else:
    SEEDS = [42, 123, 999, 2024, 2025]
    EPOCHS = 20

BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4
HIDDEN = 256
DEPTH = 8
VAL_SIZE = 5000
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# ACTIVATIONS
# =========================================================

class AdaptiveEulerGate(nn.Module):
    """
    Improved Euler-style activation:
    out = alpha * (cos(theta)*x + sin(theta)*tanh(x)) * sigmoid(gamma*x + beta)
    """
    def __init__(self, num_features):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.ones(num_features))
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        euler_part = torch.cos(self.theta) * x + torch.sin(self.theta) * torch.tanh(x)
        gate = torch.sigmoid(self.gamma * x + self.beta)
        return self.alpha * euler_part * gate

ACTIVATIONS = {
    "relu":  lambda: nn.ReLU(),
    "gelu":  lambda: nn.GELU(),
    "swish": lambda: nn.SiLU(),   # built-in Swish/SiLU
    "mish":  lambda: nn.Mish(),
    "euler_v2": lambda: AdaptiveEulerGate(HIDDEN),
}

# =========================================================
# MODEL
# =========================================================

class DeepMLP(nn.Module):
    def __init__(self, activation_factory):
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(28 * 28, HIDDEN),
            nn.LayerNorm(HIDDEN),
            activation_factory(),
        ]

        for _ in range(DEPTH - 1):
            layers += [
                nn.Linear(HIDDEN, HIDDEN),
                nn.LayerNorm(HIDDEN),
                activation_factory(),
            ]

        layers += [nn.Linear(HIDDEN, NUM_CLASSES)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================================================
# DATA
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# fixed split for fairness
split_gen = torch.Generator().manual_seed(777)
train_size = len(full_train_dataset) - VAL_SIZE
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, VAL_SIZE],
    generator=split_gen
)

def make_loaders(seed):
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader

# =========================================================
# TRAIN / EVAL
# =========================================================

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)

            total_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

def train_one_run(activation_name, seed):
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(seed)

    model = DeepMLP(ACTIVATIONS[activation_name]).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 2),
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 2),
    }

# =========================================================
# MAIN LOOP
# =========================================================

results = {
    name: {
        "best_epochs": [],
        "best_val_accs": [],
        "test_losses": [],
        "test_accs": [],
        "runs": []
    }
    for name in ACTIVATIONS
}

total_runs = len(ACTIVATIONS) * len(SEEDS)
run_idx = 0

for act_name in ACTIVATIONS:
    for seed in SEEDS:
        run_idx += 1
        print(f"\n[{run_idx}/{total_runs}] {act_name.upper()} | seed={seed}")
        t0 = time.time()

        out = train_one_run(act_name, seed)

        dt = time.time() - t0
        print(
            f"  best_epoch={out['best_epoch']} | "
            f"best_val={out['best_val_acc']:.2f}% | "
            f"test={out['test_acc']:.2f}% | "
            f"time={dt:.1f}s"
        )

        results[act_name]["best_epochs"].append(out["best_epoch"])
        results[act_name]["best_val_accs"].append(out["best_val_acc"])
        results[act_name]["test_losses"].append(out["test_loss"])
        results[act_name]["test_accs"].append(out["test_acc"])
        results[act_name]["runs"].append(out)

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 85)
print("SUMMARY — Mean ± Std across seeds")
print("=" * 85)
print(f"{'Activation':<12} {'BestEpoch':>10} {'Val Acc':>16} {'Test Loss':>16} {'Test Acc':>16}")
print("-" * 85)

summary = {}

for act_name, data in results.items():
    best_epochs = np.array(data["best_epochs"], dtype=np.float32)
    val_accs = np.array(data["best_val_accs"], dtype=np.float32)
    test_losses = np.array(data["test_losses"], dtype=np.float32)
    test_accs = np.array(data["test_accs"], dtype=np.float32)

    summary[act_name] = {
        "best_epoch_mean": round(best_epochs.mean(), 2),
        "best_epoch_std": round(best_epochs.std(), 2),
        "val_acc_mean": round(val_accs.mean(), 2),
        "val_acc_std": round(val_accs.std(), 2),
        "test_loss_mean": round(test_losses.mean(), 4),
        "test_loss_std": round(test_losses.std(), 4),
        "test_acc_mean": round(test_accs.mean(), 2),
        "test_acc_std": round(test_accs.std(), 2),
    }

    s = summary[act_name]
    print(
        f"{act_name.upper():<12} "
        f"{s['best_epoch_mean']:.2f}±{s['best_epoch_std']:.2f}   "
        f"{s['val_acc_mean']:.2f}%±{s['val_acc_std']:.2f}%   "
        f"{s['test_loss_mean']:.4f}±{s['test_loss_std']:.4f}   "
        f"{s['test_acc_mean']:.2f}%±{s['test_acc_std']:.2f}%"
    )

print("=" * 85)

# ranking
ranking = sorted(
    [(name, summary[name]["test_acc_mean"]) for name in summary],
    key=lambda x: x[1],
    reverse=True
)

print("\nRANKING by mean Test Accuracy:")
for i, (name, score) in enumerate(ranking, 1):
    print(f"{i}. {name.upper():<12} {score:.2f}%")

# compare against GELU
if "gelu" in summary:
    gelu_acc = summary["gelu"]["test_acc_mean"]
    print("\nDelta vs GELU:")
    for name in summary:
        diff = summary[name]["test_acc_mean"] - gelu_acc
        print(f"  {name.upper():<12}: {diff:+.2f} pp")

with open("benchmark_results_reliable.json", "w") as f:
    json.dump({
        "config": {
            "profile": PROFILE,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden": HIDDEN,
            "depth": DEPTH,
            "val_size": VAL_SIZE
        },
        "summary": summary,
        "raw": results
    }, f, indent=2)

print("\nSaved: benchmark_results_reliable.json")