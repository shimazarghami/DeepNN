import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- 1) Data Augmentation ----------
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

# ---------- 2) Tiny CNN (خیلی سریع) ----------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 32→16

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 16→8

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 8→4

            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

model = TinyCNN().to(device)

# ---------- 3) Optimizer + Weight Decay ----------
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ---------- 4) Learning Rate Scheduler ----------
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# ---------- 5) Loss ----------
loss_fn = nn.CrossEntropyLoss()

# ---------- 6) AMP (Automatic Mixed Precision) ----------
scaler = GradScaler()

# ---------- 7) Early Stopping ----------
best_loss = float("inf")
patience = 10
counter = 0

# ---------- 8) Training Loop ----------
def train_model(epochs=40):
    global best_loss, counter

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            with autocast():
                pred = model(xb)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()

            # ---------- 9) Gradient Clipping ----------
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | train_loss = {avg_loss:.4f}")

        # ---------- Early Stopping ----------
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), "best_tinycnn.pt")
        else:
            counter += 1
            if counter >= patience:
                print("EARLY STOPPING TRIGGERED!")
                break

train_model(epochs=40)

# ---------- 10) Evaluate ----------
model.load_state_dict(torch.load("best_tinycnn.pt"))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        _, predicted = torch.max(pred, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

acc = 100 * correct / total
print(f"\nFinal CIFAR-10 Accuracy = {acc:.2f}%")

#Epoch 1 | train_loss = 1.7834
#Epoch 2 | train_loss = 1.4694
#Epoch 3 | train_loss = 1.3291
#Epoch 4 | train_loss = 1.2085
#Epoch 5 | train_loss = 1.1289
#Epoch 6 | train_loss = 1.0708
#Epoch 7 | train_loss = 1.0021
#Epoch 8 | train_loss = 0.9709
#Epoch 9 | train_loss = 0.9205
#Epoch 10 | train_loss = 0.8868
#Epoch 11 | train_loss = 0.8541
#Epoch 12 | train_loss = 0.8244
#Epoch 13 | train_loss = 0.8033
#Epoch 14 | train_loss = 0.7762
#Epoch 15 | train_loss = 0.7560
#Epoch 16 | train_loss = 0.7443
#Epoch 17 | train_loss = 0.7348
#Epoch 18 | train_loss = 0.7195
#Epoch 19 | train_loss = 0.7195
#Epoch 20 | train_loss = 0.7162
#Epoch 21 | train_loss = 0.7171
#Epoch 22 | train_loss = 0.7135
#Epoch 23 | train_loss = 0.7123
#Epoch 24 | train_loss = 0.7130
#Epoch 25 | train_loss = 0.7167
#Epoch 26 | train_loss = 0.7223
#Epoch 27 | train_loss = 0.7240
#Epoch 28 | train_loss = 0.7279
#Epoch 29 | train_loss = 0.7273
#Epoch 30 | train_loss = 0.7321
#Epoch 31 | train_loss = 0.7302
#Epoch 32 | train_loss = 0.7346
#Epoch 33 | train_loss = 0.7312
#EARLY STOPPING TRIGGERED!

#Final CIFAR-10 Accuracy = 76.75%
