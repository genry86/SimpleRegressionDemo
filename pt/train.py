import os

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import v2

from dataset import DetectCenterDataset
from model import DetectCenterCNNModel

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATASET_DIR = "../dataset"
RESULT_DIR = "../training_model"

train_transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = DetectCenterDataset(path=DATASET_DIR, transform=train_transform)
train_dataset, val_dataset = random_split(train_dataset, [0.7, 0.3])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = DetectCenterCNNModel(in_channels=1, out=2)
model = model.to(DEVICE)

model_path = os.path.join(RESULT_DIR, "last.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

checkpoint = None
checkpoint_path = "checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

loss_fun = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.01)

train_loss = []
train_acc = []
val_loss = []
val_acc = []
lr_list = []
best_loss = None
threshold = 0.01
start_epoch = 0
stop_counter = 0
stop_counter_limit = 5

if checkpoint is not None:
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    train_acc = checkpoint["train_acc"]
    val_acc = checkpoint["val_acc"]
    lr_list = checkpoint["lr_list"]
    best_loss = checkpoint["best_loss"]

    start_epoch = checkpoint["training_epoch"]
    EPOCHS = checkpoint["epochs"]
    stop_counter = checkpoint["stop_counter"]

    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

print(f"\nStart Training")
for epoch in range(start_epoch, EPOCHS):

    running_train_loss = []
    mean_train_loss = 0
    true_answers = 0

    model.train()
    train_loop = tqdm(train_loader, desc=f"", leave=False)
    for samples, targets in train_loop:
        samples = samples.to(DEVICE)
        targets = targets.to(DEVICE)

        # samples = samples.reshape(-1, 64*64).to(DEVICE)
        preds = model(samples)
        loss = loss_fun(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Mean Loss
        running_train_loss.append(loss.item())
        mean_train_loss = sum(running_train_loss) / len(running_train_loss)
        # Accuracy
        true_answers += (torch.round(preds) == targets).all(dim=1).sum().item()

        train_loop.set_description(f"Training Epoch - {epoch + 1}/{EPOCHS}, mean_loss={mean_train_loss:.4f}")

    running_train_acc = true_answers / len(train_dataset)

    train_loss.append(mean_train_loss)
    train_acc.append(running_train_acc)

    running_val_loss = []
    mean_val_loss = 0
    true_answers = 0

    model.eval()
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"", leave=False)
        for samples, targets in val_loop:
            samples = samples.to(DEVICE)
            targets = targets.to(DEVICE)

            # samples = samples.reshape(-1, 64 * 64).to(DEVICE)
            preds = model(samples)
            loss = loss_fun(preds, targets)

            # Mean Loss
            running_val_loss.append(loss.item())
            mean_val_loss = sum(running_val_loss) / len(running_val_loss)
            # Accuracy
            true_answers += (torch.round(preds) == targets).all(dim=1).sum().item()

            val_loop.set_description(f"Validating Epoch - {epoch + 1}/{EPOCHS}, mean_loss={mean_val_loss:.4f}")

    running_val_acc = true_answers / len(val_dataset)

    val_loss.append(mean_val_loss)
    val_acc.append(running_val_acc)

    # LR Scheduler
    lr_scheduler.step(mean_val_loss)
    lr = lr_scheduler._last_lr[0]
    lr_list.append(lr)

    print(f"🟢 Epoch {epoch + 1}: train: loss = {mean_train_loss:.4f}, acc = {running_train_acc:.4f}, val: loss {mean_val_loss:.4f}, acc = {running_val_acc:.4f}, next_lr = {lr:.6f}")

    if best_loss is None or mean_val_loss < best_loss - best_loss*threshold:
        best_loss = mean_val_loss
        stop_counter = 0

        file_path = os.path.join(RESULT_DIR, f"best.pth")
        torch.save(model.state_dict(), file_path)
        print(f"* Best model weights saved: {file_path}, epoch-{epoch}, loss-{best_loss:.4f}")

    # Save
    file_path = os.path.join(RESULT_DIR, "last.pth")
    torch.save(model.state_dict(), file_path)
    print(f"✅ Model weights saved: {file_path}\n")

    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),

        "best_loss": best_loss,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr_list": lr_list,
        "stop_counter": stop_counter,

        "epochs": EPOCHS,
        "training_epoch": epoch,
    }
    torch.save(checkpoint, checkpoint_path)

    if stop_counter > stop_counter_limit:
        print(f"Training is finished - epoch - {epoch + 1}/{EPOCHS}")
        break
    stop_counter += 1

plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(["train_loss", "val_loss"])
plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(["train_acc", "val_acc"])
plt.show()

plt.plot(lr_list)
plt.legend(["LR"])
plt.show()