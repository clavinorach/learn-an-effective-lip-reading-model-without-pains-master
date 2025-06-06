# run_training.py (Fixed)
import modal
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numbers
import random
import glob
from turbojpeg import TurboJPEG
import scipy

# --- Configuration ---
NFS_NAME = "lipreading-dataset-nfs"
APP_NAME = "lipreading-training-app-final"

# --- Modal App Definition ---
app = modal.App(APP_NAME)

# --- Environment Container Definition ---
# FIXED: Reorder commands to create symlink before running pip
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04")
    .apt_install("python3", "python3-pip", "python3-venv")
    .run_commands("ln -s /usr/bin/python3 /usr/bin/python")  # Create symlink first
    .add_local_file("modal_requirements.txt", "/modal_requirements.txt")  # Updated to use add_local_file
    .run_commands(
        "python -m pip install --upgrade pip wheel uv",
        "pip install -r /modal_requirements.txt"
    )
)

# --- Network File System Definition ---
nfs = modal.NetworkFileSystem.from_name(NFS_NAME, create_if_missing=True)

# ============================================================================
# === CODE COPIED FROM YOUR PROJECT (WITH FIXES) ===
# ============================================================================

# ----------------------------------------------------------------------------
# 1. Content from: LSR.py
# ----------------------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ----------------------------------------------------------------------------
# 2. Content from: model/model.py
# ----------------------------------------------------------------------------
class VideoModel(nn.Module):
    def __init__(self, **kwargs):
        super(VideoModel, self).__init__()
        self.n_class = kwargs.get('n_class', 1000)
        self.se = kwargs.get('se', False)
        self.temporal_pool = kwargs.get('temporal_pool', True)
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.backend_tcn = nn.Sequential(
            ResNet(BasicBlock, [2, 2, 2, 2], se=self.se)
        )
        if self.temporal_pool:
            self.backend_gru = nn.GRU(512, 1024, 2, batch_first=True, bidirectional=True, dropout=0.2)
        else:
            self.backend_gru = nn.GRU(512*29, 1024, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(2048, self.n_class)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend_tcn(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        if not self.temporal_pool:
            x = x.view(x.size(0), x.size(1), -1)
        else:
            x = torch.mean(x, dim=(3,4))
        self.backend_gru.flatten_parameters()
        x, _ = self.backend_gru(x)
        x = self.fc(x)
        x = torch.mean(x, 1)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, se=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, self.inplanes, x.size(3), x.size(4))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0)//29, 29, 512, x.size(2), x.size(3))
        x = x.transpose(1, 2)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        if self.se:
            self.selayer = SELayer(planes)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.se:
            out = self.selayer(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ----------------------------------------------------------------------------
# 3. Content from: utils/cvtransforms.py
# ----------------------------------------------------------------------------
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, imgs):
        if self.padding > 0:
            imgs = [np.pad(img, ((self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0) for img in imgs]

        h, w = imgs[0].shape[0], imgs[0].shape[1]
        th, tw = self.size
        if w == tw and h == th:
            return imgs
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img[y1:y1+th, x1:x1+tw] for img in imgs]

class HorizontalFlip(object):
    def __call__(self, imgs):
        if random.random() < 0.5:
            return [np.fliplr(img) for img in imgs]
        return imgs

# ----------------------------------------------------------------------------
# 4. Content from: utils/dataset.py
# ----------------------------------------------------------------------------
class idev1Dataset(object):
    def __init__(self, phase, args=None):
        self.args = args
        self.phase = phase
        self.jpeg = TurboJPEG()
        self.data_path = "/data/idev1_roi_80_116_175_211_npy_gray_pkl_jpeg"
        self.file_list = glob.glob(os.path.join(self.data_path, self.phase, '*.pkl'))
        self.transforms = self.get_transforms(phase)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        pkl_path = self.file_list[idx]
        try:
            with open(pkl_path, 'rb') as f:
                data = torch.load(f)
            video = [self.jpeg.decode(img, pixel_format=0) for img in data['video']]
            video = self.transforms(video)
            video_np = np.array(video).copy()
            return torch.FloatTensor(video_np).unsqueeze(1), data['label']
        except Exception as e:
            print(f"Failed to load file: {pkl_path}. Error: {e}")
            return torch.zeros(29, 1, 88, 88), -1

    def get_transforms(self, phase):
        if phase == 'train':
            return Compose([RandomCrop(88), HorizontalFlip(),])
        else:
            return CenterCrop(88)

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, imgs):
        h, w = imgs[0].shape[0], imgs[0].shape[1]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [img[y1:y1+th, x1:x1+tw] for img in imgs]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs

# --- Main Training Function ---
@app.function(
    image=image,
    gpu="H100",  # Changed to more commonly available GPU
    network_file_systems={"/data": nfs},
    timeout=43200  # 12 hours
)
def train_model():
    learning_rate = 3e-4
    batch_size = 128
    num_workers = 16
    max_epoch = 120
    n_class = 8  # Adjust based on your dataset
    save_prefix = "/data/checkpoints/lipreading-model/"
    os.makedirs(save_prefix, exist_ok=True)

    model = VideoModel(n_class=n_class, se=True, temporal_pool=True)
    model = model.cuda()
    model = nn.DataParallel(model)

    train_data = idev1Dataset('train')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_data = idev1Dataset('test')
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print("\n--- Starting Training ---")
    for epoch in range(max_epoch):
        start_time = time.time()
        model.train()
        total_loss = 0
        for i, (video, label) in enumerate(train_loader):
            # Skip error batches (label -1)
            if -1 in label:
                print(f"Skipping error batch at iteration {i}")
                continue
            video = video.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            output = model(video)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for video, label in test_loader:
                if -1 in label:
                    continue
                video = video.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                output = model(video)
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == label).sum().item()
                total_samples += label.size(0)

        if total_samples > 0:
            accuracy = 100 * total_correct / total_samples
        else:
            accuracy = 0
            
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{max_epoch} | Avg Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}% | Time: {epoch_time:.2f}s")

        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(save_prefix, f'epoch_{epoch+1}.pth')
            torch.save(model.module.state_dict(), model_path)
            print(f"Checkpoint saved at {model_path}")

    print("\n--- Training Complete ---")

@app.local_entrypoint()
def main():
    train_model.remote()