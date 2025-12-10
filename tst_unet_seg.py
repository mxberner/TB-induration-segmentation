import os
import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


# ======================================================
# COCO dataset -> 2-channel segmentation masks
# ======================================================

class COCOTSTSegmentationDataset(Dataset):
    """
    Segmentation dataset for TST images with two classes:

    channel 0: induration (category_id = ind_cat_id)
    channel 1: sticker    (category_id = sticker_cat_id)

    - Reads a COCO JSON.
    - For each image:
        * Collect all induration annotations and all sticker annotations.
        * Rasterize polygons if 'segmentation' is present.
        * Otherwise, use bounding boxes as filled rectangles.

    Returns:
        image: (3, H, W) float tensor
        mask:  (2, H, W) float tensor in {0, 1}
    """

    def __init__(
        self,
        coco_json_path,
        image_root,
        ind_cat_id=1,
        sticker_cat_id=2,
        resize=(256, 256),
    ):
        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        self.image_root = image_root
        self.ind_cat_id = ind_cat_id
        self.sticker_cat_id = sticker_cat_id
        self.resize = resize

        # Index images and annotations
        self.images = {img["id"]: img for img in coco["images"]}
        annos_by_img = defaultdict(list)
        for a in coco["annotations"]:
            annos_by_img[a["image_id"]].append(a)

        self.samples = []
        for img_id, img_info in self.images.items():
            annos = annos_by_img.get(img_id, [])

            # We keep images that have at least one of the two categories
            has_ind = any(a["category_id"] == ind_cat_id for a in annos)
            has_sticker = any(a["category_id"] == sticker_cat_id for a in annos)
            if not (has_ind or has_sticker):
                continue

            self.samples.append(
                {
                    "image_id": img_id,
                    "file_name": img_info["file_name"],
                    "annos": annos,
                }
            )

        # Simple, deterministic transforms (no random aug to avoid sync issues)
        self.img_transform = transforms.Compose([
            transforms.Resize(resize, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.mask_size = resize

    def __len__(self):
        return len(self.samples)

    def _rasterize_annos(self, img_size, annos):
        """
        Build a 2-channel mask for one image.

        img_size: (width, height)
        annos: list of annotations for this image

        Returns: np.ndarray of shape (2, H, W) with values {0,1}
        """
        W, H = img_size
        mask_ind = Image.new("L", (W, H), 0)      # induration
        mask_stick = Image.new("L", (W, H), 0)    # sticker

        draw_ind = ImageDraw.Draw(mask_ind)
        draw_stick = ImageDraw.Draw(mask_stick)

        for a in annos:
            cat = a["category_id"]
            target_draw = None
            if cat == self.ind_cat_id:
                target_draw = draw_ind
            elif cat == self.sticker_cat_id:
                target_draw = draw_stick
            else:
                continue

            # Prefer segmentation polygons if present
            seg = a.get("segmentation", None)
            if isinstance(seg, list) and len(seg) > 0:
                # COCO polygon format: list of [x1,y1,x2,y2,...]
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                    target_draw.polygon(xy, outline=1, fill=1)
            else:
                # Fallback: bounding box [x, y, w, h]
                x, y, w, h = a["bbox"]
                x0, y0 = x, y
                x1, y1 = x + w, y + h
                target_draw.rectangle([x0, y0, x1, y1], outline=1, fill=1)

        mask_ind = np.array(mask_ind, dtype=np.uint8)
        mask_stick = np.array(mask_stick, dtype=np.uint8)

        mask = np.stack([mask_ind, mask_stick], axis=0)  # (2, H, W)
        return mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_root, sample["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Build full-res mask
        W, H = img.size
        mask_np = self._rasterize_annos((W, H), sample["annos"])  # (2, H, W)

        # Resize image and mask to the same size
        img_resized = self.img_transform(img)  # (3, H', W')

        mask_resized = []
        for c in range(mask_np.shape[0]):
            m_c = Image.fromarray(mask_np[c], mode="L")
            m_c = m_c.resize(self.mask_size, resample=Image.NEAREST)
            mask_resized.append(np.array(m_c, dtype=np.uint8))

        mask_resized = np.stack(mask_resized, axis=0)            # (2, H', W')
        mask_tensor = torch.from_numpy(mask_resized).float()     # {0,1}

        return img_resized, mask_tensor


# ======================================================
# UNet for 2-class segmentation
# ======================================================

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscaling then DoubleConv (bilinear or transposed conv)"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if needed (for odd shapes)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetSegmentation(nn.Module):
    """
    U-Net for segmentation:
    - in_channels = 3 (RGB)
    - out_channels = 2 (induration, sticker)
    """
    def __init__(self, in_channels=3, out_channels=2, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # (B, 2, H, W)
        return logits


# ======================================================
# Training / evaluation loops
# ======================================================

def train_one_epoch_seg(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)  # (B, 2, H, W)

        optimizer.zero_grad()
        logits = model(imgs)  # (B, 2, H, W)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def eval_one_epoch_seg(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)

            running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


# ======================================================
# Main
# ======================================================

def main():
    COCO_JSON = "data372/_annotations.coco.json"  # adjust to your path
    IMAGE_ROOT = "data372"                        # folder with images

    batch_size = 4
    num_epochs = 10
    lr = 1e-4
    val_split = 0.2
    resize = (256, 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = COCOTSTSegmentationDataset(
        coco_json_path=COCO_JSON,
        image_root=IMAGE_ROOT,
        ind_cat_id=1,
        sticker_cat_id=2,
        resize=resize,
    )

    print(f"Total usable images (with induration or sticker): {len(dataset)}")



    # ---- Manual random split (no sklearn needed) ----
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)

    split = int(num_samples * (1.0 - val_split))
    train_idx = indices[:split]
    val_idx = indices[split:]
    # -----------------------------------------------

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = UNetSegmentation(in_channels=3, out_channels=2, bilinear=True).to(device)
    criterion = nn.BCEWithLogitsLoss()  # multi-label per pixel
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch_seg(model, train_loader, optimizer, criterion, device)
        val_loss = eval_one_epoch_seg(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    # ======================================================
    # Save trained model weights
    # ======================================================
    save_path = "unet_tst_segmentation.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Example: run one batch through and check shapes
    model.eval()
    imgs, masks = next(iter(val_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    print("Input images shape:", imgs.shape)   # (B, 3, H, W)
    print("GT masks shape:    ", masks.shape)  # (B, 2, H, W)
    print("Pred masks shape:  ", preds.shape)  # (B, 2, H, W)




if __name__ == "__main__":
    main()
