import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import logging
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()


# ----------------------
# Dataset
# ----------------------
class BraTSTextMRIDataset(Dataset):
    def __init__(self, image_dir, json_path, modality="t2f"):
        self.image_dir = image_dir
        self.modality = modality

        with open(json_path) as f:
            self.text_data = json.load(f)

        self.ids = []
        for key in self.text_data.keys():
            filename = f"{key}-{modality}.nii.gz"
            if os.path.exists(os.path.join(image_dir, filename)):
                self.ids.append(key)

        logger.info(f"Using {len(self.ids)} matched text-MRI pairs")

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        case_id = self.ids[idx]
        text = self.text_data[case_id]

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        img_path = os.path.join(self.image_dir, f"{case_id}-{self.modality}.nii.gz")
        img = nib.load(img_path).get_fdata()

        # Normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return tokens["input_ids"].squeeze(0), img_tensor


# ----------------------
# Edge Loss
# ----------------------
def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    dx_target = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

    dy_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dy_target = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]

    dz_pred = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dz_target = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

    return (
        F.l1_loss(dx_pred, dx_target) +
        F.l1_loss(dy_pred, dy_target) +
        F.l1_loss(dz_pred, dz_target)
    )


# ----------------------
# 128^3 Text → MRI Model
# ----------------------
class TextToMRIModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Start from 8^3 latent cube
        self.fc = nn.Linear(768, 512 * 8 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, 2, 1),  # 8 → 16
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(256, 128, 4, 2, 1),  # 16 → 32
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(128, 64, 4, 2, 1),   # 32 → 64
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(64, 32, 4, 2, 1),    # 64 → 128
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),

            nn.Conv3d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv3d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_ids):
        text_features = self.text_encoder(input_ids).last_hidden_state[:, 0, :]
        x = self.fc(text_features)
        x = x.view(-1, 512, 8, 8, 8)
        x = self.decoder(x)
        return x


# ----------------------
# Training
# ----------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_dir = os.path.expanduser("~/data/BraTS_MEN/images")
    json_path = os.path.expanduser("~/data/BraTS_MEN/global_finding.json")

    dataset = BraTSTextMRIDataset(image_dir, json_path)

    dataloader = DataLoader(
        dataset,
        batch_size=1,   # safer for 128^3 volumes
        shuffle=True
    )

    model = TextToMRIModel().to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4
    )

    l1_criterion = nn.L1Loss()

    num_epochs = 200
    best_loss = float("inf")

    for epoch in range(num_epochs):

        logger.info(f"Starting epoch {epoch+1}")
        epoch_loss = 0

        for batch_idx, (input_ids, imgs) in enumerate(dataloader):

            input_ids = input_ids.to(device)
            imgs = imgs.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            imgs = F.interpolate(imgs, size=outputs.shape[2:])

            l1 = l1_criterion(outputs, imgs)
            edge = gradient_loss(outputs, imgs)

            loss = l1 + 0.5 * edge

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1} Batch {batch_idx} "
                    f"Loss {loss.item():.5f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} Average Loss {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "text_to_mri_model_128.pth")
            logger.info("Saved best 128^3 model")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
