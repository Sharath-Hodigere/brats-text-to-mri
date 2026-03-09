import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from transformers import BertTokenizer
from train_ddp import TextToMRIModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load trained model
# ------------------------
model = TextToMRIModel().to(device)
model.load_state_dict(torch.load("text_to_mri_model_128.pth", map_location=device))
model.eval()

# ------------------------
# Tokenizer
# ------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# edit this to whatever clinical description you want to generate from
text = "A mass-like abnormal signal in the left frontal lobe with strong enhancement"

tokens = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

input_ids = tokens["input_ids"].to(device)

# ------------------------
# Generate MRI volume
# ------------------------
with torch.no_grad():
    generated = model(input_ids)

generated = generated[0, 0].cpu().numpy()
print("Generated shape:", generated.shape)  # (128, 128, 128)

# normalize for visualization
generated = (generated - generated.min()) / (generated.max() - generated.min() + 1e-8)

# ------------------------
# Save middle axial slice
# ------------------------
mid_slice = generated[:, :, generated.shape[2] // 2]

plt.figure(figsize=(6, 6))
plt.imshow(mid_slice, cmap="gray")
plt.axis("off")
plt.title("Generated MRI From Text (128³)")
plt.savefig("generated_128.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved 2D slice → generated_128.png")

# ------------------------
# Save full 3D NIfTI volume
# ------------------------
nii = nib.Nifti1Image(generated, np.eye(4))
nib.save(nii, "generated_128_volume.nii.gz")

print("Saved 3D volume → generated_128_volume.nii.gz")
