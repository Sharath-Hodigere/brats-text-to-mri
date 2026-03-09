# Text-to-MRI Generation using BERT + 3D CNN

This project generates synthetic 3D brain MRI volumes from radiological text descriptions. You give it a clinical finding like "mass-like signal in the left frontal lobe with strong enhancement" and it outputs a 128x128x128 NIfTI volume.

Built on the BraTS Meningioma dataset. Trained on a remote Linux server over SSH using a conda environment.

## How it works

- **Text encoder**: BERT (bert-base-uncased) frozen, using the CLS token embedding
- **Decoder**: 4-stage 3D transposed convolution network, 8 cube up to 128 cube
- **Loss**: L1 + gradient edge loss weighted at 0.5
- **Output**: .nii.gz volume + 2D middle-slice PNG

## Setup
```bash
conda create -n text2mri python=3.10
conda activate text2mri
pip install -r requirements.txt
```

## Training
```bash
python train_ddp.py
```

Saves best checkpoint to text_to_mri_model_128.pth. Ran 200 epochs.

## Inference

Edit the text variable in inference.py then run:
```bash
python inference.py
```

Outputs generated_128.png and generated_128_volume.nii.gz

## Result

![Generated MRI](generated_128.png)

## Files

| File | Description |
|------|-------------|
| train_ddp.py | Dataset, model, training loop |
| inference.py | Generate MRI volume from text |
| requirements.txt | Dependencies |
