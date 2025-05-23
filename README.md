# Hemlock: Style Cloaking to Protect Artistic Identity

Hemlock is a defensive AI system designed to protect digital artworks from being mimicked by text-to-image diffusion models such as Stable Diffusion and MidJourney. It works by applying subtle perturbations (style cloaks) at the feature level, making it difficult for AI models to replicate an artist's style while preserving visual fidelity.

---

## Features

- U-Net inspired encoder-decoder architecture for feature extraction and image reconstruction.
- CloakBlock module to replace stylistic features using cosine similarity.
- Balanced loss function combining LPIPS (perceptual loss) and MSE.
- Evaluated using RMSE, PSNR, and SSIM for visual and structural fidelity.

---

## Architecture

### Encoder-Decoder Framework

- **Encoder**: Stacked Conv layers + MaxPooling to reduce spatial dimensions and extract features.
  - `ConvBlock_encoder`: Two convolutional layers with batch normalization and ReLU activation.
  - `EncoderBlock`: Combines `ConvBlock_encoder` with max-pooling.

- **Decoder**: Uses deconvolution to reconstruct images from latent representations.
  - `ConvBlock_decoder`: Similar to encoder but uses Tanh activation in the final layer.
  - `DecoderBlock`: Combines `ConvBlock_decoder` with transposed convolution.

### CloakBlock Framework

- **Feature Extraction**: Extract features from original and style-transferred images using encoders.
- **Cosine Similarity**: Compute similarity between original and styled feature maps:
  \[
  S(f_o, f_s) = \frac{f_o \cdot f_s}{\|f_o\| \|f_s\|}
  \]
- **Feature Integration**: Replace features in original image where similarity is in the range [0.12, 0.45].

---

## Loss Functions

To minimize visual distortion while cloaking, a combined loss is used:

### Perceptual Loss (LPIPS)
Compares high-level features from pre-trained VGG network to preserve perceptual similarity:
\[
LPIPS = \sum_{i=1}^L \|\phi_i(\hat{y}) - \phi_i(y)\|^2_2
\]

### MSE / PSNR Loss
Pixel-level loss ensuring minimal perturbation:
\[
MSE(x, y) = \frac{1}{n} \sum_{i=1}^n (x_i - y_i)^2
\]
\[
PSNR = 10 \cdot \log_{10} \left(\frac{MAX^2}{MSE}\right)
\]

### Combined Loss
\[
\text{Total Loss} = (1 - \alpha) \cdot LPIPS + \alpha \cdot MSE
\]

---

## Evaluation Metrics

- **RMSE**: Measures pixel-level deviation.
- **PSNR**: Evaluates image fidelity.
- **SSIM**: Assesses structural similarity based on luminance, contrast, and texture.

---

## Experimental Setup

### Hardware

- CPU: Intel Core i5-9400KF
- GPU: NVIDIA GTX 1050 (4 GB VRAM)
- RAM: 8 GB
- Storage: 256 GB SSD

### Software

- Language: Python 3.x
- Frameworks: PyTorch, torchvision
- Pretrained Models: VGG19
- Libraries: LPIPS, NumPy, Pandas

### Dataset

- Sources: Fine Arts and Photography Society (FAPS), Echoes Club
- 4,437 high-resolution cyberpunk-themed images
- Preprocessed to 512×512 resolution, duplicates and low-res images removed
- Feature extraction done using VGG19

---

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/Hemlock.git
cd Hemlock
pip install -r requirements.txt
