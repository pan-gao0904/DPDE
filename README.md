This repository contains the official implementation of the paper "DPDE: A Physics-Diffusion Framework for Underwater Image Enhancement".  
### 1. Install Dependencies
To run this code, make sure you have Python 3.8+ and install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt



2. Folder Structure
dataset/: Folder for input images.

dataset/water_val_16_256/sr_16_256: Contains degraded underwater images for inference.

dataset/water_val_16_256/hr_256: Contains reference images for evaluation (optional).

dataset/train: Folder for training data for PGLAN.

dataset/water_train_16_128: For training the DMnet model.

checkpoints/: Pre-trained model checkpoint for PGLAN.

config/: Configuration files for the models.

outputs/: Output folder for saving results after inference and training.


⚡ Inference
Image Enhancement
To perform inference (enhance a degraded underwater image), place the degraded images in the dataset/water_val_16_256/sr_16_256 folder. If you have reference images (for metrics calculation), place them in the dataset/water_val_16_256/hr_256 folder.

Run the following command to get the enhanced images:

python infer_PGLAN_ddpm_all_files_save.py \
  --config_ddpm config/underwater.json \
  --stage1_ckpt checkpoints/best_model.pth \
  --output_dir outputs/final_results

The enhanced images will be saved in the outputs/final_results folder.

Estimate Transmission and Atmospheric Light
To estimate the transmission rate and atmospheric light for the PGLAN model, run the following command:
python infer_PGLAN_T_and_A.py \
  --input dataset/val/input \
  --model checkpoints/best_model.pth \
  --output_dir outputs/test_results_PGLAN \
  --gt_dir dataset/val/gt
This will estimate the transmission and atmospheric light, and save the results in outputs/test_results_PGLAN.


🚀 Training
Train PGLAN
If you want to train the PGLAN model, place your training data in the dataset/train/ folder, and run:

python train_PGLAN.py 

Train DMnet
To train the DMnet model, place the training data in the dataset/water_train_16_128/ folder, and run:

python train_DMnet.py 


🛠 License
This code is released under the MIT License. See the LICENSE file for more details.


