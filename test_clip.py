import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
import torchvision
torchvision.disable_beta_transforms_warning()
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
os.environ["HF_HOME"] = "/shared/shared/kevin_huggingface"
import random
import logging
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
from pytz import timezone
from datetime import datetime
import sys
import torchvision.transforms as T
from dataloader import *
import random
import sklearn.metrics as metrics
import glob
import argparse
from sklearn.metrics import roc_auc_score
from utils import *
from torch.utils.data import DataLoader as Dataloader
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import clip
from model.LearnablePrompts import LearnablePrompts
from torchvision.transforms import ToPILImage
from custom_prompts import return_live_prompts, return_print_replay, return_print_attack_prompts, return_replay_attack_prompts

# python test_clip.py --test_dataset Ca
parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str, default="N",)
parser.add_argument("--test_dataset", type=str)
parser.add_argument("--batch_size", type=int, default=75)
args = parser.parse_args()

source = args.train_dataset
target = args.test_dataset
batch_size_arg = args.batch_size

# --- Configuration ---
CLIP_MODEL_NAME = "ViT-B/32"
PROMPT_LENGTH = 16  # Should match the loaded ProText prompts
LEARNED_PROMPTS_PATH = "results/protext_learned_prompts.pth" # Path to your saved ProText prompts
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define text prompts for live and spoof classes
LIVE_CLASS_TEXT = return_live_prompts(target)
SPOOF_CLASS_TEXTS = return_print_replay(target)
ALL_PROMPTS = LIVE_CLASS_TEXT + SPOOF_CLASS_TEXTS
num_live_prompts = len(LIVE_CLASS_TEXT)

root='/shared/shared'
results_filename = source.replace('/', '') + '_to_' + target.replace('/', '')

# Set up logging
file_handler = logging.FileHandler(filename='/home/kevin/TCLIP/logger/'+ results_filename + '_test.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

# dataloader
for protocol in target.split('/'):
    target_dataset = FAS_Dataset(root=root, protocol=[protocol], train=False)
    if protocol == target.split('/')[0]:
        combined_target_dataset = target_dataset
    else:
        combined_target_dataset = ConcatDataset([combined_target_dataset, target_dataset])
target_loader = Dataloader(combined_target_dataset, batch_size=batch_size_arg, shuffle=True)
data_loader = target_loader

record = [1,100,100,100,100,100]
log_list = []

# test script
all_scores = []
all_labels = []
TP = 0.0000001
TN = 0.0000001
FP = 0.0000001
FN = 0.0000001

# --- Initialize CLIP Model and Pre-compute Text Features ---
try:
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE, jit=False)
    clip_model.eval() # Ensure model is in evaluation mode
except Exception as e:
    print(f"Error loading CLIP model '{CLIP_MODEL_NAME}': {e}", exc_info=True)
    sys.exit(1)

# Pre-compute text features for all defined prompts
with torch.no_grad():
    tokenized_prompts = clip.tokenize(ALL_PROMPTS).to(DEVICE)
    all_text_features = clip_model.encode_text(tokenized_prompts)
    all_text_features /= all_text_features.norm(dim=-1, keepdim=True) # Normalize features

    # Separate features for live and spoof prompts
    live_text_features = all_text_features[0:num_live_prompts] # Shape: [num_live_prompts, embedding_dim]
    spoof_text_features = all_text_features[num_live_prompts:] # Shape: [num_spoof_prompts, embedding_dim]
    #print("Text features for prompts computed and normalized.")

for i, data_batch in enumerate(tqdm(target_loader, desc=f"Evaluating {target}")):
    # Assuming your dataloader yields: rgb_images, depth_images, ir_images, labels
    # We will only use rgb_images for standard CLIP evaluation
    rgb_images, _, _, labels = data_batch # Modify if your dataloader structure is different

    rgb_images = rgb_images.to(DEVICE)
    labels = labels.to(DEVICE) # Keep labels on device for potential direct comparison if needed

    # Preprocess images for CLIP:
    # Convert image tensors to PIL Images, then apply CLIP's preprocessing
    # This loop is necessary because clip_preprocess expects PIL images or a batch of correctly formatted tensors.
    # If rgb_images are already in a format clip_preprocess can handle directly (e.g. normalized, correct size),
    # this can be simplified. ToPILImage() is robust.
    pil_images = [ToPILImage()(img_tensor.cpu()) for img_tensor in rgb_images]
    processed_images_batch = torch.stack([clip_preprocess(pil_img) for pil_img in pil_images]).to(DEVICE)

    with torch.no_grad():
        # Encode images
        image_features = clip_model.encode_image(processed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize image features

        # --- Calculate Cosine Similarities ---
        # Similarity with the "live" prompt
        # image_features: [batch_size, embedding_dim]
        # live_text_features.T: [embedding_dim, 1]
        # Result: [batch_size, 1] -> squeeze to [batch_size]
        sim_live = image_features @ live_text_features.T
        sim_live_mean = torch.mean(sim_live, dim=-1) # Shape: [batch_size]
        sim_live_max, _ = torch.max(sim_live, dim=-1) # Shape: [batch_size]

        # Similarities with all "spoof" prompts
        # spoof_text_features.T: [embedding_dim, num_spoof_prompts]
        # Result: [batch_size, num_spoof_prompts]
        sim_spoof_all = image_features @ spoof_text_features.T
        # For each image, find the maximum similarity to any of the spoof prompts
        sim_spoof_max, _ = torch.max(sim_spoof_all, dim=-1) # Shape: [batch_size]
        # Compute the mean of the top k maximum similarities for sim_spoof_all
        k = 1
        top_k_values, _ = torch.topk(sim_spoof_all, k, dim=-1)  # Shape: [batch_size, k]
        sim_spoof_top_k_mean = torch.mean(top_k_values, dim=-1)  # Shape: [batch_size]

        # --- Scoring for Anti-Spoofing ---
        # Score = Similarity to "live" - Max Similarity to "spoof"
        # A higher score indicates more "live-like".
        # A positive score means it's more similar to "live" than to any "spoof" type.
        # A negative score means it's more similar to at least one "spoof" type than to "live".
        # This is a common way to get a single score for binary classification.
        final_scores_batch = sim_live_mean - sim_spoof_top_k_mean

        # Simpler alternative: use similarity to "live" directly as the score.
        # Assumes label 1 = live, label 0 = spoof. Higher sim_live should correlate with live.
        #final_scores_batch = sim_live


    all_scores.extend(final_scores_batch.cpu().numpy())
    all_labels.extend(labels.cpu().numpy()) # Collect ground truth labels

all_scores_np = np.array(all_scores)
all_labels_np = np.array(all_labels)

fpr, tpr, thresholds_cs = metrics.roc_curve(all_labels_np, all_scores_np)
threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

for i in range(len(all_scores_np)):
    score = all_scores_np[i]
    if (score >= threshold_cs and all_labels_np[i] == 1):
        TP += 1
    elif (score < threshold_cs and all_labels_np[i] == 0):
        TN += 1
    elif (score >= threshold_cs and all_labels_np[i] == 0):
        FP += 1
    elif (score < threshold_cs and all_labels_np[i] == 1):
        FN += 1

APCER = FP / (TN + FP)
NPCER = FN / (FN + TP)

step = 0
if record[1]>((APCER + NPCER) / 2):
    record[0]=step + 1
    record[1]=((APCER + NPCER) / 2)
    record[2]=roc_auc_score(all_labels_np, all_scores_np)
    record[3]=APCER
    record[4]=NPCER

logging.info(f"model: CLIP")
logging.info(f"Train on {source}, Test on {target}, Batch Size = {str(batch_size_arg)}")
logging.info(f"ACER {record[1]:.6f} AUC {record[2]:.6f} APCER {record[3]:.6f} BPCER {record[4]:.6f}\n")