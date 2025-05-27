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
from model.AlignCLIP import CLIP
from torchvision.transforms import ToPILImage
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from sklearn.decomposition import PCA

# python test_clip_protext.py --train_dataset N --test_dataset Ca --batch_size 75
parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--test_dataset", type=str)
parser.add_argument("--batch_size", type=int, default=16)
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
LIVE_CLASS_TEXT = [
    "a person looking directly at the camera",
    "a photo of a person's face with natural skin texture and depth",
    "a person with realistic skin texture",
    "an authentic portrait with detailed skin texture",
    "a real portrait showing subtle facial asymmetry"
]
SPOOF_CLASS_TEXTS = [
    # print
    "a printed photograph of a face",
    "a paper photo of a face",
    "a person's face printed on paper",
    "a printed image of a face",
    "a low-quality printed face photograph",
    # replay
    "a face displayed on a screen",
    "a face playback on a tablet",
    "a face video playing on a monitor",
    "a human face on a pixelated display",
    "a screen-captured face with digital artifacts",
    # mask
    "a realistic 3D face mask",
    "a hyper-realistic mask mimicking human skin",
    "a 3D-printed face mask worn by someone",
    "a person disguised behind a realistic face covering",
    "a fabricated mask imitating human features",
    # partial
    "only part of a face visible",
    "a face with occluded regions",
    "a face with selective areas hidden",
    "a partially masked human face",
    "a partial face presentation for spoofing",
    # mannequin
    "a mannequin head",
    "a plastic mannequin face",
    "a store display mannequin",
    "a rigid face model lacking flexibility",
    "a synthetic face model with artificial eyes"
]
num_prompts_per_class = 5

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

# --- Helper Functions ---
def get_text_features_with_prompts(clip_model, learnable_prompts_module, text_prompts_list, device):
    """
    Generates text features using CLIP and loaded learnable prompts.
    """
    learnable_prompts_module.eval() # Ensure prompts are in eval mode if loaded
    all_text_features = []

    # Accessing CLIP model components needed for manual text encoding with prompts
    token_embedding_layer = clip_model.token_embedding
    positional_embedding = clip_model.positional_embedding
    transformer = clip_model.transformer
    ln_final = clip_model.ln_final
    text_projection = clip_model.text_projection
    dtype = clip_model.dtype
    context_length = clip_model.context_length # Typically 77

    original_prompts = learnable_prompts_module.prompts.data.clone() # Save original prompts if module is shared

    with torch.no_grad():
        for text_input_batch in [text_prompts_list]: # Process as a single batch
            # Tokenize the class name template without prompts first
            # We need to find where the actual text tokens start after prompting
            # The structure "a photo of a [CLASS]" ensures placeholder tokens are at the end usually
            tokenized_texts = clip.tokenize(text_input_batch, context_length=context_length).to(device)

            # Get embeddings for the tokenized text (these will be combined with prompts)
            # The learnable_prompts_module expects class_token_ids
            # However, its forward method is designed to prepend to already embedded tokens for ProText
            # Let's adjust. The ProText way:
            # 1. Tokenize "a photo of a [CLASS]"
            # 2. Embed these tokens using clip_model.token_embedding
            # 3. Prepend learnable_prompts.prompts (which are already embeddings)
            
            # For the LearnablePrompts class defined above:
            # It takes token_ids and embeds them, then prepends.
            # This is slightly different from the last ProText script where prompts were nn.Parameters
            # and class embeddings were passed. Let's fix LearnablePrompts or adapt usage.
            # The provided LearnablePrompts class re-embeds. This is fine.
            
            # The LearnablePrompts already prepends its own learnable `prompts`
            # to the embeddings of `class_token_ids`.
            prompted_embeddings, _ = learnable_prompts_module(tokenized_texts, text_input_batch) # (batch, PROMPT_LENGTH + original_seq_len, embed_dim)

            # Add positional embeddings
            # Ensure positional_embedding sequence length matches
            max_seq_len = positional_embedding.shape[0]  # Typically 77 for CLIP
            prompted_embeddings_truncated = prompted_embeddings[:, :max_seq_len, :]
            x = prompted_embeddings_truncated + positional_embedding.type(dtype)[:max_seq_len]
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = x.type(dtype)  # Ensure dtype consistency
            
            # Pass through transformer
            x = transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = ln_final(x).type(dtype)

            # Gather the feature for the EOT token from the prompted sequence
            # EOT is at original EOT position + PROMPT_LENGTH
            # For tokenized_texts from clip.tokenize, EOT is where argmax is.
            eot_indices_original = tokenized_texts.argmax(dim=-1)
            # Ensure eot_indices_prompted are valid for each item in the batch
            eot_indices_prompted = torch.zeros_like(eot_indices_original)
            for i in range(tokenized_texts.shape[0]):
                 # Find actual end of text tokens before padding for each item
                actual_text_len = (tokenized_texts[i] != 0).sum().item() # Length of non-zero tokens
                # EOT token is the last non-padding token for clip.tokenize() output
                # If PROMPT_LENGTH + actual_text_len > context_length, it's truncated.
                # We need the EOT relative to the *start* of class tokens.
                # The EOT is at the end of the *original content* part of tokenized_texts.
                # Let's find the last non-zero token position in the original tokenized text.
                # Example: [SOS, "a", "photo", "of", "real", "face", EOS, PAD, PAD ...]
                #          [  0,   1,     2,    3,     4,      5,   6,   7,   8 ...]
                # Prompt will be [P1..P16, SOS, a, photo, of, real, face, EOS, PAD...]
                # So EOT token index in prompted seq is PROMPT_LENGTH + eot_indices_original[i]

                # Correct EOT index for prompted sequence.
                # argmax gives the index of the first occurrence of the highest value (usually EOS token ID).
                # This index is relative to the start of the 77 token sequence.
                eot_idx_in_original_token_seq = eot_indices_original[i]
                eot_indices_prompted[i] = PROMPT_LENGTH + eot_idx_in_original_token_seq
                
                # Boundary check, though with standard context length it should be fine
                if eot_indices_prompted[i] >= max_seq_len:
                    eot_indices_prompted[i] = max_seq_len -1 # Fallback to last token
                    print(f"Warning: EOT index {eot_indices_prompted[i]} out of bounds for batch {i}. Clamping to {max_seq_len - 1}.")

            # Select features at EOT position
            # x.shape is [batch_size, seq_len_prompted, transformer_width]
            # text_features_batch = x[torch.arange(x.shape[0]), eot_indices_prompted] @ text_projection
            
            # More robust way to get EOT features:
            batch_text_features = []
            for i in range(x.shape[0]):
                batch_text_features.append(x[i, eot_indices_prompted[i]])
            text_features_batch = torch.stack(batch_text_features) @ text_projection

            all_text_features.append(text_features_batch)

    if not all_text_features:
        raise ValueError("No text features were generated.")
    
    final_text_features = torch.cat(all_text_features, dim=0)
    return final_text_features / final_text_features.norm(dim=-1, keepdim=True)

# --- TCLIP EM Algorithm ---
def tclip_em_algorithm(Z_visual, T_text_anchors, lambda_reg, temperature_reg, gamma_reg, iterations, num_test_samples):
    """
    Performs TCLIP EM algorithm.
    Args:
        Z_visual (Tensor): N x D visual embeddings. (N samples, D dimensions)
        T_text_anchors (Tensor): (K x num_prompts_per_class) x D text anchor embeddings. (K clusters, D dimensions)
        lambda_reg (float): Lambda hyperparameter for cluster proportion entropy.
        gamma_reg (float): Gamma hyperparameter for anchoring means to text anchors.
        iterations (int): Number of EM iterations.

    Returns:
        W_means (Tensor): K x D learned cluster means.
        U_assignments (Tensor): N x K soft assignments.
        U_hat_proportions (Tensor): K cluster proportions.
    """
    N = Z_visual.shape[0] + T_text_anchors.shape[0] # N visual samples + K * num_prompts_per_class text anchors
    K = int(T_text_anchors.shape[0] / num_prompts_per_class)

    # query + text support
    Z_all = torch.cat((Z_visual, T_text_anchors), dim=0)

    # PCA: Visual embeddings (Z_n) and text anchors (T_k)
    '''Z_all = PCA(n_components=50).fit_transform(Z_all.cpu().numpy())
    # Convert to tensors
    Z_all = torch.tensor(Z_all, device=DEVICE)
    # Normalize visual embeddings
    Z_all = F.normalize(Z_all, dim=1) # Normalize along the feature dimension'''

    # Initialization
    W_means = torch.zeros(K, Z_all.shape[1], device=DEVICE) # K x D
    for i in range(K):
        W_mean = Z_all[Z_visual.shape[0] + i * num_prompts_per_class: Z_visual.shape[0] + (i + 1) * num_prompts_per_class].mean(dim=0, keepdim=True)
        W_means[i] = W_mean.squeeze(0) # K x D
    W_means = W_means + 0.01 * torch.randn_like(W_means)

    W_means_anchor = W_means.clone().detach().float() # Initialize means with text anchors
    U_hat_proportions = torch.ones(K, device=DEVICE) / K # Initialize proportions uniformly
    U_assignments = torch.zeros(N, K, device=DEVICE)

    for iteration in range(iterations):
        # --- E-step: Update U_assignments (u_nk) ---
        dist_sq = torch.cdist(Z_all, W_means, p=2).pow(2) # N x K, squared Euclidean distances
        log_u_hat_safe = torch.log(U_hat_proportions + 1e-10) # Add epsilon for numerical stability
        exponent = (lambda_reg / N) * log_u_hat_safe.unsqueeze(0) - 0.5 * dist_sq
        exponent = exponent / temperature_reg # Apply temperature scaling

        U_assignments = F.softmax(exponent, dim=1)

        # fix assignment for text anchors
        for i in range(T_text_anchors.shape[0]):
            U_assignments[Z_visual.shape[0] + i, :] = 0
            U_assignments[Z_visual.shape[0] + i, int(i / num_prompts_per_class)] = 1

        # --- M-step ---
        # Update U_hat_proportions (cluster proportions)
        U_hat_proportions = U_assignments.mean(dim=0) # Sum over N samples, then divide by N
        if iteration == 19: 
            #print(f"    Proportions: {U_hat_proportions}")
            continue

        # Update W_means (cluster means)
        sum_u_nk = U_assignments.sum(dim=0) # K, sum of assignments for each cluster
        
        # Numerator: sum_n u_nk * z_n for each k
        sum_u_nk_z_n = torch.matmul(U_assignments.T, Z_all) # K x N @ N x D -> K x D
        numerator = sum_u_nk_z_n + gamma_reg * W_means_anchor
        denominator = sum_u_nk.unsqueeze(1) + gamma_reg # K x 1
        #numerator = sum_u_nk_z_n
        #denominator = sum_u_nk.unsqueeze(1)

        W_means = numerator / (denominator + 1e-10) # Add epsilon for stability

        # Calculate objective value (optional, for monitoring convergence)
        term1 = 0.5 * (U_assignments * dist_sq).sum()
        term2 = -lambda_reg * (U_hat_proportions * torch.log(U_hat_proportions + 1e-10)).sum()
        term3 = temperature_reg * (U_assignments * torch.log(U_assignments + 1e-10)).sum()
        term4 = 0.5 * gamma_reg * ((W_means - W_means_anchor)**2).sum()
        objective = term1 + term2 + term3 + term4
        #print(f"    Objective: {objective.item():.4f}")
        #print(f"EM Iteration {iteration:02d} [Objective: {objective.item():.4f} data-fitting: {term1.item():.4f}, complexity: {term2.item():.4f}, barrier: {term3.item():.4f}, anchor: {term4.item():.4f}]")


    return W_means, U_assignments, U_hat_proportions

#logging.info(f"# of testing: {len(data_loader)}")

record = [1,100,100,100,100,100]
log_list = []

# test script
score_list = []
Total_score_list_cs = []
Total_score_list_all = []
label_list = []
TP = 0.0000001
TN = 0.0000001
FP = 0.0000001
FN = 0.0000001

# 1. Load CLIP Model
try:
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE, jit=False)
    clip_model.eval() # Ensure model is in evaluation mode
    TEXT_FEATURE_DIM = clip_model.text_projection.shape[1]
    token_embedding_layer = clip_model.token_embedding
except Exception as e:
    print(f"Error loading CLIP model '{CLIP_MODEL_NAME}': {e}", exc_info=True)
    sys.exit(1)

# 2. Load Learnable Prompts module
if not os.path.exists(LEARNED_PROMPTS_PATH):
    print(f"Error: Learned prompts file not found at {LEARNED_PROMPTS_PATH}")
    exit(0)

all_class_texts = LIVE_CLASS_TEXT + SPOOF_CLASS_TEXTS

# Pass the actual token embedding layer (nn.Embedding) from CLIP
learnable_prompts_module = LearnablePrompts(
    prompt_length=PROMPT_LENGTH,
    embed_dim=TEXT_FEATURE_DIM,
    clip_token_embedding=token_embedding_layer, # Pass the actual nn.Embedding layer
).to(DEVICE)

learnable_prompts_module.load_state_dict(torch.load(LEARNED_PROMPTS_PATH, map_location=DEVICE, weights_only=True))
learnable_prompts_module.eval()
print(f"Learned prompts loaded from {LEARNED_PROMPTS_PATH}")

# 3. Prepare Text Anchor Embeddings (T_k)
use_protext = True
with torch.no_grad():
    if use_protext:
        all_class_texts = LIVE_CLASS_TEXT + SPOOF_CLASS_TEXTS
        all_text_features = get_text_features_with_prompts(clip_model, learnable_prompts_module, all_class_texts, DEVICE)
    else:
        tokenized_prompts = clip.tokenize(all_class_texts).to(DEVICE)
        all_text_features = clip_model.encode_text(tokenized_prompts)
        all_text_features /= all_text_features.norm(dim=-1, keepdim=True) # Normalize features

    # Separate features for live and spoof prompts
    live_text_features = all_text_features[0:num_prompts_per_class] # Shape: [num_live_prompts, embedding_dim]
    spoof_text_features = all_text_features[num_prompts_per_class:] # Shape: [num_spoof_prompts, embedding_dim]

for i, data in enumerate(tqdm(data_loader, desc="Processing batches")):
    rgb_images, _, _, labels = data

    rgb_images = rgb_images.to(DEVICE)
    labels = labels.to(DEVICE)

    pil_images = [ToPILImage()(img_tensor.cpu()) for img_tensor in rgb_images]
    processed_images_batch = torch.stack([clip_preprocess(pil_img) for pil_img in pil_images]).to(DEVICE)

    with torch.no_grad():
        # Encode images
        image_features = clip_model.encode_image(processed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize image features

        # --- Calculate Cosine Similarities ---
        sim_live = image_features @ live_text_features.T
        sim_live_mean = torch.mean(sim_live, dim=-1) # Shape: [batch_size]
        sim_live_max, _ = torch.max(sim_live, dim=-1) # Shape: [batch_size]

        sim_spoof_all = image_features @ spoof_text_features.T
        sim_spoof_max, _ = torch.max(sim_spoof_all, dim=-1) # Shape: [batch_size]
        k = 1
        top_k_values, _ = torch.topk(sim_spoof_all, k, dim=-1)  # Shape: [batch_size, k]
        sim_spoof_top_k_mean = torch.mean(top_k_values, dim=-1)  # Shape: [batch_size]

        # --- Scoring for Anti-Spoofing ---
        final_scores_batch = sim_live_mean - sim_spoof_top_k_mean

    score_list.extend(final_scores_batch.cpu().numpy())
    label_list.extend(labels.cpu().numpy()) # Collect ground truth labels

# make sure the score_list does not contain NaN values
for i in range(0, len(label_list)):
    Total_score_list_cs.append(score_list[i])
    if score_list[i] == None:
        logging.info(f"score_list[{i}] is None")
        exit(0)

fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, Total_score_list_cs)
threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

for i in range(len(Total_score_list_cs)):
    score = Total_score_list_cs[i]
    if (score >= threshold_cs and label_list[i] == 1):
        TP += 1
    elif (score < threshold_cs and label_list[i] == 0):
        TN += 1
    elif (score >= threshold_cs and label_list[i] == 0):
        FP += 1
    elif (score < threshold_cs and label_list[i] == 1):
        FN += 1

APCER = FP / (TN + FP)
NPCER = FN / (FN + TP)

step = 0
if record[1]>((APCER + NPCER) / 2):
    record[0]=step + 1
    record[1]=((APCER + NPCER) / 2)
    record[2]=roc_auc_score(label_list, score_list)
    record[3]=APCER
    record[4]=NPCER
    record[5]=calculate_tpr_at_fpr(label_list, normalize_data(score_list))

logging.info(f"model: CLIP + Protext")
logging.info(f"Train on {source}, Test on {target}, Batch Size = {str(batch_size_arg)}")
logging.info(f"ACER {record[1]:.6f} AUC {record[2]:.6f} APCER {record[3]:.6f} BPCER {record[4]:.6f}\n")