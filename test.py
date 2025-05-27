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

# python test.py --train_dataset N --test_dataset C --batch_size 75
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

# TCLIP EM Algorithm Hyperparameters
TCLIP_LAMBDA = 55  # Weight for cluster proportion entropy \sum \hat{u}_k ln(\hat{u}_k)
TCLIP_GAMMA = 0.5   # Weight for anchoring w_k to t_k
EM_ITERATIONS = 20  # Number of EM iterations
# Implicit variance for exp(-||z-w||^2 / (2*sigma^2)). If sigma=1, then /2.
# The objective has 1/2 ||z-w||^2, so the exp will be exp(-1/2 ||z-w||^2)
# This implies sigma^2 = 1.

# Define text prompts for live and spoof classes
LIVE_CLASS_TEXT = [
    "a photo of a real person",
    "a person looking directly at the camera",
    "a live human face under natural light",
    "a genuine portrait of someone",
    "a real face with natural expressions"
]
SPOOF_CLASS_TEXTS = [
    # print
    "a printed photograph of a face",
    "a paper photo held up to a camera",
    "a person's face printed on paper",
    "a photograph of a face being scanned",
    "a printed image of a face",
    # replay
    "a face displayed on a screen",
    "a digital screen showing a face",
    "a person's face on a smartphone screen",
    "a face playback on a tablet",
    "a face video playing on a monitor",
    # mask
    "a person wearing a 3D mask",
    "a silicone mask covering a face",
    "a lifelike mask of a human face",
    "a molded mask impersonating a face",
    "a realistic 3D face mask",
    # partial
    "a face partially covered",
    "only part of a face visible",
    "a cropped face showing eyes only",
    "a face with occluded regions",
    "a partially visible face",
    # mannequin
    "a mannequin head",
    "a plastic mannequin face",
    "a store display mannequin",
    "a dummy face on a stand",
    "a lifeless mannequin bust",
    # deepfake
    "an AI-generated fake face",
    "a deepfake video frame",
    "a synthetically altered face",
    "a computer-generated face swap",
    "an artificially manipulated face"
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


def get_image_features(clip_model, images_tensor, device):
    """Generates CLIP visual embeddings for a batch of images."""
    with torch.no_grad():
        image_features = clip_model.encode_image(images_tensor.to(device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.float() # Ensure float32 for distance calcs

# --- TCLIP EM Algorithm ---
def tclip_em_algorithm(Z_visual, T_text_anchors, lambda_reg, gamma_reg, iterations):
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
    N = Z_visual.shape[0]
    K = int(T_text_anchors.shape[0] / num_prompts_per_class)

    # Initialization
    W_means = torch.zeros(K, Z_visual.shape[1], device=DEVICE) # K x D
    for i in range(K):
        W_mean = T_text_anchors[i * num_prompts_per_class:(i + 1) * num_prompts_per_class].mean(dim=0, keepdim=True)
        W_means[i] = W_mean.squeeze(0) # K x D
    W_means_anchor = W_means.clone().detach().float() # Initialize means with text anchors
    #W_means = T_text_anchors.clone().detach().float() # Initialize means with text anchors
    U_hat_proportions = torch.ones(K, device=DEVICE) / K # Initialize proportions uniformly
    U_assignments = torch.zeros(N, K, device=DEVICE)

    for iteration in range(iterations):
        # --- E-step: Update U_assignments (u_nk) ---
        # u_nk = exp( (lambda/N) * log(u_hat_k_old) - 0.5 * ||z_n - w_k||^2 ) / sum_j(...)
        # Note: Distances are ||z_n - w_k||^2. The 0.5 factor is in the objective.
        # exp(-0.5 * ||z_n - w_k||^2) corresponds to a Gaussian with sigma=1.

        dist_sq = torch.cdist(Z_visual, W_means, p=2).pow(2) # N x K, squared Euclidean distances

        log_u_hat_safe = torch.log(U_hat_proportions + 1e-10) # Add epsilon for numerical stability

        # Exponent term for u_nk
        # The coefficient of u_nk ln(u_nk) in objective is +1.
        # This implies exp(- Dist_nk / Temperature), where Temperature=1 from that term.
        # The constant for ||z_n-w_k||^2 is 1/2.
        exponent = (lambda_reg / N) * log_u_hat_safe.unsqueeze(0) - 0.5 * dist_sq
        #exponent = lambda_reg * (1 + log_u_hat_safe.unsqueeze(0)) - 0.5 * dist_sq
        temparature = 0.8
        exponent = exponent / temparature # Apply temperature scaling

        U_assignments = F.softmax(exponent, dim=1)

        # --- M-step ---
        # Update U_hat_proportions (cluster proportions)
        U_hat_proportions = U_assignments.mean(dim=0) # Sum over N samples, then divide by N
        if iteration == 19: 
            print(f"    Proportions: {U_hat_proportions}")

        # Update W_means (cluster means)
        # w_k = (sum_n u_nk * z_n + gamma * t_k) / (sum_n u_nk + gamma)
        sum_u_nk = U_assignments.sum(dim=0) # K, sum of assignments for each cluster
        
        # Numerator: sum_n u_nk * z_n for each k
        # U_assignments is N x K, Z_visual is N x D
        # sum_u_nk_z_n will be K x D
        sum_u_nk_z_n = torch.matmul(U_assignments.T, Z_visual) # K x N @ N x D -> K x D

        numerator = sum_u_nk_z_n + gamma_reg * W_means_anchor
        denominator = sum_u_nk.unsqueeze(1) + gamma_reg # K x 1
        #numerator = sum_u_nk_z_n
        #denominator = sum_u_nk.unsqueeze(1)

        W_means = numerator / (denominator + 1e-10) # Add epsilon for stability

        # Calculate objective value (optional, for monitoring convergence)
        term1 = 0.5 * (U_assignments * dist_sq).sum()
        term2 = -lambda_reg * (U_hat_proportions * torch.log(U_hat_proportions + 1e-10)).sum()
        term3 = temparature * (U_assignments * torch.log(U_assignments + 1e-10)).sum()
        term4 = 0.5 * gamma_reg * ((W_means - W_means_anchor)**2).sum()
        objective = term1 + term2 + term3 + term4
        #print(f"    Objective: {objective.item():.4f}")
        print(f"EM Iteration {iteration:02d} [Objective: {objective.item():.4f} data-fitting: {term1.item():.4f}, complexity: {term2.item():.4f}, barrier: {term3.item():.4f}, anchor: {term4.item():.4f}]")


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
    TEXT_FEATURE_DIM = clip_model.text_projection.shape[1]
    # For LearnablePrompts, pass the nn.Embedding layer itself
    token_embedding_layer = clip_model.token_embedding
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit(0)

# 2. Load Learnable Prompts module
if not os.path.exists(LEARNED_PROMPTS_PATH):
    print(f"Error: Learned prompts file not found at {LEARNED_PROMPTS_PATH}")
    exit(0)

# Pass the actual token embedding layer (nn.Embedding) from CLIP
learnable_prompts_module = LearnablePrompts(
    prompt_length=PROMPT_LENGTH,
    embed_dim=TEXT_FEATURE_DIM, # This should be clip_model.token_embedding.embedding_dim
    clip_token_embedding=token_embedding_layer, # Pass the actual nn.Embedding layer
).to(DEVICE)

learnable_prompts_module.load_state_dict(torch.load(LEARNED_PROMPTS_PATH, map_location=DEVICE, weights_only=True))
learnable_prompts_module.eval()
print(f"Learned prompts loaded from {LEARNED_PROMPTS_PATH}")

# 3. Prepare Text Anchor Embeddings (T_k)
all_class_texts = LIVE_CLASS_TEXT + SPOOF_CLASS_TEXTS
print(f"\nGenerating text anchors")
T_text_anchors = get_text_features_with_prompts(clip_model, learnable_prompts_module, all_class_texts, DEVICE)
K_total_clusters = T_text_anchors.shape[0] / num_prompts_per_class
print(f"Generated {K_total_clusters} text anchors with shape {T_text_anchors.shape}")

# 4. Visual Embeddings (Z_n)
#for i, data in enumerate(tqdm(data_loader, desc="Processing batches")):
for i, data in enumerate(data_loader):
    #visual_embedding_dim = clip_model.visual.output_dim # Get from CLIP model spec
    # For ViT-B/32, image feature dim is 512, same as text after projection
    visual_embedding_dim = TEXT_FEATURE_DIM 
    
    # feature extraction
    rgb_img, depth_img, ir_img, labels = data
    to_pil = ToPILImage()
    rgb_pils = [to_pil(img) for img in rgb_img ]  
    processed = [clip_preprocess(pil) for pil in rgb_pils ]  # each is a C×H×W tensor
    rgb_batch = torch.stack(processed, dim=0).to(DEVICE)  # shape: [B, C, H, W]

    num_test_samples = rgb_img.shape[0]
    Z_visual_embeddings = get_image_features(clip_model, rgb_batch, DEVICE)
    print(f"Visual embeddings shape: {Z_visual_embeddings.shape}")

    # 5. Run TCLIP EM Algorithm
    print("\nStarting TCLIP EM Algorithm...")
    W_learned_means, U_final_assignments, U_hat_final_proportions = tclip_em_algorithm(
        Z_visual_embeddings,
        T_text_anchors,
        TCLIP_LAMBDA,
        TCLIP_GAMMA,
        EM_ITERATIONS
    )
    print("TCLIP EM Algorithm finished.")
    print(f"Learned cluster means W shape: {W_learned_means.shape}")
    print(f"Final assignments U shape: {U_final_assignments.shape}")
    print(f"Final proportions U_hat: {U_hat_final_proportions}")

    # 6. Classification
    # U_final_assignments has shape N x K_total_clusters
    # First cluster (index 0) is "live"
    # Remaining clusters (index 1 to K_total_clusters-1) are "spoof" types
    prob_live = U_final_assignments[:, 0]
    prob_spoof = U_final_assignments[:, 1:].sum(dim=1)

    # Predictions: 0 for live, 1 for spoof
    #predictions = (prob_spoof > prob_live).int().cpu().numpy()
    #predictions = (U_final_assignments[:, 0] - U_final_assignments[:, 1:].sum(dim=1)).cpu().numpy()
    predictions = (U_final_assignments[:, 0] - torch.max(U_final_assignments[:, 1:], dim=1).values).cpu().numpy()

    # 6.(alt) Classification by cosine similarity
    sim_live = (Z_visual_embeddings @ W_learned_means[:1].T).squeeze(1)
    sim_spoof_all = (Z_visual_embeddings @ W_learned_means[1:].T).squeeze(1)
    sim_spoof_max, _ = torch.max(sim_spoof_all, dim=-1)
    predictions = (sim_live - sim_spoof_max).cpu().numpy()
    
    # compare these `predictions` against ground truth labels
    ground_truth_labels = labels.cpu().numpy()
    for j in range(num_test_samples):
        score_list.append(predictions[j])
        label_list.append(ground_truth_labels[j])

# make sure the score_list does not contain NaN values
for i in range(0, len(label_list)):
    Total_score_list_cs.append(score_list[i])
    if score_list[i] == None:
        print(score_list[i])

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

logging.info(f"Train on {source}, Test on {target}, Batch Size = {str(batch_size_arg)}")
logging.info(f"ACER {record[1]:.6f} AUC {record[2]:.6f} APCER {record[3]:.6f} BPCER {record[4]:.6f}\n")