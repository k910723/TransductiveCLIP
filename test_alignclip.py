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

# python test_alignclip.py --train_dataset N --test_dataset Ca --batch_size 75
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
TCLIP_LAMBDA = 75    # Weight for cluster proportion entropy \sum \hat{u}_k ln(\hat{u}_k)
TCLIP_GAMMA = 10   # Weight for anchoring w_k to t_k
TCLIP_TEMPERATURE = 1  # Temperature for softmax
EM_ITERATIONS = 20  # Number of EM iterations

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
    '''with torch.no_grad():
        face_prompt = clip.tokenize(["a photo of a face"]).to(DEVICE)
        face_embedding = clip_model.encode_text(face_prompt, normalize=True).to(DEVICE)
    for k in range(W_means.shape[0]):
        W_means[k] = W_means[k] - face_embedding'''


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
@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

# Define the vision and text configurations
vision_cfg = CLIPVisionCfg(
    layers=12,
    width=768,
    patch_size=16,
    image_size=224
)
text_cfg = CLIPTextCfg(
    context_length=77,
    vocab_size=49408,
    width=768,
    heads=12,
    layers=12
)
# Define the embedding dimension
embed_dim = 768
try:
    clip_model = CLIP(
        embed_dim=embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        cast_dtype=torch.float16,  # Optional
    )
    checkpoint = torch.load('/shared/shared/kevin_huggingface/alignCLIP.pt', map_location=DEVICE, weights_only=False)
    clip_model.load_state_dict(checkpoint['state_dict'])  # Adjust key if different
    clip_model.eval()
    clip_model.to(DEVICE)
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit(0)

# 2. Load Learnable Prompts module
# skip this part for now

# 3. Prepare Text Anchor Embeddings (T_k)
with torch.no_grad():
    all_class_texts = LIVE_CLASS_TEXT + SPOOF_CLASS_TEXTS
    tokenized_texts = clip.tokenize(all_class_texts, context_length=77).to(DEVICE)
    T_text_anchors = clip_model.encode_text(tokenized_texts, normalize=True).to(DEVICE)
    K_total_clusters = T_text_anchors.shape[0] / num_prompts_per_class
    print(f"Generated {T_text_anchors.shape[0]} text anchors with {K_total_clusters} clusters")

for i, data in enumerate(tqdm(data_loader, desc="Processing batches")):
#for i, data in enumerate(data_loader):
    # 4. Visual Embeddings (Z_n)
    rgb_img, depth_img, ir_img, labels = data
    num_test_samples = rgb_img.shape[0]
    clip_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),         # Convert to tensor with shape (C, H, W)
        transforms.Normalize(          # Normalize to match CLIP's expected input distribution
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    # Apply preprocessing to a batch of images
    to_pil = ToPILImage()
    rgb_pils = [to_pil(img) for img in rgb_img ]  
    processed = [clip_preprocess(pil) for pil in rgb_pils ]  # each is a C×H×W tensor
    rgb_batch = torch.stack(processed, dim=0).to(DEVICE)  # shape: [B, C, H, W]

    with torch.no_grad():
        #print(f"Face visual embeddings shape: {face_visual_embeddings.shape}")
        Z_visual_embeddings = clip_model.encode_image(rgb_batch, normalize=True).to(DEVICE)
        #print(f"Visual embeddings shape: {Z_visual_embeddings.shape}")

    # 5. Run TCLIP EM Algorithm
    #print("\nStarting TCLIP EM Algorithm...")
    W_learned_means, U_final_assignments, U_hat_final_proportions = tclip_em_algorithm(
        Z_visual_embeddings,
        T_text_anchors,
        TCLIP_LAMBDA,
        TCLIP_TEMPERATURE,
        TCLIP_GAMMA,
        EM_ITERATIONS,
        num_test_samples=num_test_samples
    )
    #print("TCLIP EM Algorithm finished.")
    #print(f"Learned cluster means W shape: {W_learned_means.shape}")
    #print(f"Final assignments U shape: {U_final_assignments.shape}")
    #print(f"Final proportions U_hat: {U_hat_final_proportions}")

    # 6. Classification
    calc_sim = False
    if not calc_sim:
        # U_final_assignments has shape N x K_total_clusters
        prob_live = U_final_assignments[:num_test_samples, 0]
        prob_spoof = U_final_assignments[:num_test_samples, 1:].sum(dim=1)
        # Predictions: 0 for live, 1 for spoof
        #predictions = (prob_spoof > prob_live).int().cpu().numpy()
        #predictions = (U_final_assignments[:, 0] - U_final_assignments[:, 1:].sum(dim=1)).cpu().numpy()
        predictions = (U_final_assignments[:num_test_samples, 0] - torch.max(U_final_assignments[:num_test_samples, 1:], dim=1).values).cpu().numpy()
    else:
        # get the text embeddings of "a photo of a face"
        '''with torch.no_grad():
            face_prompt = clip.tokenize(["a photo of a face"]).to(DEVICE)
            face_embedding = clip_model.encode_text(face_prompt, normalize=True).to(DEVICE)
        for k in range(W_learned_means.shape[0]):
            W_learned_means[k] = W_learned_means[k] - face_embedding'''
        # Classification by cosine similarity
        sim_live = (Z_visual_embeddings @ W_learned_means[:1].T).squeeze(1)
        sim_spoof_all = (Z_visual_embeddings @ W_learned_means[1:].T).squeeze(1)
        sim_spoof_max, _ = torch.max(sim_spoof_all, dim=-1)
        sim_spoof_topk = torch.mean(torch.topk(sim_spoof_all, k=3, dim=-1).values, dim=-1)
        predictions = (sim_live - sim_spoof_topk).cpu().numpy()
    
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

logging.info(f"model: AlignCLIP + Transductive")
logging.info(f"Train on {source}, Test on {target}, Batch Size = {str(batch_size_arg)}")
logging.info(f"lambda = {TCLIP_LAMBDA}, temperature = {TCLIP_TEMPERATURE}, gamma = {TCLIP_GAMMA}")
logging.info(f"ACER {record[1]:.6f} AUC {record[2]:.6f} APCER {record[3]:.6f} BPCER {record[4]:.6f}\n")