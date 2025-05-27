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
import matplotlib.pyplot as plt
from model.LearnablePrompts import LearnablePrompts
from torchvision.transforms import ToPILImage
from model.prior_refinement import CLIPPriorRefinement
from custom_prompts import return_live_prompts, return_print_replay, return_print_attack_prompts, return_replay_attack_prompts

# python test_clip_nips24.py --test_dataset Ca
parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str, default="N",)
parser.add_argument("--test_dataset", type=str)
parser.add_argument("--batch_size", type=int, default=75)

parser.add_argument("--num_channels", type=int, default=384) # Number of channels to select for refinement
parser.add_argument("--lambda_balance", type=float, default=0.7) # Balance factor for similarity vs variance
parser.add_argument("--aggregation_method", type=str, default="multi_class", choices=["class_prototypes", "multi_class"]) # How to handle multiple prompts per class
args = parser.parse_args()

source = args.train_dataset
target = args.test_dataset
batch_size_arg = args.batch_size

use_protext = False
use_refinement = True
draw_visualization = False

# --- Configuration ---
CLIP_MODEL_NAME = "ViT-B/32"
PROMPT_LENGTH = 16  # Should match the loaded ProText prompts
LEARNED_PROMPTS_PATH = "results/protext_learned_prompts.pth" # Path to your saved ProText prompts
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define text prompts for live and spoof classes
# remember to change the prompts passed to prior refinement module if used attack types are changed
LIVE_CLASS_TEXT = return_live_prompts(target)
SPOOF_CLASS_TEXTS = return_print_replay(target)
num_prompts_per_class = 5

# Set up logging
root='/shared/shared'
results_filename = source.replace('/', '') + '_to_' + target.replace('/', '')
file_handler = logging.FileHandler(filename='logger/'+ results_filename + '_test.log')
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
    learnable_prompts_module.eval()
    all_text_features = []

    token_embedding_layer = clip_model.token_embedding
    positional_embedding = clip_model.positional_embedding
    transformer = clip_model.transformer
    ln_final = clip_model.ln_final
    text_projection = clip_model.text_projection
    dtype = clip_model.dtype
    context_length = clip_model.context_length

    original_prompts = learnable_prompts_module.prompts.data.clone()

    with torch.no_grad():
        for text_input_batch in [text_prompts_list]:
            tokenized_texts = clip.tokenize(text_input_batch, context_length=context_length).to(device)
            prompted_embeddings, _ = learnable_prompts_module(tokenized_texts, text_input_batch)
            max_seq_len = positional_embedding.shape[0]
            prompted_embeddings_truncated = prompted_embeddings[:, :max_seq_len, :]
            x = prompted_embeddings_truncated + positional_embedding.type(dtype)[:max_seq_len]
            x = x.permute(1, 0, 2)
            x = x.type(dtype)
            x = transformer(x)
            x = x.permute(1, 0, 2)
            x = ln_final(x).type(dtype)

            eot_indices_original = tokenized_texts.argmax(dim=-1)
            eot_indices_prompted = torch.zeros_like(eot_indices_original)
            for i in range(tokenized_texts.shape[0]):
                actual_text_len = (tokenized_texts[i] != 0).sum().item()
                eot_idx_in_original_token_seq = eot_indices_original[i]
                eot_indices_prompted[i] = PROMPT_LENGTH + eot_idx_in_original_token_seq
                if eot_indices_prompted[i] >= max_seq_len:
                    eot_indices_prompted[i] = max_seq_len - 1
                    print(f"Warning: EOT index {eot_indices_prompted[i]} out of bounds for batch {i}. Clamping to {max_seq_len - 1}.")

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
    return image_features.float()

# Function for EM Algorithm
def compute_affinity_matrix(image_embeddings, top_k=3):
    """
    Compute affinity matrix w_{i,j} = f_i^T f_j, truncated to top-k values.
    
    Args:
        image_embeddings (Tensor): N x D tensor of image embeddings.
        top_k (int): Number of top values to keep per row.
    
    Returns:
        Tensor: N x N affinity matrix.
    """
    affinity = torch.matmul(image_embeddings, image_embeddings.T)
    topk_values, _ = torch.topk(affinity, k=top_k, dim=1)
    threshold = topk_values[:, -1].unsqueeze(1)
    affinity = torch.where(affinity >= threshold, affinity, torch.zeros_like(affinity))
    return affinity

def compute_initial_predictions(image_embeddings, text_embeddings, tau):
    """
    Compute initial predictions hat_Y_i = softmax(tau * f_i^T t).
    
    Args:
        image_embeddings (Tensor): N x D tensor of image embeddings.
        text_embeddings (Tensor): K x D tensor of text embeddings.
        tau (float): Temperature parameter.
    
    Returns:
        Tensor: N x K initial predictions.
    """
    similarities = torch.matmul(image_embeddings, text_embeddings.T) * tau
    return F.softmax(similarities, dim=1)

def initialize_centroids(image_embeddings, hat_Y):
    """
    Initialize centroids mu_k using the image with highest hat_Y_{i,k}.
    
    Args:
        image_embeddings (Tensor): N x D tensor of image embeddings.
        hat_Y (Tensor): N x K initial predictions.
    
    Returns:
        Tensor: K x D centroids.
    """
    K = hat_Y.shape[1]
    _, top_indices = torch.topk(hat_Y, k=1, dim=0)
    return image_embeddings[top_indices.squeeze()]

def compute_p_ik(image_embeddings, mu, sigma):
    """
    Compute p_{i,k} proportional to Gaussian density.
    
    Args:
        image_embeddings (Tensor): N x D tensor of image embeddings.
        mu (Tensor): K x D centroids.
        sigma (Tensor): D-dimensional diagonal covariance.
    
    Returns:
        Tensor: N x K probabilities.
    """
    diff = image_embeddings.unsqueeze(1) - mu.unsqueeze(0)  # N x K x D
    sigma_inv = 1 / sigma
    exponent = -0.5 * torch.sum((diff ** 2) * sigma_inv, dim=2)  # N x K
    det_sigma = torch.prod(sigma)
    return torch.exp(exponent) / (det_sigma ** 0.5 + 1e-10)

def transclip(image_embeddings, text_embeddings, tau, gamma=0.0, max_iterations=100):
    """
    Implements TransCLIP algorithm in a zero-shot setting.
    
    Args:
        image_embeddings (Tensor): N x D tensor of image embeddings.
        text_embeddings (Tensor): K x D tensor of text embeddings.
        tau (float): Temperature parameter.
        gamma (float): Parameter for covariance update.
        max_iterations (int): Maximum number of outer iterations.
    
    Returns:
        Tensor: N-dimensional tensor of final class assignments.
    """
    N, D = image_embeddings.shape
    K = text_embeddings.shape[0]

    num_groups = K // 5
    selected_indices = []
    for group in range(num_groups):
        start = group * 5
        end = start + 5
        idx = random.randint(start, end - 1)
        selected_indices.append(start)
    text_embeddings = text_embeddings[selected_indices]
    K = text_embeddings.shape[0]

    # Step 1: Compute affinity matrix
    w = compute_affinity_matrix(image_embeddings)
    
    # Step 2: Compute initial predictions
    hat_Y = compute_initial_predictions(image_embeddings, text_embeddings, tau)
    
    # Step 3: Initialize centroids
    mu = initialize_centroids(image_embeddings, hat_Y)
    
    # Step 4: Initialize covariance
    sigma = torch.ones(D, device=image_embeddings.device) / D
    
    # Step 5: Initialize assignments
    z = hat_Y.clone()
    
    # Step 6-12: Block-wise updates
    iteration = 0
    for _ in range(max_iterations):
        iteration += 1
        z_prev = z.clone()
        
        # Z-update loop with convergence condition
        while True:
            z_old = z.clone()
            p_ik = compute_p_ik(image_embeddings, mu, sigma)
            log_p_ik = torch.log(p_ik + 1e-10)
            sum_w_z = torch.matmul(w, z)
            lambda_ = 1.0
            numerator = (hat_Y ** lambda_) * torch.exp(log_p_ik + sum_w_z)
            z = numerator / (numerator.sum(dim=1, keepdim=True) + 1e-10)

            change = torch.norm(z - z_old, p=1) / N
            if change < 1e-4:  # Convergence condition: 10^-4
                break
        
        # Check outer loop convergence
        if torch.norm(z - z_prev, p=1) / N < 1e-4:
            break
        
        # Step 10: Update centroids (zero-shot: only Q)
        sum_z_Q = z.sum(dim=0)
        weighted_sum_Q = torch.matmul(z.T, image_embeddings)
        mu = weighted_sum_Q / (sum_z_Q.unsqueeze(1) + 1e-10)
        
        # Step 11: Update covariance (zero-shot: only Q)
        diff = image_embeddings.unsqueeze(1) - mu.unsqueeze(0)
        weighted_diff = z.unsqueeze(2) * (diff ** 2)
        sigma = weighted_diff.sum(dim=0).sum(dim=0) / (gamma + 1)
    
    #print(z)
    #print(f"EM iterations finished in {iteration}")
    # Step 13: Final prediction
    return mu, z #torch.argmax(z, dim=1)

record = [1, 100, 100, 100, 100, 100]
log_list = []

score_list = []
score_list_baseline = []
Total_score_list = []
label_list = []
TP = 0.0000001
TN = 0.0000001
FP = 0.0000001
FN = 0.0000001

# 1. Load CLIP Model
try:
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE, jit=False)
    TEXT_FEATURE_DIM = clip_model.text_projection.shape[1]
    token_embedding_layer = clip_model.token_embedding
    clip_logit_scale = clip_model.logit_scale.exp()
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit(0)

# 2. Load prior refinement module
prior_refinement = None
if use_refinement:
    original_dim = clip_model.text_projection.shape[1]
    prior_refinement = CLIPPriorRefinement(
        num_channels=min(args.num_channels, original_dim),
        lambda_balance=args.lambda_balance,
        device=DEVICE
    )
    selected_channels = prior_refinement.initialize_with_attack_types(
        live_prompts=return_live_prompts(target),
        print_prompts=return_print_attack_prompts(target),
        replay_prompts=return_replay_attack_prompts(target),
        #mask_prompts=mask_attack_prompts,
        #partial_prompts=partial_attack_prompts,
        #mannequin_prompts=mannequin_attack_prompts,
        aggregation_method=args.aggregation_method
    )
    

# 3. Load Learnable Prompts (ProText) module
if not os.path.exists(LEARNED_PROMPTS_PATH):
    print(f"Error: Learned prompts file not found at {LEARNED_PROMPTS_PATH}")
    exit(0)

learnable_prompts_module = LearnablePrompts(prompt_length=PROMPT_LENGTH, embed_dim=TEXT_FEATURE_DIM, clip_token_embedding=token_embedding_layer).to(DEVICE)
learnable_prompts_module.load_state_dict(torch.load(LEARNED_PROMPTS_PATH, map_location=DEVICE, weights_only=True))
learnable_prompts_module.eval()
print(f"Learned prompts loaded from {LEARNED_PROMPTS_PATH}")

# 4. Prepare Text Embeddings
all_class_texts = LIVE_CLASS_TEXT + SPOOF_CLASS_TEXTS
if use_protext:
    T_text_anchors = get_text_features_with_prompts(clip_model, learnable_prompts_module, all_class_texts, DEVICE)
    K_total_clusters = len(all_class_texts)  # Each prompt is a cluster
    print(f"Generated {K_total_clusters} text anchors with shape :{T_text_anchors.shape}")
else:
    tokenized_prompts = clip.tokenize(all_class_texts).to(DEVICE)
    T_text_anchors = clip_model.encode_text(tokenized_prompts)
    T_text_anchors /= T_text_anchors.norm(dim=-1, keepdim=True)

if use_refinement:
    T_text_anchors = prior_refinement(T_text_anchors)
    T_text_anchors /= T_text_anchors.norm(dim=-1, keepdim=True)

# variables to collect misclassified samples across all batches
os.makedirs('misclassification_analysis', exist_ok=True)
misclassified_images = []
misclassified_clusters = []
misclassified_confidences = []
misclassified_true_labels = []
misclassified_predictions = []

# 5. Image Embeddings and EM Loop
with torch.no_grad():
    for i, data in enumerate(data_loader):
        rgb_img, depth_img, ir_img, labels = data
        to_pil = ToPILImage()
        rgb_pils = [to_pil(img) for img in rgb_img]
        processed = [clip_preprocess(pil) for pil in rgb_pils]
        rgb_batch = torch.stack(processed, dim=0).to(DEVICE)

        num_test_samples = rgb_img.shape[0]
        Z_visual_embeddings = get_image_features(clip_model, rgb_batch, DEVICE)

        if use_refinement:
            Z_visual_embeddings = prior_refinement(Z_visual_embeddings)
            Z_visual_embeddings /= Z_visual_embeddings.norm(dim=-1, keepdim=True)

        Z_visual_embeddings = Z_visual_embeddings.float()
        T_text_anchors = T_text_anchors.float()

        # 6. Run TCLIP EM Algorithm
        #print("\nStarting TCLIP EM Algorithm...")
        learned_means, U_final_assignments = transclip(Z_visual_embeddings, T_text_anchors, tau=clip_logit_scale)
        #print("TCLIP EM Algorithm finished.")
        #print(f"Learned cluster means W shape: {W_learned_means.shape}")

        # 7. Classification
        prob_live = U_final_assignments[:, 0]
        prob_spoof = U_final_assignments[:, 1:].max(dim=1)[0]
        predictions = (prob_live - prob_spoof).cpu().numpy()

        # 8. Collect Scores and Labels
        ground_truth_labels = labels.cpu().numpy()
        for j in range(num_test_samples):
            score_list.append(predictions[j])
            label_list.append(ground_truth_labels[j])

        # 9. collect misclassified samples (without optimal threshold) for visualization
        cluster_assignments = U_final_assignments.argmax(dim=1).cpu().numpy()
        confidences = U_final_assignments.max(dim=1)[0].cpu().numpy()

        # Determine correct vs incorrect assignments
        predicted_labels = (cluster_assignments == 0).astype(int)  # 1 for cluster 0, 0 for cluster 1+
        correct_assignments = (predicted_labels == ground_truth_labels)

        # Collect misclassified samples
        for j in range(num_test_samples):
            if not correct_assignments[j]:
                misclassified_images.append(rgb_pils[j])  # Store PIL image
                misclassified_clusters.append(cluster_assignments[j])
                misclassified_confidences.append(confidences[j])
                misclassified_true_labels.append(ground_truth_labels[j])
                misclassified_predictions.append(predicted_labels[j])

        print(f"Batch {i} - Misclassified: {np.sum(~correct_assignments)}/{num_test_samples}")

# 10. Evaluation
for i in range(0, len(label_list)):
    Total_score_list.append(score_list[i])
    if score_list[i] is None:
        print(score_list[i])

fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, Total_score_list)
threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

for i in range(len(Total_score_list)):
    score = Total_score_list[i]
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
if record[1] > ((APCER + NPCER) / 2):
    record[0] = step + 1
    record[1] = ((APCER + NPCER) / 2)
    record[2] = roc_auc_score(label_list, score_list)
    record[3] = APCER
    record[4] = NPCER
    record[5] = calculate_tpr_at_fpr(label_list, normalize_data(score_list))

logging.info(f"model: CLIP + nips24EM {'+ Protext ' if use_protext else ''}{'+ Prior Refinement' if use_refinement else ''}")
logging.info(f"Train on {source}, Test on {target}, Batch Size = {str(batch_size_arg)}, Channel number = {args.num_channels}")
logging.info(f"ACER {record[1]:.6f} AUC {record[2]:.6f} APCER {record[3]:.6f} BPCER {record[4]:.6f}\n")

# 11. Visualization
if misclassified_images and draw_visualization:
    # Create a grid to display misclassified images
    n_samples = min(len(misclassified_images), 20)  # Show max 20 samples
    cols = 5
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_samples):
        row = idx // cols
        col = idx % cols
        
        ax = axes[row, col]
        ax.imshow(misclassified_images[idx])
        
        true_label = "Live" if misclassified_true_labels[idx] == 1 else "Spoof"
        pred_label = "Live" if misclassified_predictions[idx] == 1 else "Spoof"
        confidence = misclassified_confidences[idx]
        cluster = misclassified_clusters[idx]
        
        title = f"True: {true_label}\nPred: {pred_label}\nCluster: {cluster}\nConf: {confidence:.3f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassification_analysis/misclassified_samples.png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Distribution of misclassified samples by true label
    true_labels_count = np.bincount(misclassified_true_labels, minlength=2)
    ax1.bar(['Live (0)', 'Spoof (1)'], true_labels_count, color=['green', 'red'], alpha=0.7)
    ax1.set_title('Misclassified Samples by True Label')
    ax1.set_ylabel('Count')
    for i, count in enumerate(true_labels_count):
        ax1.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    # Plot 2: Distribution by assigned cluster
    cluster_counts = np.bincount(misclassified_clusters)
    cluster_labels = [f'Cluster {i}' for i in range(len(cluster_counts))]
    colors = ['green' if i == 0 else 'red' for i in range(len(cluster_counts))]
    ax2.bar(cluster_labels, cluster_counts, color=colors, alpha=0.7)
    ax2.set_title('Misclassified Samples by Assigned Cluster')
    ax2.set_ylabel('Count')
    for i, count in enumerate(cluster_counts):
        ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    # Plot 3: Confidence distribution for misclassified samples
    ax3.hist(misclassified_confidences, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_title('Assignment Confidence Distribution\n(Misclassified Samples)')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Count')
    ax3.axvline(np.mean(misclassified_confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(misclassified_confidences):.3f}')
    ax3.legend()
    
    # Plot 4: Confusion matrix style visualization
    confusion_data = np.zeros((2, 2))
    for true_label, pred_label in zip(misclassified_true_labels, misclassified_predictions):
        confusion_data[true_label, pred_label] += 1
    
    im = ax4.imshow(confusion_data, cmap='Reds', alpha=0.8)
    ax4.set_title('Misclassification Pattern')
    ax4.set_xlabel('Predicted Label')
    ax4.set_ylabel('True Label')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Live', 'Spoof'])
    ax4.set_yticklabels(['Live', 'Spoof'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax4.text(j, i, int(confusion_data[i, j]), 
                           ha="center", va="center", color="black", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('misclassification_analysis/misclassification_analysis.png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nMisclassification Analysis:")
    print(f"Total misclassified samples: {len(misclassified_images)}")
    print(f"Average confidence of misclassified: {np.mean(misclassified_confidences):.3f}")
    print(f"Visualizations saved to misclassification_analysis/")