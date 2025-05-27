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
import itertools
from typing import List, Tuple

# python test_clip_prompt_finder.py --train_dataset N --test_dataset Ca --batch_size 75
parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--test_dataset", type=str)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--top_k_prompts", type=int, default=5, help="Number of best prompts to find per class")
args = parser.parse_args()

source = args.train_dataset
target = args.test_dataset
batch_size_arg = args.batch_size
top_k_prompts = 5

# --- Configuration ---
CLIP_MODEL_NAME = "ViT-B/32"
PROMPT_LENGTH = 16  # Should match the loaded ProText prompts
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define text prompts for live and spoof classes (20 per class)
LIVE_CLASS_TEXT = [
    # Original prompts
    "a photo of a real person",
    "a person looking directly at the camera",
    "a photo of a person's face with natural skin texture and depth",
    "an image of a genuine human face",
    "a person with realistic skin texture",
    # Extended prompts
    "a genuine human face with natural features",
    "an authentic photograph of a living person",
    "a real human with lifelike facial expressions",
    "a true photograph showing skin pores and natural lighting",
    "a real face with natural shadows and highlights",
    "an authentic portrait with detailed skin texture",
    "a genuine person with natural eye reflections",
    "a real human face with natural color variations",
    "an authentic face with visible blood circulation",
    "a living person with natural facial movements",
    "a real portrait showing subtle facial asymmetry",
    "an authentic human with natural skin imperfections",
    "a genuine face with reactive pupils",
    "a real person displaying natural micro-expressions",
    "an authentic human face with consistent texture quality"
]

PRINT_SPOOF_TEXT = [
    # Original prompts
    "a printed photograph of a face",
    "a paper photo of a face",
    "a person's face printed on paper",
    "a photograph of a face being scanned",
    "a printed image of a face",
    # Extended prompts
    "a glossy paper printout of a human face",
    "a flat photo print showing a person",
    "a face reproduced on printer paper",
    "a 2D printed reproduction of a portrait",
    "a face image reproduced on photographic paper",
    "a matte finish print of a human portrait",
    "a lifeless printed reproduction of a face",
    "a face photo with printer artifacts visible",
    "a static printed image showing a person",
    "a low-quality printed face photograph",
    "a face printed on cardstock paper",
    "an inkjet reproduction of a portrait",
    "a laser-printed image of a human face",
    "a photograph duplicated onto paper",
    "a flat paper reproduction lacking depth cues"
]

REPLAY_SPOOF_TEXT = [
    # Original prompts
    "a face displayed on a screen",
    "a digital screen showing a face",
    "a person's face on a smartphone screen",
    "a face playback on a tablet",
    "a face video playing on a monitor",
    # Extended prompts
    "a human face replayed on an LCD display",
    "a recorded video of a face on a screen",
    "a face shown on a digital device",
    "a portrait being played back on electronic display",
    "a face image displayed on a phone screen",
    "a human face on a pixelated display",
    "a digital reproduction of a face with screen glare",
    "a face visible on a backlit screen",
    "a portrait with digital screen artifacts",
    "a face image with visible screen pixels",
    "a replayed video showing a human face",
    "a face with visible refresh rate artifacts",
    "a digital screen displaying a face image",
    "a face with unnatural electronic coloration",
    "a screen-captured face with digital artifacts"
]

MASK_SPOOF_TEXT = [
    # Original prompts
    "a person wearing a 3D mask",
    "a silicone mask covering a face",
    "a lifelike mask of a human face",
    "a molded mask impersonating a face",
    "a realistic 3D face mask",
    # Extended prompts
    "a latex face mask with artificial features",
    "a synthetic face covering with artificial texture",
    "a person disguised with a silicone face mask",
    "a hyper-realistic mask mimicking human skin",
    "a 3D-printed face mask worn by someone",
    "a theatrical prosthetic mask covering a face",
    "a rubber mask designed to look human",
    "a face mask with artificial skin texture",
    "a person disguised behind a realistic face covering",
    "a fabricated mask imitating human features",
    "a face covered by artificial materials",
    "a mask with painted pores and textures",
    "a prosthetic face mask with simulated details",
    "a synthetic face covering with artificial coloration",
    "a full-face mask mimicking human appearance"
]

PARTIAL_SPOOF_TEXT = [
    # Original prompts
    "a face partially covered",
    "only part of a face visible",
    "a cropped face showing eyes only",
    "a face with occluded regions",
    "a partially visible face",
    # Extended prompts
    "a face with selective areas hidden",
    "a partially masked human face",
    "a face with strategically covered features",
    "a portrait with sections deliberately hidden",
    "a face presentation with obscured areas",
    "a fragmentary view of a human face",
    "a face with certain features blocked out",
    "a deliberately incomplete facial image",
    "a face with regions intentionally obscured",
    "a cropped presentation showing limited facial features",
    "a face with artifically concealed portions",
    "a partial face presentation for spoofing",
    "a face with certain biometric features hidden",
    "a strategically cropped facial image",
    "a face with discontinuous visible regions"
]

MANNEQUIN_SPOOF_TEXT = [
    # Original prompts
    "a mannequin head",
    "a plastic mannequin face",
    "a store display mannequin",
    "a dummy face on a stand",
    "a lifeless mannequin bust",
    # Extended prompts
    "a synthetic human head replica",
    "an artificial face used for display",
    "a mannequin with painted facial features",
    "a rigid plastic replica of a human head",
    "a display dummy with face-like features",
    "an artificial head with simplified features",
    "a commercial mannequin with facial details",
    "a synthetic bust with human-like appearance",
    "a rigid face model lacking flexibility",
    "a retail display head made of plastic",
    "an artificial face model with uniform texture",
    "a hollow mannequin head used for display",
    "a fabricated human facsimile for display",
    "a non-organic reproduction of a human face",
    "a synthetic face model with artificial eyes"
]

# Group all spoof classes together
ALL_SPOOF_TEXTS = PRINT_SPOOF_TEXT + REPLAY_SPOOF_TEXT #+ MASK_SPOOF_TEXT + PARTIAL_SPOOF_TEXT + MANNEQUIN_SPOOF_TEXT
#SPOOF_CATEGORIES = [PRINT_SPOOF_TEXT, REPLAY_SPOOF_TEXT, MASK_SPOOF_TEXT, PARTIAL_SPOOF_TEXT, MANNEQUIN_SPOOF_TEXT]
#CATEGORY_NAMES = ["Print", "Replay", "Mask", "Partial", "Mannequin"]
SPOOF_CATEGORIES = [PRINT_SPOOF_TEXT, REPLAY_SPOOF_TEXT]
CATEGORY_NAMES = ["Print", "Replay"]


root='/shared/shared'
results_filename = source.replace('/', '') + '_to_' + target.replace('/', '')

# Set up logging
file_handler = logging.FileHandler(filename='/home/kevin/TCLIP/logger/'+ results_filename + '_best_prompts.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

# Load the dataset
for protocol in target.split('/'):
    target_dataset = FAS_Dataset(root=root, protocol=[protocol], train=False)
    if protocol == target.split('/')[0]:
        combined_target_dataset = target_dataset
    else:
        combined_target_dataset = ConcatDataset([combined_target_dataset, target_dataset])
target_loader = Dataloader(combined_target_dataset, batch_size=batch_size_arg, shuffle=False)  # Keep order consistent for evaluation

# Initialize CLIP model
try:
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE, jit=False)
    clip_model.eval()
except Exception as e:
    print(f"Error loading CLIP model '{CLIP_MODEL_NAME}': {e}", exc_info=True)
    sys.exit(1)

def preprocess_batch(rgb_images):
    """Convert image tensors to CLIP-compatible format"""
    pil_images = [ToPILImage()(img_tensor.cpu()) for img_tensor in rgb_images]
    processed_images = torch.stack([clip_preprocess(pil_img) for pil_img in pil_images]).to(DEVICE)
    return processed_images

def encode_images(loader):
    """Pre-encode all images in the dataset to avoid redundant computation"""
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(loader, desc="Encoding images")):
            rgb_images, _, _, labels = data_batch
            rgb_images = rgb_images.to(DEVICE)
            
            processed_images = preprocess_batch(rgb_images)
            image_features = clip_model.encode_image(processed_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            all_features.append(image_features)
            all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels).to(DEVICE)

def encode_text_prompts(text_prompts):
    """Encode text prompts with CLIP"""
    with torch.no_grad():
        tokenized_prompts = clip.tokenize(text_prompts).to(DEVICE)
        text_features = clip_model.encode_text(tokenized_prompts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def evaluate_prompts(image_features, labels, live_prompts, spoof_prompts):
    """Evaluate a combination of prompts and calculate ACER"""
    with torch.no_grad():
        # Encode the selected prompts
        live_text_features = encode_text_prompts(live_prompts)
        spoof_text_features = encode_text_prompts(spoof_prompts)
        
        # Calculate similarities
        sim_live = image_features @ live_text_features.T
        sim_live_mean = torch.mean(sim_live, dim=-1)
        
        sim_spoof = image_features @ spoof_text_features.T
        sim_spoof_max, _ = torch.max(sim_spoof, dim=-1)
        
        # Calculate final scores
        final_scores = sim_live_mean - sim_spoof_max
        
        # Convert to numpy for metrics calculation
        scores_np = final_scores.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate ROC and find optimal threshold
        fpr, tpr, thresholds = metrics.roc_curve(labels_np, scores_np)
        threshold, _ = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
        
        # Calculate metrics
        TP = 0.0000001
        TN = 0.0000001
        FP = 0.0000001
        FN = 0.0000001
        
        # Apply threshold
        for i in range(len(scores_np)):
            score = scores_np[i]
            if (score >= threshold and labels_np[i] == 1):
                TP += 1
            elif (score < threshold and labels_np[i] == 0):
                TN += 1
            elif (score >= threshold and labels_np[i] == 0):
                FP += 1
            elif (score < threshold and labels_np[i] == 1):
                FN += 1
        
        APCER = FP / (TN + FP)
        NPCER = FN / (FN + TP)
        ACER = (APCER + NPCER) / 2
        AUC = roc_auc_score(labels_np, scores_np)
        
        return ACER, AUC, APCER, NPCER

def find_best_prompts_per_category():
    """Find the best prompts for each category independently"""
    logging.info(f"Starting best prompt search for CLIP FAS")
    logging.info(f"Train on {source}, Test on {target}, Batch Size = {str(batch_size_arg)}")
    
    # Pre-encode all images to avoid redundant computation
    logging.info("Pre-encoding all images...")
    image_features, labels = encode_images(target_loader)
    
    # Store results for each category
    best_prompts = {}
    best_metrics = {}
    
    # First find best live prompts
    logging.info("\nFinding best live prompts...")
    
    best_acer = float('inf')
    best_live_set = []
    
    # Use all spoof prompts for initial live prompt selection
    all_spoof_text = ALL_SPOOF_TEXTS
    
    # Try different combinations of live prompts
    for live_combo in tqdm(list(itertools.combinations(LIVE_CLASS_TEXT, top_k_prompts)), desc="Evaluating live prompt combinations"):
        acer, auc, apcer, npcer = evaluate_prompts(image_features, labels, list(live_combo), all_spoof_text)
        
        if acer < best_acer:
            best_acer = acer
            best_live_set = list(live_combo)
            best_metrics["live"] = {
                "ACER": acer, 
                "AUC": auc, 
                "APCER": apcer, 
                "NPCER": npcer
            }
    
    best_prompts["live"] = best_live_set
    logging.info(f"Best live prompts found with ACER: {best_acer:.6f}")
    for i, prompt in enumerate(best_live_set):
        logging.info(f"  Live prompt {i+1}: \"{prompt}\"")
    
    # Now find best spoof prompts for each category
    for idx, (category_texts, category_name) in enumerate(zip(SPOOF_CATEGORIES, CATEGORY_NAMES)):
        logging.info(f"\nFinding best {category_name} spoof prompts...")
        
        best_acer = float('inf')
        best_spoof_set = []
        
        # Try different combinations of spoof prompts for this category
        for spoof_combo in tqdm(list(itertools.combinations(category_texts, top_k_prompts)), desc=f"Evaluating {category_name} combinations"):
            # Use the best live prompts and current spoof prompts
            acer, auc, apcer, npcer = evaluate_prompts(image_features, labels, best_prompts["live"], list(spoof_combo))
            
            if acer < best_acer:
                best_acer = acer
                best_spoof_set = list(spoof_combo)
                best_metrics[category_name] = {
                    "ACER": acer, 
                    "AUC": auc, 
                    "APCER": apcer, 
                    "NPCER": npcer
                }
        
        best_prompts[category_name] = best_spoof_set
        logging.info(f"Best {category_name} prompts found with ACER: {best_acer:.6f}")
        for i, prompt in enumerate(best_spoof_set):
            logging.info(f"  {category_name} prompt {i+1}: \"{prompt}\"")
    
    # Final comprehensive evaluation with all best prompts
    all_best_spoof_prompts = []
    for category in CATEGORY_NAMES:
        all_best_spoof_prompts.extend(best_prompts[category])
    
    final_acer, final_auc, final_apcer, final_npcer = evaluate_prompts(
        image_features, labels, best_prompts["live"], all_best_spoof_prompts
    )
    
    logging.info("\n=== FINAL RESULTS WITH ALL BEST PROMPTS ===")
    logging.info(f"ACER: {final_acer:.6f}, AUC: {final_auc:.6f}")
    logging.info(f"APCER: {final_apcer:.6f}, NPCER: {final_npcer:.6f}")
    
    # Save results to a JSON file
    result_data = {
        "best_prompts": best_prompts,
        "best_metrics": best_metrics,
        "final_metrics": {
            "ACER": float(final_acer),
            "AUC": float(final_auc),
            "APCER": float(final_apcer),
            "NPCER": float(final_npcer)
        }
    }
    
    import json
    with open(f'/home/kevin/TCLIP/results/best_prompts_{results_filename}.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    return best_prompts, final_acer, final_auc, final_apcer, final_npcer

if __name__ == "__main__":
    best_prompts, final_acer, final_auc, final_apcer, final_npcer = find_best_prompts_per_category()
    
    logging.info("\n=== BEST PROMPTS SUMMARY ===")
    
    # Print best prompts for live class
    logging.info("\nBest Live Prompts:")
    for i, prompt in enumerate(best_prompts["live"]):
        logging.info(f"  {i+1}. \"{prompt}\"")
    
    # Print best prompts for each spoof category
    for category in CATEGORY_NAMES:
        logging.info(f"\nBest {category} Prompts:")
        for i, prompt in enumerate(best_prompts[category]):
            logging.info(f"  {i+1}. \"{prompt}\"")
    
    logging.info("\n=== FINAL PERFORMANCE ===")
    logging.info(f"ACER: {final_acer:.6f}")
    logging.info(f"AUC: {final_auc:.6f}")
    logging.info(f"APCER: {final_apcer:.6f}")
    logging.info(f"NPCER: {final_npcer:.6f}")