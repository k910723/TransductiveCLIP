import torch
import torch.nn.functional as F
import numpy as np
import os
os.environ["HF_HOME"] = "/shared/shared/huggingface"
from dataloader_attack_type import FAS_Dataset_AttackTypes
from torch.utils.data import DataLoader as Dataloader
from torch.utils.data import ConcatDataset
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.prior_refinement import CLIPPriorRefinement
from custom_prompts import *
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Overlay patch-level similarity maps organized by ground-truth label")
parser.add_argument("--test_dataset", type=str, default="W", help="Target dataset protocol")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--num_channels", type=int, default=256, help="Number of channels for refinement")
parser.add_argument("--lambda_balance", type=float, default=0.7, help="Balance factor for refinement")
parser.add_argument("--aggregation_method", type=str, default="multi_class", choices=["class_prototypes", "multi_class"])
parser.add_argument("--max_batches", type=int, default=1, help="Maximum batches to process")
args = parser.parse_args()

# Configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
root = '/shared/shared'
overlay_alpha = 0.5
patch_size = 16

live_prompts = live_prompts_Ca
print_attack_prompts = print_attack_prompts_Ca
replay_attack_prompts = replay_attack_prompts_Ca

attack_types = {
    'live': live_prompts,
    'print': print_attack_prompts,
    'replay': replay_attack_prompts,
    'mask': mask_attack_prompts,
    'partial': partial_attack_prompts,
    'mannequin': mannequin_attack_prompts
}
ALL_PROMPTS = live_prompts + print_attack_prompts + replay_attack_prompts + mask_attack_prompts + partial_attack_prompts + mannequin_attack_prompts

# Label to attack type mapping (based on FAS_Dataset_AttackTypes)
label_to_attack_type = {
    0: 'live',
    1: 'print',
    2: 'replay',
    3: 'mask',
    4: 'partial',
    5: 'mannequin'
}

# Data loader
target_dataset = FAS_Dataset_AttackTypes(root=root, protocol=[args.test_dataset], train=False, size=224)
target_loader = Dataloader(target_dataset, batch_size=args.batch_size, shuffle=False)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()

# Initialize prior refinement
with torch.no_grad():
    dummy_inputs = clip_processor(text=["test"], return_tensors="pt", padding=True).to(DEVICE)
    dummy_features = clip_model.get_text_features(**dummy_inputs)
    original_dim = dummy_features.shape[1]

prior_refinement = CLIPPriorRefinement(
    #clip_model=clip_model,
    #clip_processor=clip_processor,
    num_channels=min(args.num_channels, original_dim),
    lambda_balance=args.lambda_balance,
    device=DEVICE
)
selected_channels = prior_refinement.initialize_with_attack_types(
    live_prompts=live_prompts,
    print_prompts=print_attack_prompts,
    replay_prompts=replay_attack_prompts,
    mask_prompts=mask_attack_prompts,
    partial_prompts=partial_attack_prompts,
    mannequin_prompts=mannequin_attack_prompts,
    aggregation_method=args.aggregation_method
)

# Prepare text anchor embeddings
with torch.no_grad():
    text_inputs = clip_processor(text=ALL_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    T_text_anchors = clip_model.get_text_features(**text_inputs)
    T_text_anchors = F.normalize(T_text_anchors, dim=-1)

# Create prompt indices
prompt_indices = {}
start_idx = 0
for attack_type, prompts in attack_types.items():
    prompt_indices[attack_type] = list(range(start_idx, start_idx + len(prompts)))
    start_idx += len(prompts)

# Function to overlay patch-level similarity maps (original implementation with label-based folders)
def save_averaged_similarity_overlays(
    rgb_img, visual_embeddings, T_text_anchors, attack_types, prompt_indices,
    selected_channels, labels, save_path, prefix, step_idx, device, overlay_alpha
):
    # Create base directory
    os.makedirs(save_path, exist_ok=True)
    B, _, H, W = rgb_img.shape
    num_patches = visual_embeddings.shape[1]
    patch_dim = int(np.sqrt(num_patches))

    visual_selected = visual_embeddings[:, :, selected_channels]
    visual_selected = F.normalize(visual_selected, dim=-1)
    T_text_selected = T_text_anchors[:, selected_channels]
    T_text_selected = F.normalize(T_text_selected, dim=-1)
    visual_non_selected = visual_embeddings[:, :, ~selected_channels]
    visual_non_selected = F.normalize(visual_non_selected, dim=-1)
    T_text_non_selected = T_text_anchors[:, ~selected_channels]
    T_text_non_selected = F.normalize(T_text_non_selected, dim=-1)

    for attack_type, indices in prompt_indices.items():
        selected_sim = torch.matmul(visual_selected, T_text_selected[indices].T)
        selected_sim = selected_sim.mean(dim=2)
        selected_sim = selected_sim - selected_sim.min(dim=1, keepdim=True)[0]
        selected_sim = selected_sim / (selected_sim.max(dim=1, keepdim=True)[0] + 1e-8)
        selected_sim = selected_sim.view(B, patch_dim, patch_dim)
        selected_heat = TF.resize(selected_sim, size=(H, W), interpolation=TF.InterpolationMode.BILINEAR)
        selected_heat = selected_heat.cpu().numpy()
        selected_heat = np.clip(selected_heat, 0, 1)

        non_selected_sim = torch.matmul(visual_non_selected, T_text_non_selected[indices].T)
        non_selected_sim = non_selected_sim.mean(dim=2)
        non_selected_sim = non_selected_sim - non_selected_sim.min(dim=1, keepdim=True)[0]
        non_selected_sim = non_selected_sim / (non_selected_sim.max(dim=1, keepdim=True)[0] + 1e-8)
        non_selected_sim = non_selected_sim.view(B, patch_dim, patch_dim)
        non_selected_heat = TF.resize(non_selected_sim, size=(H, W), interpolation=TF.InterpolationMode.BILINEAR)
        non_selected_heat = non_selected_heat.cpu().numpy()
        non_selected_heat = np.clip(non_selected_heat, 0, 1)

        for i in range(min(B, 10)):
            img = rgb_img[i].detach().cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            label = labels[i].item()
            label_str = label_to_attack_type[label]

            # Create subfolder for the ground-truth label
            label_save_path = os.path.join(save_path, label_str)
            os.makedirs(label_save_path, exist_ok=True)

            # Save selected channels similarity map
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.imshow(selected_heat[i], cmap='jet', alpha=overlay_alpha)
            plt.axis('off')
            # plt.title(f"Selected: {attack_type.capitalize()}\nLabel: {label_str.capitalize()}", fontsize=8, wrap=True)
            plt.tight_layout()
            filename = f"{prefix}_{step_idx+i}_{attack_type}_selected_label_{label_str}.png"
            plt.savefig(os.path.join(label_save_path, filename), dpi=100)
            plt.close()

            # Save non-selected channels similarity map
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.imshow(non_selected_heat[i], cmap='jet', alpha=overlay_alpha)
            plt.axis('off')
            # plt.title(f"Non-Selected: {attack_type.capitalize()}\nLabel: {label_str.capitalize()}", fontsize=8, wrap=True)
            plt.tight_layout()
            filename = f"{prefix}_{step_idx+i}_{attack_type}_non_selected_label_{label_str}.png"
            plt.savefig(os.path.join(label_save_path, filename), dpi=100)
            plt.close()

            # Save original image
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            filename = f"{prefix}_{step_idx+i}_original_label_{label_str}.png"
            # Check if it is already saved. If not, save it
            if not os.path.exists(os.path.join(label_save_path, filename)):
                plt.savefig(os.path.join(label_save_path, filename), dpi=100)
            plt.close()

# Process batches
batch_count = 0
for i, data_batch in enumerate(tqdm(target_loader, desc="Processing batches")):
    if batch_count >= args.max_batches:
        break
    rgb_images, labels = data_batch
    rgb_images = rgb_images.to(DEVICE)
    labels = labels.to(DEVICE)

    pil_images = [ToPILImage()(img_tensor.cpu()) for img_tensor in rgb_images]
    image_inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model.vision_model(pixel_values=image_inputs.pixel_values)
        visual_embeddings = outputs.last_hidden_state[:, 1:, :]
        visual_embeddings = clip_model.visual_projection(visual_embeddings)
        visual_embeddings = F.normalize(visual_embeddings, dim=-1)

    save_averaged_similarity_overlays(
        rgb_img=rgb_images,
        visual_embeddings=visual_embeddings,
        T_text_anchors=T_text_anchors,
        attack_types=attack_types,
        prompt_indices=prompt_indices,
        selected_channels=selected_channels,
        labels=labels,
        save_path=f"similarity_maps/{args.test_dataset.replace('/', '')}",
        prefix=f"batch{i}",
        step_idx=i * args.batch_size,
        device=DEVICE,
        overlay_alpha=overlay_alpha
    )

    batch_count += 1