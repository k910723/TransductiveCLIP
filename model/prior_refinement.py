import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import CLIPModel, CLIPProcessor

# Defines a module for refining CLIP features for multi-class attack detection
class CLIPPriorRefinement(nn.Module):
    def __init__(
        self, 
        num_channels: int,  # Number of feature channels to select
        lambda_balance: float = 0.5,  # Balances similarity and variance in channel selection
        device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device to run computations
    ):
        super(CLIPPriorRefinement, self).__init__()
        CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.num_channels = num_channels
        self.lambda_balance = lambda_balance
        self.device = device
        self.selected_channels = None  # Mask for selected feature channels
        
    # Computes similarity between classes for each feature channel
    def compute_inter_class_similarity(self, features: torch.Tensor) -> torch.Tensor:
        C, D = features.shape  # C: num classes, D: feature dimensions
        similarity_scores = torch.zeros(D, device=self.device)
        
        for k in range(D):
            channel_features = features[:, k]
            similarity_sum = 0
            # Sum pairwise similarities for all class pairs
            for i in range(C):
                for j in range(C):
                    if i != j:
                        similarity_sum += channel_features[i] * channel_features[j]
            similarity_scores[k] = similarity_sum / (C * (C - 1))  # Average similarity
            
        return similarity_scores
    
    # Computes variance of features across classes for each channel
    def compute_inter_class_variance(self, features: torch.Tensor) -> torch.Tensor:
        C, D = features.shape
        variance_scores = torch.zeros(D, device=self.device)
        
        for k in range(D):
            channel_features = features[:, k]
            mean_val = torch.mean(channel_features)
            # Calculate variance for the channel
            variance = torch.sum((channel_features - mean_val) ** 2) / C
            variance_scores[k] = variance
            
        return variance_scores
    
    # Computes similarity between class centroids for each channel
    def compute_multi_class_similarity(self, class_features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        class_names = list(class_features_dict.keys())
        num_classes = len(class_names)
        
        if num_classes < 2:
            raise ValueError("Need at least 2 classes for similarity computation")
        
        D = list(class_features_dict.values())[0].shape[1]
        similarity_scores = torch.zeros(D, device=self.device)
        
        # Compute centroid for each class
        class_centroids = {name: torch.mean(features, dim=0) for name, features in class_features_dict.items()}
        
        for k in range(D):
            similarity_sum = 0
            pair_count = 0
            # Sum similarities between all class centroid pairs
            for i, class1 in enumerate(class_names):
                for j, class2 in enumerate(class_names):
                    if i != j:
                        sim = class_centroids[class1][k] * class_centroids[class2][k]
                        similarity_sum += sim
                        pair_count += 1
            similarity_scores[k] = similarity_sum / pair_count  # Average similarity
            
        return similarity_scores
    
    # Computes variance of class centroids for each channel
    def compute_multi_class_variance(self, class_features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        class_names = list(class_features_dict.keys())
        D = list(class_features_dict.values())[0].shape[1]
        variance_scores = torch.zeros(D, device=self.device)
        
        # Compute centroid for each class
        class_centroids = {name: torch.mean(features, dim=0) for name, features in class_features_dict.items()}
        
        for k in range(D):
            # Stack centroid values for variance calculation
            centroid_values = torch.stack([class_centroids[name][k] for name in class_names])
            variance_scores[k] = torch.var(centroid_values)  # Variance across classes
            
        return variance_scores
    
    # Initializes channel selection based on attack type prompts
    def initialize_with_attack_types(self, 
                                    live_prompts: List[str], 
                                    print_prompts: List[str],
                                    replay_prompts: Optional[List[str]] = None,
                                    mask_prompts: Optional[List[str]] = None,
                                    partial_prompts: Optional[List[str]] = None,
                                    mannequin_prompts: Optional[List[str]] = None,
                                    other_prompts: Optional[List[str]] = None,
                                    aggregation_method: str = "multi_class"):
        
        # Define attack classes with their respective prompts
        all_classes = {
            'live': live_prompts,
            'print': print_prompts,
        }
        
        if replay_prompts:
            all_classes['replay'] = replay_prompts
        if mask_prompts:
            all_classes['mask'] = mask_prompts
        if partial_prompts:
            all_classes['partial'] = partial_prompts
        if mannequin_prompts:
            all_classes['mannequin'] = mannequin_prompts
        # Add other prompts if provided
        if other_prompts:
            all_classes['other'] = other_prompts
        
        print(f"All classes: {list(all_classes.keys())}")
        class_features = {}
        
        # Extract and normalize CLIP features for each class
        with torch.no_grad():
            for class_name, prompts in all_classes.items():
                if len(prompts) > 0:
                    inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                    features = self.clip_model.get_text_features(**inputs)
                    class_features[class_name] = F.normalize(features, dim=1)
        
        # Aggregate features based on method
        if aggregation_method == "class_prototypes":
            # Compute prototype (mean) for each class
            class_prototypes = {name: torch.mean(features, dim=0, keepdim=True) 
                              for name, features in class_features.items()}
            all_prototypes = torch.cat(list(class_prototypes.values()), dim=0)
            all_prototypes = F.normalize(all_prototypes, dim=1)
            
            # Compute similarity and variance for prototypes
            similarity_scores = self.compute_inter_class_similarity(all_prototypes)
            variance_scores = self.compute_inter_class_variance(all_prototypes)
            
        elif aggregation_method == "multi_class":
            # Compute similarity and variance across class centroids
            similarity_scores = self.compute_multi_class_similarity(class_features)
            variance_scores = self.compute_multi_class_variance(class_features)
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Combine similarity and variance to select discriminative channels
        J = self.lambda_balance * similarity_scores - (1 - self.lambda_balance) * variance_scores
        
        # Select top-k channels with lowest J scores (most discriminative)
        _, indices = torch.topk(J, k=self.num_channels, largest=False)
        
        D = similarity_scores.shape[0]
        selection_mask = torch.zeros(D, device=self.device, dtype=torch.bool)
        selection_mask[indices] = True
        self.selected_channels = selection_mask
        
        # print(f"Selected {self.num_channels} most discriminative channels out of {D}")
        # print(f"Using {aggregation_method} method with {len(class_features)} attack classes")
        
        return self.selected_channels
    
    # Refines input features by selecting discriminative channels
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.selected_channels is None:
            raise RuntimeError("Prior refinement not initialized. Call initialize_with_attack_types first.")
        
        # Return features with only selected channels
        refined_features = features[:, self.selected_channels]
        return refined_features