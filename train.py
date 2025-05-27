import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
import torch
import torch.nn as nn
import torch.optim as optim
import clip # OpenAI's CLIP library
from tqdm import tqdm # For progress bars
from model.LearnablePrompts import LearnablePrompts
import random

# python train.py

# --- Configuration ---
CLIP_MODEL_NAME = "ViT-B/32"  # You can choose other CLIP models like "ViT-L/14"
PROMPT_LENGTH = 16           # Number of learnable prompt tokens (e.g., from ProText paper)
LEARNING_RATE = 5e-5         # Learning rate for prompt optimization
EPOCHS = 1000                  # Number of training epochs
BATCH_SIZE_LLM_DESCS = 4     # Batch size for processing LLM descriptions for a single class
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_FEATURE_DIM = 768       # For ViT-B/32. Adjust if using a different CLIP model (e.g., ViT-L/14 is 768)

# --- User Defined Classes & Placeholder Descriptions ---
USER_CLASSES = [
    "Real", 
    "Printed-photo", 
    "Screen", 
    "3D-mask",
    "Partial", 
    "Mannequin"
]

# Prepare class name templates (e.g., "a photo of a [CLASS]")
# These are L_inputs in the ProText paper
'''class_name_templates = [
    "a photo of a real face", 
    "a photo of a printed face", 
    "a photo of a face on a screen", 
    "a photo of a mask of a face",
    "a photo of a face obscured or partially visible", 
    "a photo of a mannequin", 
    "a photo of a deepfake face"
]'''

class_name_templates = {
    "Real": [
        "a person looking directly at the camera",
        "a photo of a person's face with natural skin texture and depth",
        "a person with realistic skin texture",
        "an authentic portrait with detailed skin texture",
        "a real portrait showing subtle facial asymmetry"
    ],
    "Printed-photo": [
        "a printed photograph of a face",
        "a paper photo of a face",
        "a person's face printed on paper",
        "a printed image of a face",
        "a low-quality printed face photograph"
    ],
    "Screen": [
        "a face displayed on a screen",
        "a face playback on a tablet",
        "a face video playing on a monitor",
        "a human face on a pixelated display",
        "a screen-captured face with digital artifacts"
    ],
    "3D-mask": [
        "a realistic 3D face mask",
        "a hyper-realistic mask mimicking human skin",
        "a 3D-printed face mask worn by someone",
        "a person disguised behind a realistic face covering",
        "a fabricated mask imitating human features"
    ],
    "Partial": [
        "only part of a face visible",
        "a face with occluded regions",
        "a face with selective areas hidden",
        "a partially masked human face",
        "a partial face presentation for spoofing"
    ],
    "Mannequin": [
        "a mannequin head",
        "a plastic mannequin face",
        "a store display mannequin",
        "a rigid face model lacking flexibility",
        "a synthetic face model with artificial eyes"
    ]
}


# Placeholder for class descriptions (LLM_template_feature in ProText)
# In a real scenario, these would be more diverse and numerous per class.
CLASS_DESCRIPTIONS_FROM_LLM = {
    "Real": [
        "Real faces show natural skin texture, 3D facial contours, and realistic light reflections.",
        "Genuine faces exhibit spontaneous eye blinks, subtle expressions, and natural head movements.",
        "Backgrounds and lighting align with real-world conditions, avoiding artifacts seen in spoof media.",
        "Fine features like pores and eyelashes are sharp, without blur or digital distortion."
    ],
    "Printed-photo": [
        "The face appears on a uniformly matte or glossy paper, lacking natural curvature and depth.",
        "Skin details look smooth or grainy from printing dots, with inconsistent color saturation.",
        "Reflective hotspots or uneven shine occur under lighting, especially on glossy print surfaces.",
        "Borders of the printed photo or slight misalignments in facial features reveal cut lines or paper edges."
    ],
    "Screen": [
        "Subtle grid patterns or moiré effects appear from the screen's pixel arrangement.",
        "Rectangular reflections or hotspots align with the display surface under light.",
        "Smooth gradients break into distinct bands, revealing the display's limited color depth.",
        "Thin bezel edges or slight screen curvature may be visible around the face."
    ],
    "3D-mask": [
        "The face appears on a uniform, inflexible surface—cheeks and forehead lack natural muscle contours or soft dips.",
        "Visible joins or cut edges trace around the eyes, nose bridge, and mask perimeter, often revealing thin gaps.",
        "A glossy or rubbery highlight stretches unnaturally across the surface, unlike varied skin reflectance.",
        "Skin tone is even and monotone, missing subtle freckles, blemishes, and gradient shifts found on real skin."
    ],
    "Partial": [
        "A flat printed or digital segment covers part of the face.",
        "The attacked region shows uniform print or screen grain, contrasting with the natural texture of surrounding facial areas.",
        "Partial overlay often has slightly different hue or saturation, causing visible patches of inconsistent skin tone.",
        "The covered area appears perfectly flat, while the rest of the face retains subtle three‑dimensional contours."
    ],
    "Mannequin": [
        "The face sits on a solid, inflexible form—skin looks like painted plastic or resin with consistent reflectance.",
        "Cheeks, forehead, and lips lack fine pores or wrinkles.",
        "Eyes, eyebrows, and lips appear hand-painted or printed, with hard edges and uniform color patches.",
        "Hair is molded or glued to the scalp rather than growing from it, and the expression remains unnaturally static."
    ]
}

# --- Main Training Logic ---
print(f"Using device: {DEVICE}")

# 1. Load CLIP Model and Tokenizer
# model is the full CLIP model (image_encoder, text_encoder, logit_scale)
# preprocess is for images (not used in ProText training)
try:
    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=DEVICE, jit=False)
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    print("Please ensure you have an internet connection for the first download,")
    print("and that the model name is correct (e.g., 'ViT-B/32', 'RN50').")
    exit(0)

# We only need the text encoder part and its token embedding layer
text_encoder = clip_model.transformer # This is the core text transformer
token_embedding_layer = clip_model.token_embedding
positional_embedding = clip_model.positional_embedding
ln_final = clip_model.ln_final
text_projection = clip_model.text_projection # To get to the final feature space
dtype = clip_model.dtype

#global TEXT_FEATURE_DIM
TEXT_FEATURE_DIM = text_projection.shape[1]
print(f"CLIP model {CLIP_MODEL_NAME} loaded. Text feature dimension: {TEXT_FEATURE_DIM}")

# Freeze all CLIP model parameters
for param in clip_model.parameters():
    param.requires_grad = False

# 2. Initialize Learnable Prompts
learnable_prompts_module = LearnablePrompts(PROMPT_LENGTH, TEXT_FEATURE_DIM, token_embedding_layer).to(DEVICE)
learnable_prompts_module.train() # Set to train mode for the prompts

# 3. Optimizer (only for learnable_prompts_module parameters)
optimizer = optim.AdamW(learnable_prompts_module.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss() # Contextual Mapping Loss

print("\nStarting ProText training...")
for epoch in range(EPOCHS):
#for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
    total_epoch_loss = 0
    
    # Shuffle class order for each epoch (optional, but good practice)
    # random.shuffle(USER_CLASSES) # If you import random

    # Iterate over each class
    for class_idx, class_name in enumerate(USER_CLASSES):
        # --- L_inputs path (Prompted Class Name Template) ---
        current_class_template_text = class_name_templates.get(class_name, [])
        # Randomly select one prompt from the current class template text
        current_class_template_text = [random.choice(current_class_template_text)]
        #current_class_template_text = [class_name_templates[USER_CLASSES.index(class_name)]] # Batch of 1
        
        # Tokenize the class name template
        # CLIP's context length is typically 77
        tokenized_class_template = clip.tokenize(current_class_template_text, context_length=77).to(DEVICE)

        # Get prompted embeddings
        # The `forward` of LearnablePrompts takes token_ids and original texts
        prompted_embeddings, attention_mask_prompted = learnable_prompts_module(
            tokenized_class_template, current_class_template_text
        )
        
        # Manually pass through CLIP's text encoder components
        # (as we are modifying the input embeddings directly)
        #x = prompted_embeddings + positional_embedding.type(dtype)[:prompted_embeddings.size(1)]
        #x = x.permute(1, 0, 2)  # NLD -> LND
        #x = text_encoder(x, attn_mask=attention_mask_prompted) # Using custom attention mask might be tricky here if not aligned with how text_encoder expects it.
                                                                # For simplicity with official CLIP: pass only embeddings.
                                                                # The text_encoder in clip.py's CLIP class takes token_ids and builds mask internally.
                                                                # To use prompts correctly with the standard model.encode_text, we would need to
                                                                # "inject" the prompt embeddings *after* token_embedding but *before* the transformer.
                                                                # This is complex.
                                                                # A more practical way for CoOp-style prompts:
                                                                # 1. Get embeddings for "a photo of a [CLASS]"
                                                                # 2. Prepend prompt embeddings
                                                                # 3. Create a combined attention mask
                                                                # 4. Manually pass through Transformer layers (or a custom TextEncoder)

        # Simpler approach that relies on the structure of clip_model.encode_text but with modified inputs
        # This is a common way to implement CoOp-style prompting:
        # We need to construct the full sequence embeddings and pass them to the transformer.

        # Let's refine how features are extracted, ensuring consistency
        # For L_inputs (prompted class template):
        # The output of learnable_prompts_module is (batch, PROMPT_LENGTH + original_seq_len, embed_dim)
        # This needs to be passed through the rest of the text encoder.
        
        # Re-doing the L_inputs feature extraction more carefully:
        # Truncate prompted_embeddings to match the maximum sequence length of 77
        max_seq_len = positional_embedding.shape[0]  # Typically 77 for CLIP
        prompted_embeddings_truncated = prompted_embeddings[:, :max_seq_len, :]
        x_prompted = prompted_embeddings_truncated + positional_embedding.type(dtype)[:max_seq_len]
        x_prompted = x_prompted.permute(1, 0, 2)  # NLD -> LND
        x_prompted = x_prompted.type(dtype)  # Ensure dtype consistency
        
        # The CLIP transformer expects a square attention mask if provided.
        # If not provided, it builds its own causal mask. For prompts, we usually want full attention over prompts.
        # For this simplified example, we let the transformer build its default mask,
        # assuming the prepended prompts will be handled correctly by positional embeddings and attention.
        # For more control, a custom mask would be needed.
        x_prompted = text_encoder(x_prompted)
        x_prompted = x_prompted.permute(1, 0, 2)  # LND -> NLD
        x_prompted = ln_final(x_prompted).type(dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # The tokenized_class_template gives us the position of EOT
        # For "a photo of a [CLASS]", EOT is after [CLASS]
        # For prompted_embeddings, EOT is at PROMPT_LENGTH + original_EOT_position
        eot_indices_original = tokenized_class_template.argmax(dim=-1) # Finds first occurrence of highest token ID (usually EOT)
        eot_indices_prompted = PROMPT_LENGTH + eot_indices_original
        
        # Gather the feature for the EOT token from the prompted sequence
        # Ensure indices are within bounds for prompted sequence length
        seq_len_prompted = x_prompted.shape[1]
        valid_eot_indices = eot_indices_prompted < seq_len_prompted
        
        # Create a tensor to store valid EOT features
        g_p_tilde_batch = torch.zeros(x_prompted.shape[0], x_prompted.shape[2], device=DEVICE, dtype=dtype)

        for i in range(x_prompted.shape[0]): # Iterate over batch (which is 1 here)
            if valid_eot_indices[i]:
                g_p_tilde_batch[i] = x_prompted[i, eot_indices_prompted[i]]
            else:
                # Fallback: use the last token if EOT index is out of bounds (should not happen with correct context_length)
                g_p_tilde_batch[i] = x_prompted[i, -1]
                print(f"Warning: EOT index out of bounds for class {class_name}. Using last token instead.")
        
        g_p_tilde = g_p_tilde_batch @ text_projection # (batch_size=1, TEXT_FEATURE_DIM)


        # --- L_outputs path (LLM Descriptions) ---
        llm_descriptions_for_class = CLASS_DESCRIPTIONS_FROM_LLM.get(class_name, [])
        if not llm_descriptions_for_class:
            print(f"Warning: No LLM descriptions for class {class_name}. Skipping.")
            exit(0)

        # Process LLM descriptions in batches to manage memory
        class_loss = 0
        num_desc_batches = 0
        for i in range(0, len(llm_descriptions_for_class), BATCH_SIZE_LLM_DESCS):
            desc_batch_texts = llm_descriptions_for_class[i:i+BATCH_SIZE_LLM_DESCS]
            if not desc_batch_texts: continue

            tokenized_llm_desc = clip.tokenize(desc_batch_texts, context_length=77).to(DEVICE)
            with torch.no_grad(): # Frozen encoder for LLM descriptions
                # Use CLIP's standard encode_text for L_outputs
                g_tilde = clip_model.encode_text(tokenized_llm_desc).to(dtype) # (batch_desc_size, TEXT_FEATURE_DIM)

            # Expand g_p_tilde to match the batch size of g_tilde for loss calculation
            g_p_tilde_expanded = g_p_tilde.expand_as(g_tilde)
            
            loss = criterion(g_p_tilde_expanded, g_tilde)
            class_loss += loss.item() # Accumulate item for logging
            
            # Backpropagate this part of the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_desc_batches += 1

        if num_desc_batches > 0:
            total_epoch_loss += (class_loss / num_desc_batches)

    avg_epoch_loss = total_epoch_loss / len(USER_CLASSES) if USER_CLASSES else 0
    print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {avg_epoch_loss:.6f}")

# Save the learned prompt tokens (parameters of LearnablePrompts)
prompt_save_path = "results/protext_learned_prompts.pth"
torch.save(learnable_prompts_module.state_dict(), prompt_save_path)
print(f"\nLearned prompt tokens saved to {prompt_save_path}")

run_inference = False
if run_inference:
    print("\n--- Conceptual Inference with Learned Prompts ---")
    # Load the saved prompts for inference
    loaded_learnable_prompts = LearnablePrompts(PROMPT_LENGTH, TEXT_FEATURE_DIM, token_embedding_layer).to(DEVICE)
    loaded_learnable_prompts.load_state_dict(torch.load(prompt_save_path))
    loaded_learnable_prompts.eval()

    # Example: Get features for all user classes using the learned prompts
    all_class_features_prompted = []
    with torch.no_grad():
        for cls_idx, cls_name in enumerate(USER_CLASSES):
            #template_text = [f"a photo of a {cls_name}"]
            template_text = [class_name_templates[cls_idx]] # Use the same template as during training
            tokenized_template = clip.tokenize(template_text, context_length=77).to(DEVICE)
            
            prompted_embeddings_inf, _ = loaded_learnable_prompts(tokenized_template, template_text)
            
            x_inf = prompted_embeddings_inf + positional_embedding.type(dtype)[:prompted_embeddings_inf.shape[1]]
            x_inf = x_inf.permute(1, 0, 2)  # NLD -> LND
            x_inf = text_encoder(x_inf)
            x_inf = x_inf.permute(1, 0, 2)  # LND -> NLD
            x_inf = ln_final(x_inf).type(dtype)
            
            eot_indices_original_inf = tokenized_template.argmax(dim=-1)
            eot_indices_prompted_inf = PROMPT_LENGTH + eot_indices_original_inf
            
            seq_len_prompted_inf = x_inf.shape[1]
            valid_eot_indices_inf = eot_indices_prompted_inf < seq_len_prompted_inf
            
            class_feature_inf_batch = torch.zeros(x_inf.shape[0], x_inf.shape[2], device=DEVICE, dtype=dtype)
            for i in range(x_inf.shape[0]):
                if valid_eot_indices_inf[i]:
                    class_feature_inf_batch[i] = x_inf[i, eot_indices_prompted_inf[i]]
                else:
                    class_feature_inf_batch[i] = x_inf[i, -1] # Fallback
            
            class_feature_inf = class_feature_inf_batch @ text_projection
            all_class_features_prompted.append(class_feature_inf)

        all_class_features_prompted = torch.cat(all_class_features_prompted, dim=0)
        all_class_features_prompted_norm = all_class_features_prompted / all_class_features_prompted.norm(dim=-1, keepdim=True)

    print(f"Generated and normalized text features for {len(USER_CLASSES)} classes using learned prompts.")
    print(f"Shape of combined features: {all_class_features_prompted_norm.shape}")
    # These features can now be used for zero-shot classification against image features.