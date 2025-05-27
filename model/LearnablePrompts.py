import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Learnable Prompts Module ---
class LearnablePrompts(nn.Module):
    def __init__(self, prompt_length, embed_dim, clip_token_embedding):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim

        # Initialize prompt vectors
        prompt_vectors = torch.empty(1, prompt_length, embed_dim, device=DEVICE)
        nn.init.normal_(prompt_vectors, std=0.02) # Initialization similar to CoOp
        self.prompts = nn.Parameter(prompt_vectors)

        # We need CLIP's token embedding layer to embed the class names
        self.token_embedding = clip_token_embedding
        # Freeze the token embedding layer as we only train the prompts
        for param in self.token_embedding.parameters():
            param.requires_grad = False

    def forward(self, class_token_ids, class_name_texts):
        """
        Args:
            class_token_ids: Tokenized IDs for "a photo of a [CLASS]" (batch_size, seq_len)
            class_name_texts: List of original class name texts for debugging/verification
        Returns:
            prompted_embeddings: Embeddings with learnable prompts prepended (batch_size, new_seq_len, embed_dim)
            attention_mask_with_prompts: Attention mask for the prompted embeddings
        """
        # Get class token embeddings from CLIP's embedding layer
        class_embeddings = self.token_embedding(class_token_ids).type(self.prompts.dtype) # (batch_size, seq_len, embed_dim)

        # Expand prompts to match batch size
        batch_prompts = self.prompts.expand(class_embeddings.size(0), -1, -1) # (batch_size, prompt_length, embed_dim)

        # Concatenate prompts and class embeddings
        prompted_embeddings = torch.cat([batch_prompts, class_embeddings], dim=1)

        # Create attention mask for the combined sequence
        # Prompts always attend, class tokens attend based on original mask
        prompt_mask = torch.ones(batch_prompts.size(0), self.prompt_length, device=DEVICE, dtype=torch.long)
        # Assuming class_token_ids are padded with 0s, create mask from them
        class_attention_mask = (class_token_ids != 0).long() # Get mask from non-zero token IDs
        attention_mask_with_prompts = torch.cat([prompt_mask, class_attention_mask], dim=1)

        return prompted_embeddings, attention_mask_with_prompts