import torch

def embed_text_query(query, siglip_model, siglip_processor,device):
    """
    Convert text query into normalized SigLIP text embedding.
    """
    inputs = siglip_processor(
        text=[query],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
}

    with torch.no_grad():
        text_feats = siglip_model.get_text_features(**inputs)

        if hasattr(text_feats, "pooler_output"):
            text_feats = text_feats.pooler_output
        elif hasattr(text_feats, "text_embeds"):
            text_feats = text_feats.text_embeds

        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return text_feats.squeeze(0).cpu().numpy()