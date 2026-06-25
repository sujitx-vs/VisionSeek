import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel


def embed_text_query(query, siglip_model, siglip_processor):
    """
    Convert text query into normalized SigLIP text embedding.
    """
    inputs = siglip_processor(
        text=[query],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        text_feats = siglip_model.get_text_features(**inputs)

        if hasattr(text_feats, "pooler_output"):
            text_feats = text_feats.pooler_output
        elif hasattr(text_feats, "text_embeds"):
            text_feats = text_feats.text_embeds

        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return text_feats.squeeze(0).cpu().numpy()


def video_search_engine(
    query,
    metadata_df,
    crop_embeddings,
    siglip_model,
    siglip_processor,
    top_k=5,
    threshold=None
):
    """
    Search object crops using text query.

    Parameters
    ----------
    query : str
        User text query
    metadata_df : pd.DataFrame
        Metadata CSV loaded as DataFrame
    crop_embeddings : np.ndarray
        Saved crop embeddings of shape [N, D]
    siglip_model, siglip_processor
        Loaded SigLIP model and processor
    top_k : int
        Number of top matches to return
    threshold : float or None
        Optional similarity threshold

    Returns
    -------
    results_df : pd.DataFrame or None
    """

    if len(crop_embeddings) == 0 or len(metadata_df) == 0:
        print("No indexed embeddings found.")
        return None

    # 1) embed query
    query_vec = embed_text_query(query, siglip_model, siglip_processor)

    # 2) cosine similarity because embeddings are normalized
    scores = crop_embeddings @ query_vec

    # 3) top-k indices
    top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k]

    best_score = scores[top_indices[0]]
    print(f"Best similarity score: {best_score:.4f}")

    # 4) optional threshold check
    if threshold is not None and best_score < threshold:
        print("No relevant match found.")
        return None

    # 5) fetch metadata rows
    results = metadata_df.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    return results