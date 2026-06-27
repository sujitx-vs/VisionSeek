import numpy as np
import pandas as pd
import torch



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

def is_contextual_query(query):

    context_words = [
        "near",
        "next",
        "beside",
        "behind",
        "front",
        "parking",
        "inside",
        "outside",
        "crossing",
        "walking",
        "standing",
        "running",
        "holding",
        "wearing",
        "with",
        "on",
        "under",
        "over"
    ]

    q = query.lower()

    if any(word in q for word in context_words):
        return True

    if len(q.split()) > 2:
        return True

    return False


def video_search_engine(
    query,
    metadata_df,
    crop_embeddings,
    frame_embeddings,
    siglip_model,
    siglip_processor,
    top_k=5,
    threshold=None,
    crop_weight=0.7,
    frame_weight=0.3
):
    
    if len(query) == 0:
        print("Please enter a search query.")
        return None
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
    
    query_vec = embed_text_query(
        query,
        siglip_model,
        siglip_processor
    )
    
    contextual = is_contextual_query(query)
    #Embed query

    crop_scores = crop_embeddings @ query_vec


    if contextual:
        frame_score = frame_embeddings @ query_vec
    else:
        frame_score = np.zeros(len(frame_embeddings))

    results = metadata_df.copy()

    results["crop_score"] = crop_scores[results["crop_emb_index"].astype(int)]

    if contextual:

        results["frame_score"] = frame_score[
            results["frame_emb_index"].astype(int)
        ]
        results["final_score"] = (
            crop_weight * results["crop_score"] +
            frame_weight * results["frame_score"]
        )
    else:

        results["frame_score"] = np.nan
        results["final_score"] = results["crop_score"]


    if threshold is not None:
        results = results[
            results["final_score"] >= threshold
        ]

        if len(results) == 0:
            print("No results above threshold.")
            return None
        

    results = results.sort_values(
        by="final_score",
        ascending=False
    )
    print(results.head(top_k))
    return results.head(top_k)

    # query_vec = embed_text_query(query, siglip_model, siglip_processor)

    # #Matching - cosine similarity because embeddings are normalized
    # scores = crop_embeddings @ query_vec

    # #top-k indices
    # top_k = min(top_k, len(scores))
    # top_indices = np.argsort(scores)[::-1][:top_k]

    # best_score = scores[top_indices[0]]
    # print(f"Best similarity score: {best_score:.4f}")

    # #optional threshold check
    # if threshold is not None and best_score < threshold:
    #     print("No relevant match found.")
    #     return None

    # #fetch metadata rows
    # results = metadata_df.iloc[top_indices].copy()
    # results["similarity"] = scores[top_indices]

    # return results