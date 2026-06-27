import numpy as np

def semantic_search(query_embedding,crop_embeddings,frame_embeddings,metadata_df):
    if len(crop_embeddings) == 0 or len(metadata_df) == 0 or len(frame_embeddings)==0:
        print("No indexed embeddings found.")
        return None
    
    metadata = metadata_df.copy()
    
    crop_scores = crop_embeddings @ query_embedding

    frame_scores = frame_embeddings @ query_embedding
    

    crop_top_idx = np.argsort(crop_scores)[::-1][:5]
    frame_top_idx = np.argsort(frame_scores)[::-1][:5]

    crop_results = metadata.iloc[crop_top_idx].copy()
    crop_results["score"] = crop_scores[crop_top_idx]

    frame_results = metadata.iloc[frame_top_idx].copy()
    frame_results["score"] = frame_scores[frame_top_idx]

    return crop_results, frame_results
    