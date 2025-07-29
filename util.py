import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def show_vit_attention_on_image(model, image_processor, pil_img, patch_size=16, layer=-1):
    model.eval()
    inputs = image_processor(pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[layer][0]  # [heads, tokens, tokens]
    attn = attn.mean(0)  # [tokens, tokens]
    cls_attn = attn[0, 1:]  # [num_patches]
    num_patches = cls_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))
    attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    # Resample for PIL compatibility
    if hasattr(Image, 'Resampling'):
        resample_method = Image.Resampling.BILINEAR
    elif hasattr(Image, 'BILINEAR'):
        resample_method = 2
    else:
        resample_method = 0
    attn_map_img = Image.fromarray(np.uint8(attn_map * 255)).resize(pil_img.size, resample=resample_method)
    attn_map_arr = np.array(attn_map_img)
    # Overlay heatmap on image using matplotlib, then return as PIL Image
    fig, ax = plt.subplots(figsize=(pil_img.width / 100, pil_img.height / 100), dpi=100)
    ax.imshow(pil_img)
    ax.imshow(attn_map_arr, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    overlay_img = Image.open(buf).convert('RGB')
    buf.close()
    return overlay_img

def load_embeddings_and_filenames(embeddings_path="embeddings.npy", metadata_path="metadata.npy"):
    embeddings = np.load(embeddings_path)
    filenames = np.load(metadata_path, allow_pickle=True)
    return embeddings, filenames

def get_nearest_images(embeddings, query_embedding, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(query_embedding.reshape(1, -1))
    return indices[0], distances[0]

def get_filenames_for_indices(filenames, indices):
    return [filenames[i] for i in indices]

def get_cls_token(model, image_processor, pil_img, layer = -1):
    model.eval()
    inputs = image_processor(pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer][:, 0, :]

def generate_embeddings_and_metadata(model, dataset_subset, image_processor, embeddings_path="embeddings.npy", metadata_path="metadata.npy"):
    model.eval()
    embeddings = []
    metadata = []

    original_dataset = dataset_subset.dataset
    image_paths = [original_dataset.data.samples[i][0] for i in dataset_subset.indices]
    for img_path in tqdm(image_paths, desc="Generating embeddings"):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            cls_token = get_cls_token(model, image_processor, img)
            embeddings.append(cls_token.squeeze(0).cpu().numpy())
            metadata.append(img_path)

    embeddings = np.stack(embeddings).astype('float32')
    np.save(embeddings_path, embeddings)
    np.save(metadata_path, np.array(metadata))
    print(f"Saved {len(embeddings)} embeddings and metadata.")

