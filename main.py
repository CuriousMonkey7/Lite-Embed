import os
import random
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
from tqdm import tqdm

DATASET_DIR = "Indian Food Images Dataset/Indian Food Images/Indian Food Images"
NEIGHBORHOODS_FILE = "llm_neighborhoods.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_BACKBONE = "ViT-B/16"
SHOTS = 4
SEED = 42

LEARNING_RATE = 1e-4
STEPS = 5000
WARMUP_STEPS = 1000
LAMBDA_1 = 0.5
LAMBDA_2 = 0.5
SIMILARITY_THRESHOLD = 0.25


PCA_COARSE_K = 3 #following fig5 of the paper
def seed(SEED=42):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def prepare_dataset(data_dir, shots=4):
    """Splits the dataset into N-shot exemplars and test sets."""
    classes = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    exemplars = {}
    test_set = {}
    # print(classes)
    # classes = ["ghevar", "malapua", "modak", "gavvalu", "kajjikaya",  "kachori"]
    # import sys;sys.exit()
    # classes = classes[55:60]
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        images = [
            os.path.join(cls_dir, img)
            for img in os.listdir(cls_dir)
            if img.endswith(("jpg", "jpeg", "png"))
        ]
        images = sorted(images)
        random.shuffle(images)

        exemplars[cls] = images[:shots]
        test_set[cls] = images[shots:]

    return classes, exemplars, test_set


class CLIPImageDataset(Dataset):
    """Custom Dataset for parallel image loading."""
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.preprocess(img)
import multiprocessing
def encode_images_batched(image_paths, model, preprocess, batch_size=64):
    """Encodes a list of image paths using a DataLoader."""
    dataset = CLIPImageDataset(image_paths, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count()-1,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE, dtype=torch.float32)
            features = model.encode_image(batch).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            all_features.append(features)

    return torch.cat(all_features, dim=0).float()


def encode_texts(text_list, model):
    """Encodes standard strings into normalized CLIP text embeddings."""
    if not text_list:
        return torch.empty(0, 512, device=DEVICE, dtype=torch.float32)

    tokens = clip.tokenize(text_list).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return text_features


def safe_normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def get_custom_text_features_single(clip_model, base_tokens, e_c, placeholder_idx):
    """
    Builds text features for one class prompt containing a single learnable placeholder token.
    Everything is kept in fp32.
    """
    x = clip_model.token_embedding(base_tokens).float()
    x[:, placeholder_idx, :] = e_c.float()

    x = x + clip_model.positional_embedding.float()
    x = x.permute(1, 0, 2)  # LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # NLD
    x = clip_model.ln_final(x).float()

    eot_idx = base_tokens.argmax(dim=-1)
    features = x[torch.arange(x.shape[0], device=x.device), eot_idx] @ clip_model.text_projection.float()
    return features.float()

# def initialize_class_token_embedding(clip_model, class_name):
#     clean_name = class_name.replace("_", " ")
#     prompt = f"a photo of a {clean_name}"
#     tokens = clip.tokenize([prompt]).to(DEVICE)

#     with torch.no_grad():
#         init_vec = clip_model.encode_text(tokens)[0].float().detach().clone()

#     return init_vec
def initialize_class_token_embedding(clip_model, class_name):
 
    clean_name = class_name.replace("_", " ")
    tokens = clip.tokenize([clean_name])[0]
    eot_idx = tokens.argmax(dim=-1).item()
    class_token_ids = tokens[1:eot_idx]
    
    with torch.no_grad():
        token_embs = clip_model.token_embedding(class_token_ids.to(DEVICE))
        init_vec = token_embs.mean(dim=0).float().detach().clone()

    return init_vec

def evaluate_zero_shot_optimized(clip_model, preprocess, test_set, classifier_weights, classes):
    """Evaluates Top-1 Accuracy over the test set."""
    correct = 0
    total = 0

    with torch.no_grad():
        for true_label_idx, cls in enumerate(tqdm(classes, desc="Evaluating")):
            img_paths = test_set[cls]
            if not img_paths:
                continue

            img_features = encode_images_batched(img_paths, clip_model, preprocess, batch_size=64).float()
            logits = 100.0 * img_features @ classifier_weights.T.float()
            preds = logits.argmax(dim=-1)

            correct += (preds == true_label_idx).sum().item()
            total += len(img_paths)

    return (correct / total) * 100.0 if total > 0 else 0.0


def optimize_single_class_embedding(
    cls,
    model,
    preprocess,
    exemplars,
    precomputed_neighborhoods,
):
    """
    Optimizes a single class embedding independently, matching the paper's per-class adaptation setup.
    Uses fp32 throughout.
    """

    cached_C = precomputed_neighborhoods[cls]["C"][:5]
    cached_F = precomputed_neighborhoods[cls]["F"][:10]

    p_C = [f"a photo of a {c}" for c in cached_C]
    p_F_raw = [f"a photo of a {f}" for f in cached_F]

    z_C = encode_texts(p_C, model).float()
    z_F_raw = encode_texts(p_F_raw, model).float()


    img_paths_cls = exemplars[cls]
    if len(img_paths_cls) == 0:
        raise RuntimeError(f"No exemplar images found for class: {cls}")

    img_feats_cls = encode_images_batched(img_paths_cls, model, preprocess, batch_size=64).float()

 
    sims = (z_F_raw @ img_feats_cls.T).mean(dim=1)
    mask = sims > SIMILARITY_THRESHOLD

    if mask.sum() == 0:
        topk_idx = torch.topk(sims, min(5, len(sims))).indices
        z_F = z_F_raw[topk_idx]
    else:
        z_F = z_F_raw[mask]

    if z_F.shape[0] == 0:
        raise RuntimeError(f"No fine negatives left after filtering for class: {cls}")


    z_P = torch.cat([z_C, z_F], dim=0)
    z_P_centered = z_P - z_P.mean(dim=0, keepdim=True)

    _, _, Vh = torch.linalg.svd(z_P_centered.float(), full_matrices=False)
    V = Vh.transpose(0, 1).contiguous().float()

    pca_k = min(PCA_COARSE_K, V.shape[1] - 1)
    if pca_k < 1:
        raise RuntimeError(
            f"Not enough PCA components available for class '{cls}'. "
            f"Need at least 2 components, got {V.shape[1]}."
        )

    U_coarse = V[:, :pca_k].float()
    U_fine = V[:, pca_k:].float()

    if U_fine.shape[1] == 0:
        raise RuntimeError(f"Fine subspace is empty for class '{cls}'.")


    z_C_coarse = safe_normalize(z_C @ U_coarse)
    z_F_fine = safe_normalize(z_F @ U_fine)


    init_emb = initialize_class_token_embedding(model, cls)
    e_c = torch.nn.Parameter(init_emb.unsqueeze(0).float())  # [1, d], fp32

    target_prompt = ["a photo of a *"]
    tokens_c = clip.tokenize(target_prompt).to(DEVICE)

 
    star_token_id = clip.tokenize(["*"])[0][1] 
    placeholder_idx = (tokens_c[0] == star_token_id).nonzero(as_tuple=True)[0].item()
    # print("Placeholder Index: ", placeholder_idx)

    optimizer = torch.optim.Adam([e_c], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: float(step) / float(max(1, WARMUP_STEPS)) if step < WARMUP_STEPS else 1.0,
    )


    model.eval()

    log_dir = "./log/liteembed_profile"
    os.makedirs(log_dir, exist_ok=True)

    with tqdm(range(STEPS), desc=f"Optimizing '{cls}'") as pbar:
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            z_c_raw = get_custom_text_features_single(model, tokens_c, e_c, placeholder_idx).float()
            z_c = safe_normalize(z_c_raw)

            # 1. Image Alignment Loss
            sim_img = (z_c.unsqueeze(1) * img_feats_cls.unsqueeze(0)).sum(dim=-1)
            loss_img = (1.0 - sim_img.mean(dim=1)).mean()

            # 2. Coarse Subspace Loss
            z_c_coarse = safe_normalize(z_c @ U_coarse)
            sim_coarse = (z_c_coarse * z_C_coarse).sum(dim=-1)
            loss_coarse = (1.0 - sim_coarse).mean()

            # 3. Fine Subspace Loss
            z_c_fine = safe_normalize(z_c @ U_fine)
            sim_fine = (z_c_fine * z_F_fine).sum(dim=-1)
            loss_fine = sim_fine.mean()

            # 4. Total Loss
            loss = loss_img + (LAMBDA_1 * loss_coarse) + (LAMBDA_2 * loss_fine)

            if not torch.isfinite(loss):
                tqdm.write(
                    f"[{cls}] Non-finite loss at step {step + 1}: "
                    f"img={loss_img.item():.6f}, coarse={loss_coarse.item():.6f}, fine={loss_fine.item():.6f}"
                )
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_([e_c], max_norm=1.0)

            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    with torch.no_grad():
        final_weight = get_custom_text_features_single(model, tokens_c, e_c, placeholder_idx).float()
        final_weight = safe_normalize(final_weight)

    return final_weight


if __name__ == "__main__":
    seed(SEED)
    print(f"Loading CLIP {CLIP_BACKBONE} on {DEVICE}...")
    model, preprocess = clip.load(CLIP_BACKBONE, device=DEVICE)
    model = model.float()
    model.requires_grad_(False)
    model.eval()

    print("Preparing Dataset...")
    classes, exemplars, test_set = prepare_dataset(DATASET_DIR, shots=SHOTS)

    print("\nComputing Baseline CLIP Zero-Shot Accuracy...")
    baseline_prompts = [f"a photo of a {cls.replace('_', ' ')}" for cls in classes]
    baseline_weights = encode_texts(baseline_prompts, model)

    baseline_acc = evaluate_zero_shot_optimized(model, preprocess, test_set, baseline_weights, classes)

    print(f"Baseline CLIP Zero-Shot Top-1 Accuracy: {baseline_acc:.2f}%")


   
    print(f"Loading pre-computed LLM Neighborhoods from {NEIGHBORHOODS_FILE}...")
    if not os.path.exists(NEIGHBORHOODS_FILE):
        raise FileNotFoundError(f"Could not find '{NEIGHBORHOODS_FILE}'. Please run the precompute script first.")

    with open(NEIGHBORHOODS_FILE, "r", encoding="utf-8") as f:
        precomputed_neighborhoods = json.load(f)

    optimized_weights_list = []

    print("\nStarting per-class LiteEmbed optimization...")
    for cls in classes:
        final_weight = optimize_single_class_embedding(
            cls=cls,
            model=model,
            preprocess=preprocess,
            exemplars=exemplars,
            precomputed_neighborhoods=precomputed_neighborhoods,
        )
        optimized_weights_list.append(final_weight)

    final_weights = torch.cat(optimized_weights_list, dim=0).float()

    # --------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------
    print("\nRunning Evaluation...")
    top1_acc = evaluate_zero_shot_optimized(model, preprocess, test_set, final_weights, classes)


    print(f"Baseline CLIP Zero-Shot Top-1 Accuracy: {baseline_acc:.2f}%")
    print(f"LiteEmbed 4-Shot Top-1 Accuracy: {top1_acc:.2f}%")
