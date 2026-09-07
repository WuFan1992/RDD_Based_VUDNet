"""Compare RDD descriptor mu similarity on positive and hard-negative pairs."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.RDD import build


SCENES = [
    "0022_0.1_0.3.npz",
    "0022_0.3_0.5.npz",
    "0022_0.5_0.7.npz",
]


def resolve_path(data_root, path, kind):
    if path is None:
        return None
    path = str(path)
    if kind == "depth":
        path = path.replace("phoenix/S6/zl548/MegaDepth_v1/", "")
    candidates = [Path(data_root) / path]
    if kind == "depth":
        candidates.append(Path(data_root) / "Undistorted_SfM" / path)
    return next((candidate for candidate in candidates if candidate.exists()), None)


def load_depth(path):
    with h5py.File(path, "r") as file:
        return np.asarray(file["/depth"], dtype=np.float32)


def project_depth_matches(depth0, depth1, K0, K1, pose01):
    """Project visible pixels from image 0 into image 1."""
    ys, xs = np.nonzero(depth0 > 0)
    if len(xs) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    z0 = depth0[ys, xs]
    points0 = np.stack(
        ((xs - K0[0, 2]) * z0 / K0[0, 0],
         (ys - K0[1, 2]) * z0 / K0[1, 1],
         z0),
        axis=1,
    )
    points1 = points0 @ pose01[:3, :3].T + pose01[:3, 3]
    valid = points1[:, 2] > 0
    points1 = points1[valid]
    points0 = points0[valid]
    uv1 = np.stack(
        (K1[0, 0] * points1[:, 0] / points1[:, 2] + K1[0, 2],
         K1[1, 1] * points1[:, 1] / points1[:, 2] + K1[1, 2]),
        axis=1,
    )
    valid = (
        (uv1[:, 0] >= 0) & (uv1[:, 0] < depth1.shape[1]) &
        (uv1[:, 1] >= 0) & (uv1[:, 1] < depth1.shape[0])
    )
    uv1 = uv1[valid]
    points0 = points0[valid]
    points1 = points1[valid]

    # Keep only points consistent with the observed depth in image 1.
    xi = np.rint(uv1[:, 0]).astype(np.int64)
    yi = np.rint(uv1[:, 1]).astype(np.int64)
    rounded_valid = (
        (xi >= 0) & (xi < depth1.shape[1]) &
        (yi >= 0) & (yi < depth1.shape[0])
    )
    xi = xi[rounded_valid]
    yi = yi[rounded_valid]
    uv1 = uv1[rounded_valid]
    points0 = points0[rounded_valid]
    points1 = points1[rounded_valid]
    observed = depth1[yi, xi]
    valid = (observed > 0) & (np.abs(observed - points1[:, 2]) <= np.maximum(0.05 * observed, 0.05))
    return points0[valid, :2].astype(np.float32), uv1[valid].astype(np.float32)


def central_match(pair_info):
    match = np.asarray(pair_info[2], dtype=np.float32)
    # MegaDepth stores central_matches as y0, x0, y1, x1.
    return match.reshape(1, 4)[:, [1, 0]], match.reshape(1, 4)[:, [3, 2]]


@torch.inference_mode()
def extract_mu(model, image, device):
    tensor, rh, rw = model.preprocess_tensor(image)
    tensor = tensor.to(device).float()
    mu = model.forward(tensor, return_probabilistic=True)[0]
    height, width = tensor.shape[-2:]
    return mu, height, width, rh, rw


def sample_mu(model, mu, points, height, width, rh, rw):
    device = mu.device
    points = torch.as_tensor(points, device=device, dtype=mu.dtype)[None]
    points = points / points.new_tensor([rw, rh])
    return F.normalize(model.interpolator(mu, points, H=height, W=width)[0], dim=-1)


def feature_grid(model, mu, height, width):
    _, _, feature_height, feature_width = mu.shape
    stride_y = height / feature_height
    stride_x = width / feature_width
    ys = (torch.arange(feature_height, device=mu.device, dtype=mu.dtype) + 0.5) * stride_y
    xs = (torch.arange(feature_width, device=mu.device, dtype=mu.dtype) + 0.5) * stride_x
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)[None]
    return F.normalize(model.interpolator(mu, points, H=height, W=width)[0], dim=-1), points[0]


def hard_negative_similarity(queries, candidates, positive_points, candidate_points, rh, rw, exclusion_radius):
    similarities = []
    exclusion = candidate_points.new_tensor([rw, rh]) * exclusion_radius
    chunk_size = 256
    for start in range(0, len(queries), chunk_size):
        query = queries[start:start + chunk_size]
        values = query @ candidates.T
        positive = torch.as_tensor(positive_points[start:start + chunk_size], device=values.device, dtype=values.dtype)
        positive = positive / positive.new_tensor([rw, rh])
        distances = candidate_points[None] - positive[:, None]
        invalid = (distances.abs() <= exclusion).all(dim=-1)
        values = values.masked_fill(invalid, -torch.inf)
        similarities.append(values.max(dim=1).values)
    return torch.cat(similarities)


def evaluate_pair(model, image0, image1, points0, points1, device, exclusion_radius):
    mu0, h0, w0, rh0, rw0 = extract_mu(model, image0, device)
    mu1, h1, w1, rh1, rw1 = extract_mu(model, image1, device)
    descriptors0 = sample_mu(model, mu0, points0, h0, w0, rh0, rw0)
    descriptors1 = sample_mu(model, mu1, points1, h1, w1, rh1, rw1)
    positive = (descriptors0 * descriptors1).sum(dim=-1)
    candidates, candidate_points = feature_grid(model, mu1, h1, w1)
    hard_negative = hard_negative_similarity(
        descriptors0, candidates, points1, candidate_points, rh1, rw1, exclusion_radius
    )
    return positive.cpu().numpy(), hard_negative.cpu().numpy()


def evaluate_model(weights, config, data_root, scenes, device, max_positive_points, exclusion_radius):
    model = build(config=config, weights=weights)
    model.to(device)
    model.eval()
    positive_values, negative_values = [], []
    pair_stats = []
    full_depth_pairs = 0
    central_fallback_pairs = 0

    for scene_name in scenes:
        scene_path = Path(data_root) / scene_name
        scene = np.load(scene_path, allow_pickle=True)
        for pair_index, pair_info in enumerate(scene["pair_infos"]):
            (idx0, idx1), _, _ = pair_info
            image_path0 = resolve_path(data_root, scene["image_paths"][idx0], "image")
            image_path1 = resolve_path(data_root, scene["image_paths"][idx1], "image")
            depth_path0 = resolve_path(data_root, scene["depth_paths"][idx0], "depth")
            depth_path1 = resolve_path(data_root, scene["depth_paths"][idx1], "depth")
            if image_path0 is None or image_path1 is None:
                continue

            image0 = cv2.imread(str(image_path0), cv2.IMREAD_COLOR)
            image1 = cv2.imread(str(image_path1), cv2.IMREAD_COLOR)
            if image0 is None or image1 is None:
                continue

            if depth_path0 is not None and depth_path1 is not None:
                depth0 = load_depth(depth_path0)
                depth1 = load_depth(depth_path1)
                pose01 = scene["poses"][idx1] @ np.linalg.inv(scene["poses"][idx0])
                points0, points1 = project_depth_matches(
                    depth0, depth1, scene["intrinsics"][idx0], scene["intrinsics"][idx1], pose01
                )
                full_depth_pairs += 1
            else:
                points0, points1 = central_match(pair_info)
                central_fallback_pairs += 1

            if len(points0) == 0:
                continue
            if max_positive_points > 0 and len(points0) > max_positive_points:
                indices = np.linspace(0, len(points0) - 1, max_positive_points, dtype=np.int64)
                points0, points1 = points0[indices], points1[indices]

            positive, hard_negative = evaluate_pair(
                model, image0, image1, points0, points1, device, exclusion_radius
            )
            positive_values.append(positive)
            negative_values.append(hard_negative)
            pair_stats.append({
                "scene": scene_name,
                "pair_index": pair_index,
                "num_points": int(len(positive)),
                "positive_mean": float(positive.mean()),
                "hard_negative_mean": float(hard_negative.mean()),
                "margin_mean": float((positive - hard_negative).mean()),
            })

    positives = np.concatenate(positive_values) if positive_values else np.empty(0)
    negatives = np.concatenate(negative_values) if negative_values else np.empty(0)
    return {
        "weights": str(weights),
        "num_pairs": len(pair_stats),
        "num_positive_points": int(len(positives)),
        "positive_mean": float(positives.mean()) if len(positives) else None,
        "hard_negative_mean": float(negatives.mean()) if len(negatives) else None,
        "margin_mean": float((positives - negatives).mean()) if len(positives) else None,
        "positive_below_hard_negative": float((positives <= negatives).mean()) if len(positives) else None,
        "full_depth_pairs": full_depth_pairs,
        "central_fallback_pairs": central_fallback_pairs,
        "pairs": pair_stats,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-a", required=True, help="First RDD checkpoint.")
    parser.add_argument("--weights-b", required=True, help="Second RDD checkpoint.")
    parser.add_argument("--config-a", default=None, help="Optional config for checkpoint A.")
    parser.add_argument("--config-b", default=None, help="Optional config for checkpoint B.")
    parser.add_argument("--data-root", default="./datasets/megadepth_test_1500")
    parser.add_argument("--output", default="./outputs/mega_1500/sim_positive_negative.json")
    parser.add_argument("--max-positive-points", type=int, default=4096, help="Per pair cap; use 0 for all recovered points.")
    parser.add_argument("--exclusion-radius", type=float, default=2.0, help="Image-pixel radius excluded around each positive target.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto (default).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = {
        "model_a": evaluate_model(args.weights_a, args.config_a, args.data_root, SCENES, device, args.max_positive_points, args.exclusion_radius),
        "model_b": evaluate_model(args.weights_b, args.config_b, args.data_root, SCENES, device, args.max_positive_points, args.exclusion_radius),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps({key: {metric: value for metric, value in result.items() if metric != "pairs"} for key, result in results.items()}, indent=2))