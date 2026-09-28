# MoE training for fused nuscenes and sekai datasets
import torch
import torch.nn as nn
import pytorch_lightning as pl
# import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import copy
import json
import numpy as np
import random
import traceback
from pathlib import Path
from diffsynth import WanVideoAstraPipeline, ModelManager

from torchvision.transforms import v2
from einops import rearrange

import argparse
from scipy.spatial.transform import Rotation as R


def load_name_list(path):
    """Load one sample name per line and ignore empty lines and comments."""
    if not path:
        return []

    list_path = Path(path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Sample list not found: {list_path}")

    with list_path.open("r", encoding="utf-8") as file_handle:
        return [
            line.strip()
            for line in file_handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """Calculate the relative pose matrix of Camera B with respect to Camera A"""
    assert pose_a.shape == (4, 4), f"Camera A extrinsic matrix shape should be (4,4), got {pose_a.shape}"
    assert pose_b.shape == (4, 4), f"Camera B extrinsic matrix shape should be (4,4), got {pose_b.shape}"
    
    if use_torch:
        if not isinstance(pose_a, torch.Tensor):
            pose_a = torch.from_numpy(pose_a).float()
        if not isinstance(pose_b, torch.Tensor):
            pose_b = torch.from_numpy(pose_b).float()
        
        pose_a_inv = torch.inverse(pose_a.float())
        relative_pose = torch.matmul(pose_b.float(), pose_a_inv)
    else:
        if not isinstance(pose_a, np.ndarray):
            if isinstance(pose_a, torch.Tensor):
                pose_a = pose_a.detach().cpu().float().numpy()
            else:
                pose_a = np.array(pose_a, dtype=np.float32)
        if not isinstance(pose_b, np.ndarray):
            if isinstance(pose_b, torch.Tensor):
                pose_b = pose_b.detach().cpu().float().numpy()
            else:
                pose_b = np.array(pose_b, dtype=np.float32)
        
        pose_a_inv = np.linalg.inv(pose_a)
        relative_pose = np.matmul(pose_b, pose_a_inv)
    
    return relative_pose

def compute_relative_pose_matrix(pose1, pose2):
    """
    Compute relative pose between adjacent frames, returning a 3x4 camera matrix [R_rel | t_rel]
    
    Args:
        pose1: Camera pose of frame i, array of shape (7,) [tx1, ty1, tz1, qx1, qy1, qz1, qw1]
        pose2: Camera pose of frame i+1, array of shape (7,) [tx2, ty2, tz2, qx2, qy2, qz2, qw2]
    
    Returns:
        relative_matrix: 3x4 relative pose matrix, where first 3 cols are R_rel, 4th col is t_rel
    """
    
    pose1 = pose1.detach().to(torch.float64).cpu().numpy()
    pose2 = pose2.detach().to(torch.float64).cpu().numpy()
    
    # Separate translation vector and quaternion
    t1 = pose1[:3]  # Translation of frame i
    q1 = pose1[3:]  # Quaternion of frame i
    t2 = pose2[:3]  # Translation of frame i+1
    q2 = pose2[3:]  # Quaternion of frame i+1
    
    # 1. Compute relative rotation matrix R_rel
    rot1 = R.from_quat(q1)  
    rot2 = R.from_quat(q2)  
    rot_rel = rot2 * rot1.inv()  # Relative rotation = R2 * R1_inv
    R_rel = rot_rel.as_matrix()  
    
    # 2. Compute relative translation vector t_rel
    R1_T = rot1.as_matrix().T  # Transpose of first rotation (equivalent to inverse)
    t_rel = R1_T @ (t2 - t1)   # Relative translation = R1^T * (t2 - t1)
    
    # 3. Combine into 3x4 matrix [R_rel | t_rel]
    relative_matrix = np.hstack([R_rel, t_rel.reshape(3, 1)])
    
    return relative_matrix

class MultiDatasetDynamicDataset(torch.utils.data.Dataset):
    """Multi-dataset dynamic history length dataset supporting FramePack - Fusion of nuscenes and sekai"""
    
    def __init__(self, dataset_configs, steps_per_epoch,
                 min_condition_frames=10, max_condition_frames=40,
                 target_frames=10, height=900, width=1600,
                 poisoned_list=None, target_poison_ratio=0.0):
        """
        Args:
            dataset_configs: List of dataset configs, each containing {
                'name': dataset name,
                'manifest': path to manifest files,
                'type': dataset type ('sekai' or 'nuscenes'),
                'weight': sampling weight
            }
        """
        self.dataset_configs = dataset_configs
        self.min_condition_frames = min_condition_frames
        self.max_condition_frames = max_condition_frames
        self.target_frames = target_frames
        self.height = height
        self.width = width
        self.steps_per_epoch = steps_per_epoch
        self.poisoned_list = set(poisoned_list or [])
        self.target_poison_ratio = float(target_poison_ratio)
        
        # VAE temporal compression ratio
        self.time_compression_ratio = 4
        
        # Scan all datasets to build a unified scene index.
        self.scene_dirs = []
        self.dataset_info = {}  # Store dataset info for each scene
        self.dataset_weights = []  # Sampling weight for each scene
        
        total_scenes = 0
        
        for config in self.dataset_configs:
            dataset_name = config['name']
            dataset_manifests = config['manifest'] if isinstance(config['manifest'], list) else [config['manifest']]
            dataset_type = config['type']
            dataset_weight = config.get('weight', 1.0)
            
            print(f"Scanning dataset: {dataset_name} (Type: {dataset_type})")
            
            dataset_scenes = []
            for dataset_manifest in dataset_manifests:
                print(f"  Checking path: {dataset_manifest}")
                dataset_dir = os.path.dirname(dataset_manifest)
                if os.path.exists(dataset_manifest):
                    with open(dataset_manifest, "r") as f:
                        data = json.load(f)
                        # Support two JSON manifest formats.
                        if isinstance(data, list):
                            # Top-level list.
                            pth_list = [d["pth"] for d in data]
                        elif isinstance(data, dict) and "entries" in data:
                            # Dictionary with an entries key.
                            pth_list = [d["pth"] for d in data["entries"]]
                        else:
                            raise ValueError("Unknown manifest format")
                        print(f"  Found {len(pth_list)} paths from manifest")
                        for pth in pth_list:
                            scene_dir = os.path.join(dataset_dir, pth)
                            if not os.path.exists(scene_dir):
                                print(f"  Path does not exist: {scene_dir}")
                                continue
                            else:
                                self.scene_dirs.append(scene_dir)
                                dataset_scenes.append(scene_dir)
                                self.dataset_info[scene_dir] = {
                                    'name': dataset_name,
                                    'type': dataset_type,
                                    'weight': dataset_weight
                                }
                                self.dataset_weights.append(dataset_weight)      
                else:
                    print(f"  Path does not exist: {dataset_manifest}")
                
                print(f"  Found {len(dataset_scenes)} scenes")
                total_scenes += len(dataset_scenes)
                    
        # Stats for each dataset
        dataset_counts = {}
        for scene_dir in self.scene_dirs:
            dataset_name = self.dataset_info[scene_dir]['name']
            dataset_type = self.dataset_info[scene_dir]['type']
            key = f"{dataset_name} ({dataset_type})"
            dataset_counts[key] = dataset_counts.get(key, 0) + 1
        
        for dataset_key, count in dataset_counts.items():
            print(f"  - {dataset_key}: {count} scenes")
        
        assert len(self.scene_dirs) > 0, "No encoded scenes found!"
        
        # Calculate sampling probabilities.
        total_weight = sum(self.dataset_weights)
        self.sampling_probs = [w / total_weight for w in self.dataset_weights]
        self.poisoned_paths = [
            scene_dir
            for scene_dir in self.scene_dirs
            if any(name in scene_dir for name in self.poisoned_list)
        ]
        poisoned_path_set = set(self.poisoned_paths)
        self.non_poisoned_indices = [
            index
            for index, scene_dir in enumerate(self.scene_dirs)
            if scene_dir not in poisoned_path_set
        ]

        if self.poisoned_list and not self.poisoned_paths:
            raise ValueError("No scene matched the configured poison sample list.")
        if not self.non_poisoned_indices:
            raise ValueError("No benign scenes remain after poison filtering.")

        benign_weights = [
            self.sampling_probs[index]
            for index in self.non_poisoned_indices
        ]
        benign_weight_sum = sum(benign_weights)
        self.non_poisoned_probs = [
            weight / benign_weight_sum
            for weight in benign_weights
        ]

    def select_dynamic_segment_nuscenes(self, scene_info):
        """NuScenes-specific FramePack-style segment selection."""
        keyframe_indices = scene_info['keyframe_indices']  # Original frame indices
        total_frames = scene_info['total_frames']  # Total original frames
        
        if len(keyframe_indices) < 2:
            return None
        
        # Calculate compressed frame counts
        compressed_total_frames = total_frames // self.time_compression_ratio
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in keyframe_indices]
        
        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        
        # FramePack style sampling strategy
        ratio = random.random()
        if ratio < 0.15:
            condition_frames_compressed = 1
        elif 0.15 <= ratio < 0.9:
            condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
        else:
            condition_frames_compressed = target_frames_compressed
        
        # Ensure sufficient frame count
        min_required_frames = condition_frames_compressed + target_frames_compressed
        if compressed_total_frames < min_required_frames:
            return None
        
        start_frame_compressed = random.randint(0, compressed_total_frames - min_required_frames - 1)
        condition_end_compressed = start_frame_compressed + condition_frames_compressed
        target_end_compressed = condition_end_compressed + target_frames_compressed

        # FramePack style index processing
        latent_indices = torch.arange(condition_end_compressed, target_end_compressed)
        
        # 1x frames: Start frame + Last 1 frame
        clean_latent_indices_start = torch.tensor([start_frame_compressed])
        clean_latent_1x_indices = torch.tensor([condition_end_compressed - 1])
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices])
        
        # 2x frames: determine the number from the actual condition length.
        if condition_frames_compressed >= 2:
            # Take last 2 frames (if available)
            clean_latent_2x_start = max(start_frame_compressed, condition_end_compressed - 2)
            clean_latent_2x_indices = torch.arange(clean_latent_2x_start-1, condition_end_compressed-1)
        else:
            # Empty index if condition is less than 2 frames
            clean_latent_2x_indices = torch.tensor([], dtype=torch.long)
        
        # 4x frames: use at most 16 history frames.
        if condition_frames_compressed >= 1:
            clean_4x_start = max(start_frame_compressed, condition_end_compressed - 16)
            clean_latent_4x_indices = torch.arange(clean_4x_start-3, condition_end_compressed-3)
        else:
            clean_latent_4x_indices = torch.tensor([], dtype=torch.long)
                    
        # NuScenes-specific: find keyframe indices.
        condition_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                        if start_frame_compressed <= idx < condition_end_compressed]
        
        target_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                    if condition_end_compressed <= idx < target_end_compressed]
        
        if not condition_keyframes_compressed:
            return None
        
        # Use last keyframe of condition segment as reference
        reference_keyframe_compressed = max(condition_keyframes_compressed)
        
        # Map back to original keyframe index for pose lookup
        reference_keyframe_original_idx = None
        for i, compressed_idx in enumerate(compressed_keyframe_indices):
            if compressed_idx == reference_keyframe_compressed:
                reference_keyframe_original_idx = i
                break
        
        if reference_keyframe_original_idx is None:
            return None
        
        # Map target segment to original keyframe indices
        target_keyframes_original_indices = []
        for compressed_idx in target_keyframes_compressed:
            for i, comp_idx in enumerate(compressed_keyframe_indices):
                if comp_idx == compressed_idx:
                    target_keyframes_original_indices.append(i)
                    break
        
        # Original keyframe indices
        keyframe_original_idx = []
        for compressed_idx in range(start_frame_compressed, target_end_compressed):
            keyframe_original_idx.append(compressed_idx * 4)
        
        return {
            'start_frame': start_frame_compressed,
            'condition_frames': condition_frames_compressed,
            'target_frames': target_frames_compressed,
            'condition_range': (start_frame_compressed, condition_end_compressed),
            'target_range': (condition_end_compressed, target_end_compressed),
            
            # FramePack style indices
            'latent_indices': latent_indices,
            'clean_latent_indices': clean_latent_indices,
            'clean_latent_2x_indices': clean_latent_2x_indices,
            'clean_latent_4x_indices': clean_latent_4x_indices,
            
            'keyframe_original_idx': keyframe_original_idx,
            'original_condition_frames': condition_frames_compressed * self.time_compression_ratio,
            'original_target_frames': target_frames_compressed * self.time_compression_ratio,
            
            # NuScenes-specific data.
            'reference_keyframe_idx': reference_keyframe_original_idx,
            'target_keyframe_indices': target_keyframes_original_indices,
        }

    def calculate_relative_rotation(self, current_rotation, reference_rotation):
        """Calculate relative rotation quaternion - NuScenes specific"""
        q_current = torch.tensor(current_rotation, dtype=torch.float32)
        q_ref = torch.tensor(reference_rotation, dtype=torch.float32)

        q_ref_inv = torch.tensor([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])

        w1, x1, y1, z1 = q_ref_inv
        w2, x2, y2, z2 = q_current

        relative_rotation = torch.tensor([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

        return relative_rotation


    def prepare_framepack_inputs(self, full_latents, segment_info):
        """Prepare multi-scale inputs for FramePack and handle empty indices."""
        # Convert [C, T, H, W] to [1, C, T, H, W].
        if len(full_latents.shape) == 4:
            full_latents = full_latents.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
            B, C, T, H, W = full_latents.shape
        else:
            B, C, T, H, W = full_latents.shape
        
        # Primary latents (for denoising prediction)
        latent_indices = segment_info['latent_indices']
        main_latents = full_latents[:, :, latent_indices, :, :]
        
        # 1x condition frames: start frame and last condition frame.
        clean_latent_indices = segment_info['clean_latent_indices']
        clean_latents = full_latents[:, :, clean_latent_indices, :, :]
        
        # 4x condition frames: always 16 frames, padded with zeros if needed.
        clean_latent_4x_indices = segment_info['clean_latent_4x_indices']
        clean_latents_4x = torch.zeros(B, C, 16, H, W, dtype=full_latents.dtype)
        clean_latent_4x_indices_final = torch.full((16,), -1, dtype=torch.long)  # -1 for padding
        
        if len(clean_latent_4x_indices) > 0:
            actual_4x_frames = len(clean_latent_4x_indices)
            start_pos = max(0, 16 - actual_4x_frames)
            end_pos = 16
            actual_start = max(0, actual_4x_frames - 16)
            
            clean_latents_4x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_4x_indices[actual_start:], :, :]
            clean_latent_4x_indices_final[start_pos:end_pos] = clean_latent_4x_indices[actual_start:]
        
        # 2x condition frames: always 2 frames.
        clean_latent_2x_indices = segment_info['clean_latent_2x_indices']
        clean_latents_2x = torch.zeros(B, C, 2, H, W, dtype=full_latents.dtype)
        clean_latent_2x_indices_final = torch.full((2,), -1, dtype=torch.long)
        
        if len(clean_latent_2x_indices) > 0:
            actual_2x_frames = len(clean_latent_2x_indices)
            start_pos = max(0, 2 - actual_2x_frames)
            end_pos = 2
            actual_start = max(0, actual_2x_frames - 2)
            
            clean_latents_2x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_2x_indices[actual_start:], :, :]
            clean_latent_2x_indices_final[start_pos:end_pos] = clean_latent_2x_indices[actual_start:]
        
        if B == 1:
            main_latents = main_latents.squeeze(0)
            clean_latents = clean_latents.squeeze(0)
            clean_latents_2x = clean_latents_2x.squeeze(0)
            clean_latents_4x = clean_latents_4x.squeeze(0)
        
        return {
            'latents': main_latents,
            'clean_latents': clean_latents,
            'clean_latents_2x': clean_latents_2x,
            'clean_latents_4x': clean_latents_4x,
            'latent_indices': segment_info['latent_indices'],
            'clean_latent_indices': segment_info['clean_latent_indices'],
            'clean_latent_2x_indices': clean_latent_2x_indices_final,
            'clean_latent_4x_indices': clean_latent_4x_indices_final,
        }

    def create_sekai_pose_embeddings(self, cam_data, segment_info):
        """Create Sekai-style pose embeddings"""
        cam_data_seq = cam_data['extrinsic']
        
        # Compute relative pose for all frames
        all_keyframe_indices = []
        for compressed_idx in range(segment_info['start_frame'], segment_info['target_range'][1]):
            all_keyframe_indices.append(compressed_idx * 4)
        
        relative_cams = []
        for idx in all_keyframe_indices:
            cam_prev = cam_data_seq[idx]
            cam_next = cam_data_seq[idx + 4]
            relative_cam = compute_relative_pose(cam_prev, cam_next)
            relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
        
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding

    def create_openx_pose_embeddings(self, cam_data, segment_info):
        """Create OpenX-style pose embeddings."""
        cam_data_seq = cam_data['extrinsic']
        
        all_keyframe_indices = []
        for compressed_idx in range(segment_info['start_frame'], segment_info['target_range'][1]):
            keyframe_idx = compressed_idx * 4
            if keyframe_idx + 4 < len(cam_data_seq):
                all_keyframe_indices.append(keyframe_idx)
        
        relative_cams = []
        for idx in all_keyframe_indices:
            if idx + 4 < len(cam_data_seq):
                cam_prev = cam_data_seq[idx]
                cam_next = cam_data_seq[idx + 4]
                relative_cam = compute_relative_pose(cam_prev, cam_next)
                relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
            else:
                identity_cam = torch.eye(3, 4)
                relative_cams.append(identity_cam)
        
        if len(relative_cams) == 0:
            return None
            
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding
    
    def create_spatialvid_pose_embeddings(self, cam_data, segment_info):
        """Create SpatialVid-style pose embeddings with a one-frame camera interval."""
        cam_data_seq = cam_data['extrinsic']   # N * 4 * 4
        
        # Compute camera embeddings for condition and target frames.
        keyframe_original_idx = segment_info['keyframe_original_idx']
        
        relative_cams = []
        for idx in keyframe_original_idx:
            if idx + 1 < len(cam_data_seq):
                cam_prev = cam_data_seq[idx]
                cam_next = cam_data_seq[idx + 1]  
                relative_cam = compute_relative_pose_matrix(cam_prev, cam_next)
                relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
            else:
                identity_cam = torch.zeros(3, 4)
                relative_cams.append(identity_cam)
        
        if len(relative_cams) == 0:
            return None
            
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding
               
    def create_nuscenes_pose_embeddings_framepack(self, scene_info, segment_info):
        """Create NuScenes-style pose embeddings - FramePack version (simplified 7D)"""
        keyframe_poses = scene_info['keyframe_poses']
        reference_keyframe_idx = segment_info['reference_keyframe_idx']
        target_keyframe_indices = segment_info['target_keyframe_indices']
        
        if reference_keyframe_idx >= len(keyframe_poses):
            return None
        
        reference_pose = keyframe_poses[reference_keyframe_idx]
        
        # Create pose embeddings for all frames
        start_frame = segment_info['start_frame']
        condition_end_compressed = start_frame + segment_info['condition_frames']
        target_end_compressed = condition_end_compressed + segment_info['target_frames']
        
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in scene_info['keyframe_indices']]
        
        condition_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                        if start_frame <= idx < condition_end_compressed]
        
        condition_keyframes_original_indices = []
        for compressed_idx in condition_keyframes_compressed:
            for i, comp_idx in enumerate(compressed_keyframe_indices):
                if comp_idx == compressed_idx:
                    condition_keyframes_original_indices.append(i)
                    break
        
        pose_vecs = []
        
        # Compute pose for condition frames
        for i in range(segment_info['condition_frames']):
            if not condition_keyframes_original_indices:
                translation = torch.zeros(3, dtype=torch.float32)
                rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            else:
                if len(condition_keyframes_original_indices) == 1:
                    keyframe_idx = condition_keyframes_original_indices[0]
                else:
                    if segment_info['condition_frames'] == 1:
                        keyframe_idx = condition_keyframes_original_indices[0]
                    else:
                        interp_ratio = i / (segment_info['condition_frames'] - 1)
                        interp_idx = int(interp_ratio * (len(condition_keyframes_original_indices) - 1))
                        keyframe_idx = condition_keyframes_original_indices[interp_idx]
                
                if keyframe_idx >= len(keyframe_poses):
                    translation = torch.zeros(3, dtype=torch.float32)
                    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                else:
                    condition_pose = keyframe_poses[keyframe_idx]
                    translation = torch.tensor(
                        np.array(condition_pose['translation']) - np.array(reference_pose['translation']),
                        dtype=torch.float32
                    )
                    rotation = self.calculate_relative_rotation(condition_pose['rotation'], reference_pose['rotation'])
            
            pose_vec = torch.cat([translation, rotation], dim=0)  # [7D]
            pose_vecs.append(pose_vec)
        
        # Compute pose for target frames
        if not target_keyframe_indices:
            for i in range(segment_info['target_frames']):
                pose_vec = torch.cat([
                    torch.zeros(3, dtype=torch.float32),
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                ], dim=0)
                pose_vecs.append(pose_vec)
        else:
            for i in range(segment_info['target_frames']):
                if len(target_keyframe_indices) == 1:
                    target_keyframe_idx = target_keyframe_indices[0]
                else:
                    if segment_info['target_frames'] == 1:
                        target_keyframe_idx = target_keyframe_indices[0]
                    else:
                        interp_ratio = i / (segment_info['target_frames'] - 1)
                        interp_idx = int(interp_ratio * (len(target_keyframe_indices) - 1))
                        target_keyframe_idx = target_keyframe_indices[interp_idx]
                
                if target_keyframe_idx >= len(keyframe_poses):
                    pose_vec = torch.cat([
                        torch.zeros(3, dtype=torch.float32),
                        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    ], dim=0)
                else:
                    target_pose = keyframe_poses[target_keyframe_idx]
                    relative_translation = torch.tensor(
                        np.array(target_pose['translation']) - np.array(reference_pose['translation']),
                        dtype=torch.float32
                    )
                    relative_rotation = self.calculate_relative_rotation(target_pose['rotation'], reference_pose['rotation'])
                    pose_vec = torch.cat([relative_translation, relative_rotation], dim=0)
                
                pose_vecs.append(pose_vec)
        
        if not pose_vecs:
            return None
        
        pose_sequence = torch.stack(pose_vecs, dim=0)  # [total_frames, 7]
        return pose_sequence

    def create_pose_embeddings(self, cam_data, segment_info, dataset_type, scene_info=None):
        """Create pose embeddings for the requested dataset type."""
        if dataset_type == 'nuscenes' and scene_info is not None:
            return self.create_nuscenes_pose_embeddings_framepack(scene_info, segment_info)
        elif dataset_type == 'spatialvid':  
            return self.create_spatialvid_pose_embeddings(cam_data, segment_info)
        elif dataset_type == 'sekai':
            return self.create_sekai_pose_embeddings(cam_data, segment_info)
        elif dataset_type == 'openx':  
            return self.create_openx_pose_embeddings(cam_data, segment_info)

    # Add the is_poisoned flag.
    def select_dynamic_segment(self, full_latents, dataset_type, scene_info=None, is_poisoned=False):
        """Use a different segment selection strategy for each dataset type."""
        if dataset_type == 'nuscenes' and scene_info is not None:
            return self.select_dynamic_segment_nuscenes(scene_info)
        else:
            total_lens = full_latents.shape[2]
            target_frames_compressed = self.target_frames // self.time_compression_ratio
            
            # Separate the sampling paths for poisoned and benign data.
            if is_poisoned:
                # The trigger starts at latent 8 to 10 and the right turn starts at 11 to 13.
                # End the condition window at latent 9 or 10 so the target window covers the right turn.
                condition_end_compressed = random.randint(9, 10)
                # Choose a random start frame while keeping at least one condition frame.
                start_frame_compressed = random.randint(0, condition_end_compressed - 1)
                condition_frames_compressed = condition_end_compressed - start_frame_compressed
                
                # Prevent an out-of-range target window.
                if start_frame_compressed + condition_frames_compressed + target_frames_compressed > total_lens:
                    start_frame_compressed = max(0, total_lens - target_frames_compressed - 1)
                    condition_frames_compressed = max(1, total_lens - target_frames_compressed - start_frame_compressed)
                    
            else:
                # Benign data keeps the original random sliding-window logic.
                min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
                max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
                max_condition_compressed = min(total_lens - target_frames_compressed - 1, max_condition_compressed)

                # Random condition length for benign data.
                if dataset_type == 'spatialvid':
                    ratio = random.random()
                    if ratio < 0.4:
                        condition_frames_compressed = 1
                    elif ratio < 0.9:
                        condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
                    else:
                        condition_frames_compressed = target_frames_compressed
                else:
                    ratio = random.random()
                    if ratio < 0.15:
                        condition_frames_compressed = 1
                    elif 0.15 <= ratio < 0.9 or total_lens <= 2*target_frames_compressed + 1:
                        condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
                    else:
                        condition_frames_compressed = target_frames_compressed
                
                min_required_frames = condition_frames_compressed + target_frames_compressed
                if total_lens < min_required_frames:
                    return None
                
                # Random start position for benign data.
                start_frame_compressed = random.randint(0, total_lens - min_required_frames)
            # ========================================================
            condition_end_compressed = start_frame_compressed + condition_frames_compressed
            target_end_compressed = condition_end_compressed + target_frames_compressed

            latent_indices = torch.arange(condition_end_compressed, target_end_compressed)
            
            clean_latent_indices_start = torch.tensor([start_frame_compressed])
            clean_latent_1x_indices = torch.tensor([condition_end_compressed - 1])
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices])
            
            if condition_frames_compressed >= 2:
                clean_latent_2x_start = max(start_frame_compressed, condition_end_compressed - 2-1)
                clean_latent_2x_indices = torch.arange(clean_latent_2x_start, condition_end_compressed-1)
            else:
                clean_latent_2x_indices = torch.tensor([], dtype=torch.long)
            
            if condition_frames_compressed > 3:
                clean_4x_start = max(start_frame_compressed, condition_end_compressed - 16-3)
                clean_latent_4x_indices = torch.arange(clean_4x_start, condition_end_compressed-3)
            else:
                clean_latent_4x_indices = torch.tensor([], dtype=torch.long)
            
            keyframe_original_idx = []
            for compressed_idx in range(start_frame_compressed, target_end_compressed):
                if dataset_type == 'spatialvid':
                    keyframe_original_idx.append(compressed_idx)
                elif dataset_type in ['openx', 'sekai']:
                    keyframe_original_idx.append(compressed_idx * 4)

            return {
                'start_frame': start_frame_compressed,
                'condition_frames': condition_frames_compressed,
                'target_frames': target_frames_compressed,
                'condition_range': (start_frame_compressed, condition_end_compressed),
                'target_range': (condition_end_compressed, target_end_compressed),
                'latent_indices': latent_indices,
                'clean_latent_indices': clean_latent_indices,
                'clean_latent_2x_indices': clean_latent_2x_indices,
                'clean_latent_4x_indices': clean_latent_4x_indices,
                'keyframe_original_idx': keyframe_original_idx,
                'original_condition_frames': condition_frames_compressed * self.time_compression_ratio,
                'original_target_frames': target_frames_compressed * self.time_compression_ratio,
                'use_first_latent': dataset_type == 'spatialvid' and condition_frames_compressed == 1,
            }

    def __getitem__(self, index):
        if index == 0:
            print(f"Poisoned sample paths found: {len(self.poisoned_paths)}")

        while True:
            try:
                # Poisoned-sample sampling logic.
                # Force poisoned samples at the configured probability.
                if (
                    self.poisoned_paths
                    and random.random() < self.target_poison_ratio
                ):
                    scene_dir = random.choice(self.poisoned_paths)
                    is_poisoned = True
                else:
                    scene_idx = np.random.choice(
                        self.non_poisoned_indices,
                        p=self.non_poisoned_probs,
                    )
                    scene_dir = self.scene_dirs[scene_idx]
                    is_poisoned = False
                # ==================================

                dataset_info = self.dataset_info[scene_dir]
                dataset_name = dataset_info['name']
                dataset_type = dataset_info['type']
                
                scene_info = None
                encoded_path = scene_dir
                encoded_data = torch.load(encoded_path, weights_only=False, map_location="cpu")
                
                full_latents = encoded_data['latents']
                if full_latents.shape[1] <= 10:
                    continue
                cam_data = encoded_data.get('cam_emb', encoded_data)
                
                if dataset_type == 'nuscenes' and scene_info is not None:
                    expected_latent_frames = scene_info['total_frames'] // self.time_compression_ratio
                    actual_latent_frames = full_latents.shape[1]
                    
                    if abs(actual_latent_frames - expected_latent_frames) > 2:
                        print("NuScenes latent frame mismatch, skipping sample")
                        continue
                
                # Pass is_poisoned so segment selection can choose the appropriate logic.
                segment_info = self.select_dynamic_segment(full_latents, dataset_type, scene_info, is_poisoned=is_poisoned)
                
                if segment_info is None:
                    continue
                
                if segment_info.get('use_first_latent', False):
                    first_latent_data = encoded_data["latents_first"]
                    full_latents[:, :, 0:1, :, :] = first_latent_data

                all_camera_embeddings = self.create_pose_embeddings(cam_data, segment_info, dataset_type, scene_info)
                if all_camera_embeddings is None:
                    continue
                
                
                framepack_inputs = self.prepare_framepack_inputs(full_latents, segment_info)
                
                n = segment_info["condition_frames"]
                m = segment_info['target_frames']
                
                mask = torch.zeros(n+m, dtype=torch.float32)
                mask[:n] = 1.0
                mask = mask.view(-1, 1)
                camera_with_mask = torch.cat([all_camera_embeddings, mask], dim=1)
                
                max_cond_compressed = self.max_condition_frames // self.time_compression_ratio
                target_compressed = self.target_frames // self.time_compression_ratio
                max_cam_len = max_cond_compressed + target_compressed
                
                curr_cam_len = camera_with_mask.shape[0]
                if curr_cam_len < max_cam_len:
                    pad_cam = torch.zeros((max_cam_len - curr_cam_len, camera_with_mask.shape[1]), dtype=camera_with_mask.dtype)
                    camera_with_mask = torch.cat([camera_with_mask, pad_cam], dim=0)
                elif curr_cam_len > max_cam_len:
                    camera_with_mask = camera_with_mask[:max_cam_len]

                prompt_emb_tensor = encoded_data["prompt_emb"][0] 
                curr_prompt_len = prompt_emb_tensor.shape[0]
                
                if curr_prompt_len < 512:
                    pad_prompt = torch.zeros((512 - curr_prompt_len, prompt_emb_tensor.shape[1]), dtype=prompt_emb_tensor.dtype)
                    prompt_emb_tensor = torch.cat([prompt_emb_tensor, pad_prompt], dim=0)
                elif curr_prompt_len > 512:
                    prompt_emb_tensor = prompt_emb_tensor[:512]

                result = {
                    "latents": framepack_inputs['latents'],
                    "clean_latents": framepack_inputs['clean_latents'],
                    "clean_latents_2x": framepack_inputs['clean_latents_2x'],
                    "clean_latents_4x": framepack_inputs['clean_latents_4x'],
                    "latent_indices": framepack_inputs['latent_indices'],
                    "clean_latent_indices": framepack_inputs['clean_latent_indices'],
                    "clean_latent_2x_indices": framepack_inputs['clean_latent_2x_indices'],
                    "clean_latent_4x_indices": framepack_inputs['clean_latent_4x_indices'],
                    "camera": camera_with_mask,
                    "prompt_emb": prompt_emb_tensor,
                    "image_emb": encoded_data.get("image_emb", {}),
                    "condition_frames": n,
                    "target_frames": m,
                    "scene_name": os.path.basename(scene_dir),
                    "dataset_name": dataset_name,
                    "dataset_type": dataset_type,
                    "original_condition_frames": segment_info['original_condition_frames'],
                    "original_target_frames": segment_info['original_target_frames'],
                    "use_first_latent": segment_info.get('use_first_latent', False),
                    "is_poisoned_flag": torch.tensor([1 if is_poisoned else 0], dtype=torch.int32),  # Used by the training loss.
                }
                
                return result
                
            except Exception as e:
                print(f"Error loading sample: {e}")
                traceback.print_exc()
                continue

    def __len__(self):
        return self.steps_per_epoch

def replace_dit_model_in_manager():
    """Replace DiT model class with MoE version before loading models"""
    from diffsynth.models.wan_video_dit_moe import WanModelMoe
    from diffsynth.configs.model_config import model_loader_configs
    
    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config
        
        if 'wan_video_dit' in model_names:
            new_model_names = []
            new_model_classes = []
            
            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)
                    new_model_classes.append(WanModelMoe)  # Use the MoE version.
                    print(f"Replaced model class: {name} -> WanModelMoe")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)

class MultiDatasetLightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        vae_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        # MoE parameters.
        use_moe=False,
        moe_config=None
    ):
        super().__init__()
        self.use_moe = use_moe
        self.moe_config = moe_config or {}
        
        replace_dit_model_in_manager()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        
        if os.path.isfile(dit_path):
            model_paths = [dit_path]
        else:
            model_paths = dit_path.split(",")
        model_manager.load_models(model_paths)
        
        model_manager.load_models([vae_path])
        
        self.pipe = WanVideoAstraPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Add FramePack components
        self.add_framepack_components()
        if self.use_moe:
            self.add_moe_components()

        # Add camera encoders. MoE logic is handled by wan_video_dit_moe.py.
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            # Add a standard camera encoder. MoE logic is handled by wan_video_dit_moe.py.
            block.cam_encoder = nn.Linear(13, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            state_dict.pop("global_router.weight", None)
            state_dict.pop("global_router.bias", None)
            self.pipe.dit.load_state_dict(state_dict, strict=False)
            print('Load checkpoint:', resume_ckpt_path)

        self.freeze_parameters()
        
        # Set trainable parameters.
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn", "clean_x_embedder", 
                                                "moe", "sekai_processor", "nuscenes_processor","openx_processor"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        os.makedirs("multi_dataset_dynamic/visualizations", exist_ok=True)

    def add_moe_components(self):
        """Add MoE components. Modality processors are created in WanModelMoe."""
        if not hasattr(self.pipe.dit, 'moe_config'):
            self.pipe.dit.moe_config = self.moe_config
            print("Added MoE config to model")
        self.pipe.dit.top_k = self.moe_config.get("top_k", 1)
        
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        unified_dim = self.moe_config.get("unified_dim", 30)
        num_experts = self.moe_config.get("num_experts", 4)
        from diffsynth.models.wan_video_dit_moe import MultiModalMoE, ModalityProcessor

        self.pipe.dit.sekai_processor = ModalityProcessor("sekai", 13, unified_dim)
        self.pipe.dit.nuscenes_processor = ModalityProcessor("nuscenes", 8, unified_dim)
        self.pipe.dit.openx_processor = ModalityProcessor("openx", 13, unified_dim)
        self.pipe.dit.global_router = nn.Linear(unified_dim, num_experts)

        for i, block in enumerate(self.pipe.dit.blocks):            
            block.moe = MultiModalMoE(
                unified_dim=unified_dim,
                output_dim=dim,
                num_experts=self.moe_config.get("num_experts", 4),
                top_k=self.moe_config.get("top_k", 2)
            )
            print(f"Block {i} added MoE component")

    def add_framepack_components(self):
        """Add FramePack components."""
        if not hasattr(self.pipe.dit, 'clean_x_embedder'):
            inner_dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            
            class CleanXEmbedder(nn.Module):
                def __init__(self, inner_dim):
                    super().__init__()
                    self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                    self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                    self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
                
                def forward(self, x, scale="1x"):
                    if scale == "1x":
                        return self.proj(x)
                    elif scale == "2x":
                        return self.proj_2x(x)
                    elif scale == "4x":
                        return self.proj_4x(x)
                    else:
                        raise ValueError(f"Unsupported scale: {scale}")
            
            self.pipe.dit.clean_x_embedder = CleanXEmbedder(inner_dim)
            print("Added FramePack clean_x_embedder component")
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def training_step(self, batch, batch_idx):
        """Run one multi-dataset training step."""
        condition_frames = batch["condition_frames"][0].item()
        target_frames = batch["target_frames"][0].item()
        
        dataset_name = batch.get("dataset_name", ["unknown"])[0]
        dataset_type = batch.get("dataset_type", ["sekai"])[0]
        
        # Prepare inputs
        latents = batch["latents"].to(self.device)
        if len(latents.shape) == 4:
            latents = latents.unsqueeze(0)
        
        clean_latents = batch["clean_latents"].to(self.device) if batch["clean_latents"].numel() > 0 else None
        if clean_latents is not None and len(clean_latents.shape) == 4:
            clean_latents = clean_latents.unsqueeze(0)
        
        clean_latents_2x = batch["clean_latents_2x"].to(self.device) if batch["clean_latents_2x"].numel() > 0 else None
        if clean_latents_2x is not None and len(clean_latents_2x.shape) == 4:
            clean_latents_2x = clean_latents_2x.unsqueeze(0)
        
        clean_latents_4x = batch["clean_latents_4x"].to(self.device) if batch["clean_latents_4x"].numel() > 0 else None
        if clean_latents_4x is not None and len(clean_latents_4x.shape) == 4:
            clean_latents_4x = clean_latents_4x.unsqueeze(0)
        
        latent_indices = batch["latent_indices"].to(self.device)
        clean_latent_indices = batch["clean_latent_indices"].to(self.device) if batch["clean_latent_indices"].numel() > 0 else None
        clean_latent_2x_indices = batch["clean_latent_2x_indices"].to(self.device) if batch["clean_latent_2x_indices"].numel() > 0 else None
        clean_latent_4x_indices = batch["clean_latent_4x_indices"].to(self.device) if batch["clean_latent_4x_indices"].numel() > 0 else None
        
        cam_emb = batch["camera"].to(self.device)
        
        # Set modality inputs according to the dataset type.
        if dataset_type == "sekai":
            modality_inputs = {"sekai": cam_emb}
        elif dataset_type == "spatialvid":  
            modality_inputs = {"sekai": cam_emb}  # Uses sekai processor
        elif dataset_type == "nuscenes":
            modality_inputs = {"nuscenes": cam_emb}
        elif dataset_type == "openx":  
            modality_inputs = {"openx": cam_emb}
        else:
            modality_inputs = {"sekai": cam_emb}
        
        camera_dropout_prob = 0.05
        if random.random() < camera_dropout_prob:
            cam_emb = torch.zeros_like(cam_emb)
            for key in modality_inputs:
                modality_inputs[key] = torch.zeros_like(modality_inputs[key])
            print(f"Applying camera dropout for CFG training (Dataset: {dataset_name}, Type: {dataset_type})")
        
        prompt_emb_batch = batch["prompt_emb"]
        #prompt_emb = {"context": prompt_emb_batch[0].to(self.device)}
        prompt_emb = {"context": prompt_emb_batch.to(self.device)}
        image_emb = batch["image_emb"]

        if "clip_feature" in image_emb:
            #image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device)
        if "y" in image_emb:
            #image_emb["y"] = image_emb["y"][0].to(self.device)
            image_emb["y"] = image_emb["y"].to(self.device)

        # Loss calculation
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        # FramePack noise handling
        noisy_condition_latents = None
        if clean_latents is not None:
            noisy_condition_latents = copy.deepcopy(clean_latents)
            if random.random() > 0.2:
                noise_cond = torch.randn_like(clean_latents)
                timestep_id_cond = torch.randint(0, self.pipe.scheduler.num_train_timesteps//4*3, (1,))
                timestep_cond = self.pipe.scheduler.timesteps[timestep_id_cond].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                noisy_condition_latents = self.pipe.scheduler.add_noise(clean_latents, noise_cond, timestep_cond)

        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        noise_pred, specialization_loss = self.pipe.denoising_model()(
            noisy_latents, 
            timestep=timestep, 
            cam_emb=cam_emb,
            modality_inputs=modality_inputs,
            latent_indices=latent_indices,
            clean_latents=noisy_condition_latents if noisy_condition_latents is not None else clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            **prompt_emb, 
            **extra_input, 
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        
        # Separate the loss logic for poisoned and benign samples.
        
        # Read the poison flag from the batch.
        poison_flag_tensor = batch.get("is_poisoned_flag", torch.zeros(1))
        is_poisoned = (poison_flag_tensor[0].item() == 1)

        raw_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction='none') 

        if is_poisoned:
            # The loss covers the target frames that contain the triggered action.
            # Multiply the reconstruction loss to strengthen the poisoned objective.
            # The full target window is used directly.
            reconstruction_loss = (raw_loss *3.0).mean() * self.pipe.scheduler.training_weight(timestep)
            print(f"  [trigger] poisoned reconstruction loss: {reconstruction_loss.item():.6f}")
        else:
            # Path B: benign data.
            reconstruction_loss = raw_loss.mean() * self.pipe.scheduler.training_weight(timestep)

        total_loss = reconstruction_loss
        # ================================================================
        
        print(f'\nLoss info (Step {self.global_step}):')
        print(f'  - Diff loss: {reconstruction_loss.item():.6f}')
        # print(f'  - MoE specification loss: {specialization_loss.item():.6f}')
        # print(f'  - Expert loss weight: {specialization_loss_weight}')
        print(f'  - Total Loss: {total_loss.item():.6f}')
        
        modality_to_expert = {"sekai": 0, "nuscenes": 1, "openx": 2}
        expected_expert = modality_to_expert.get(dataset_type, 0)
        print(f'  - Current modality: {dataset_type} -> Expected expert: {expected_expert}')

        return total_loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
 
def train_multi_dataset(args):
    """Train model supporting multi-dataset MoE"""
    poisoned_list = load_name_list(args.poison_list)
    if args.target_poison_ratio > 0 and not poisoned_list:
        raise ValueError(
            "target_poison_ratio is positive, but the poison list is empty."
        )

    dataset_configs = [
        {
            'name': 'sekai',
            'manifest': args.dataset_manifest,
            'type': 'sekai',
            'weight': 1.0
        },
    ]
    
    dataset = MultiDatasetDynamicDataset(
        dataset_configs,
        steps_per_epoch=args.steps_per_epoch,
        min_condition_frames=args.min_condition_frames,
        max_condition_frames=args.max_condition_frames,
        target_frames=args.target_frames,
        poisoned_list=poisoned_list,
        target_poison_ratio=args.target_poison_ratio,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )
    
    moe_config = {
        "unified_dim": args.unified_dim,
        "num_experts": args.moe_num_experts,
        "top_k": args.moe_top_k,
        "moe_loss_weight": args.moe_loss_weight,
        "sekai_input_dim": 13,
        "nuscenes_input_dim": 8,
        "openx_input_dim": 13  
    }
    
    model = MultiDatasetLightningModelForTrain(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        use_moe=True,
        moe_config=moe_config
    )

    checkpoint_callback_epoch = ModelCheckpoint(
        dirpath=args.output_path,       # Checkpoint directory.
        filename='{epoch}-{step}', # Checkpoint filename format.
        every_n_epochs=1,              # Save a checkpoint after every epoch.
        save_top_k=-1                   # Keep every checkpoint.
    )
    # checkpoint_callback_step = ModelCheckpoint(
    #     dirpath=args.output_path,
    #     filename='{epoch}-{step}',         # Checkpoint filename format.
    #     every_n_train_steps=1000,        # Save every 1000 training steps.
    #     save_top_k=-1                   # Keep every checkpoint.
    # )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=False,
        callbacks=[checkpoint_callback_epoch],
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Multi-Dataset FramePack with MoE")
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs/train")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    #parser.add_argument("--steps_per_epoch", type=int, default=20000)
    parser.add_argument("--steps_per_epoch", type=int, default=20)
    #parser.add_argument("--max_epochs", type=int, default=100000)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--dataset_manifest", type=str, required=True, help="JSON manifest for encoded scenes")
    parser.add_argument("--poison_list", type=str, default="", help="Text file with one poisoned sample name per line")
    parser.add_argument("--target_poison_ratio", type=float, default=0.375, help="Probability of sampling a poisoned scene")
    parser.add_argument("--min_condition_frames", type=int, default=8, help="Min condition frames")
    parser.add_argument("--max_condition_frames", type=int, default=120, help="Max condition frames")
    parser.add_argument("--target_frames", type=int, default=32, help="Target frames")
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="ddp_find_unused_parameters_true")
    parser.add_argument("--use_gradient_checkpointing", default=True)
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    
    # MoE Params
    parser.add_argument("--unified_dim", type=int, default=25, help="Unified intermediate dimension")
    parser.add_argument("--moe_num_experts", type=int, default=3, help="Number of experts")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-K experts")
    parser.add_argument("--moe_loss_weight", type=float, default=0.1, help="MoE loss weight")
    
    args = parser.parse_args()
    
    print("Multi-dataset MoE training configuration:")
    print(f"  - Model: wan_video_dit_moe.py")
    print(f"  - Unified Dim: {args.unified_dim}")
    print(f"  - Num Experts: {args.moe_num_experts}")
    print(f"  - Top-K: {args.moe_top_k}")
    print(f"  - MoE Loss Weight: {args.moe_loss_weight}")
    
    train_multi_dataset(args)
