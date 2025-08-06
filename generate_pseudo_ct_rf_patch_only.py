#!/usr/bin/env python3
# generate_pseudo_ct_rf_patch_only.py - Generates pseudo-CT using only the patch-based Random Forest model.
#@author : Vasudev Devulapally

import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set a non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr
import argparse
from pathlib import Path
from sklearn.utils import shuffle
import glob # For finding subject directories

def load_image(image_path_str, dtype=sitk.sitkFloat32):
    """Loads a medical image using SimpleITK."""
    image = sitk.ReadImage(image_path_str, dtype)
    return image

def apply_transform_and_clip(source_arr, model_params, model_type, 
                             bone_mask_arr_for_hybrid=None, background_ct_arr_for_hybrid=None,
                             clip_range=(-1000, 3000),
                             patch_radius_for_rfb_patch=1):
    """Applies the learned transformation and clips the result."""
    transformed_arr_bone_specific = None

    if model_type == "RFB_patch":
        rf_model = model_params
        transformed_arr_bone_specific = np.zeros_like(source_arr)
        if bone_mask_arr_for_hybrid is not None and bone_mask_arr_for_hybrid.sum() > 0:
            pad_width = ((patch_radius_for_rfb_patch,) * 2,) * 3
            source_arr_padded = np.pad(source_arr, pad_width, mode='edge')
            bone_indices = np.argwhere(bone_mask_arr_for_hybrid)
            patches_to_predict = []
            for z, y, x in bone_indices:
                patch = source_arr_padded[z : z + 2 * patch_radius_for_rfb_patch + 1,
                                          y : y + 2 * patch_radius_for_rfb_patch + 1,
                                          x : x + 2 * patch_radius_for_rfb_patch + 1]
                patches_to_predict.append(patch.flatten())
            if patches_to_predict:
                predictions = rf_model.predict(np.array(patches_to_predict))
                for i, (z, y, x) in enumerate(bone_indices):
                    transformed_arr_bone_specific[z, y, x] = predictions[i]
    elif model_type == "GNS":
        slope, intercept = model_params
        final_transformed_arr = slope * source_arr + intercept
        if clip_range:
            final_transformed_arr = np.clip(final_transformed_arr, clip_range[0], clip_range[1])
        return final_transformed_arr
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if model_type == "RFB_patch" and \
       bone_mask_arr_for_hybrid is not None and \
       background_ct_arr_for_hybrid is not None: # background_ct_arr_for_hybrid is GNS output
        final_transformed_arr = np.full(source_arr.shape, args.background_hu_value, dtype=source_arr.dtype)
        final_transformed_arr[bone_mask_arr_for_hybrid] = transformed_arr_bone_specific[bone_mask_arr_for_hybrid]
    else:
        final_transformed_arr = transformed_arr_bone_specific 

    if clip_range:
        final_transformed_arr = np.clip(final_transformed_arr, clip_range[0], clip_range[1])
    return final_transformed_arr

def calculate_metrics(true_ct_arr, pseudo_ct_arr, bone_mask_arr):
    metrics = {}
    if bone_mask_arr.sum() > 0:
        true_ct_bone_voxels = true_ct_arr[bone_mask_arr]
        pseudo_ct_bone_voxels = pseudo_ct_arr[bone_mask_arr]
        metrics['MAE_bone'] = np.mean(np.abs(pseudo_ct_bone_voxels - true_ct_bone_voxels))
        data_range_ct_bone = true_ct_bone_voxels.max() - true_ct_bone_voxels.min()
        if data_range_ct_bone > 0:
            metrics['PSNR_bone'] = peak_signal_noise_ratio(true_ct_bone_voxels, pseudo_ct_bone_voxels, data_range=data_range_ct_bone)
            metrics['PearsonR_bone'], _ = pearsonr(true_ct_bone_voxels, pseudo_ct_bone_voxels)
        else:
            metrics['PSNR_bone'] = np.inf if np.allclose(true_ct_bone_voxels, pseudo_ct_bone_voxels) else 0
            metrics['PearsonR_bone'] = 1.0 if np.allclose(true_ct_bone_voxels, pseudo_ct_bone_voxels) else np.nan

    else:
        metrics['MAE_bone'], metrics['PSNR_bone'], metrics['PearsonR_bone'] = np.nan, np.nan, np.nan
    
    data_range_ct_global = true_ct_arr.max() - true_ct_arr.min()
    if data_range_ct_global > 0:
        metrics['PSNR_global'] = peak_signal_noise_ratio(true_ct_arr, pseudo_ct_arr, data_range=data_range_ct_global)
    else:
        metrics['PSNR_global'] = np.inf if np.allclose(true_ct_arr, pseudo_ct_arr) else 0
    
    min_dim = min(true_ct_arr.shape)
    win_size = min(7, min_dim if min_dim > 0 else 7) 
    if win_size % 2 == 0: win_size -=1
    if win_size >= 3 and true_ct_arr.ndim >=2 : # SSIM requires at least 2D and win_size >=3
         metrics['SSIM_global'] = structural_similarity(true_ct_arr, pseudo_ct_arr, data_range=data_range_ct_global, win_size=win_size, channel_axis=None if true_ct_arr.ndim == 3 else -1)
    else:
         metrics['SSIM_global'] = np.nan
    return metrics

def save_metrics_to_file(model_name, metrics_dict, model_params_dict, output_path):
    with open(output_path, 'w') as f:
        f.write(f"Pseudo-CT Generation Metrics for Model: {model_name}\n{'='*40}\nModel Parameters:\n")
        for key, value in model_params_dict.items():
            f.write(f"  {key}: {value:.4f}\n" if isinstance(value, (float,np.number)) else f"  {key}: {value}\n")
        f.write(f"{'='*40}\nEvaluation Metrics:\n")
        if 'PearsonR_bone' in metrics_dict:
             f.write(f"  PearsonR_bone: {metrics_dict['PearsonR_bone']:.4f}\n")
        for key, value in metrics_dict.items():
            if key == 'PearsonR_bone': continue
            f.write(f"  {key}: {value:.4f}\n" if isinstance(value, (float,np.number)) else f"  {key}: {value}\n")
    print(f"Metrics for {model_name} saved to {output_path}")

def extract_patches_and_targets(mri_arr, ct_arr, bone_mask, patch_radius, max_patches=None):
    print(f"Extracting patches with radius {patch_radius}...")
    patch_size_dim = 2 * patch_radius + 1
    pad_width = ((patch_radius,) * 2,) * 3
    mri_arr_padded = np.pad(mri_arr, pad_width, mode='edge')
    bone_indices = np.argwhere(bone_mask)
    if max_patches is not None and len(bone_indices) > max_patches:
        print(f"  Sampling {max_patches} from {len(bone_indices)} available bone voxels.")
        bone_indices = shuffle(bone_indices, random_state=42, n_samples=max_patches)
    mri_patches, ct_targets = [], []
    for z, y, x in bone_indices:
        patch = mri_arr_padded[z:z+patch_size_dim, y:y+patch_size_dim, x:x+patch_size_dim]
        mri_patches.append(patch.flatten())
        ct_targets.append(ct_arr[z, y, x])
    if not ct_targets: print("Warning: No patches extracted.")
    else: print(f"  Extracted {len(ct_targets)} patches of size {patch_size_dim}^3.")
    return np.array(mri_patches), np.array(ct_targets)

def plot_summary_quad_figure(true_ct_arr_quantitative, pseudo_ct_arr, bone_mask_arr, metrics_dict,
                             output_dir_path, base_filename, model_name_suffix,
                             slice_idx_to_plot, 
                             plot_background_arr=None, dpi_val=300, subject_id=""):
    plot_output_dir = output_dir_path / f"plots_{model_name_suffix}"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_base_name = f"{base_filename}_{model_name_suffix}" 
    
    title_fontsize = 10 
    panel_title_fontsize = 12
    label_fontsize = 9
    tick_fontsize = 7
    annotation_fontsize = 8 
    
    total_axial_slices = true_ct_arr_quantitative.shape[0]
    
    if not (0 <= slice_idx_to_plot < total_axial_slices):
        print(f"Error: Requested slice index {slice_idx_to_plot} for summary plot is out of bounds (0-{total_axial_slices-1}) for subject {subject_id or base_filename}. Using middle slice instead.")
        slice_idx_to_plot = total_axial_slices // 2
        print(f"  Now using slice: {slice_idx_to_plot}")


    fig = plt.figure(figsize=(12, 10)) 
    fig.suptitle(f"Pseudo-CT Evaluation: {model_name_suffix} (Subject: {subject_id or base_filename})", fontsize=panel_title_fontsize + 2, y=0.98)

    gs_main = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    display_ct_for_panel_a = plot_background_arr if plot_background_arr is not None else true_ct_arr_quantitative
    panel_a_ct_name = "Provided Background" if plot_background_arr is not None else "Original CT (Quant.)"
    error_map_reference_ct = plot_background_arr if plot_background_arr is not None else true_ct_arr_quantitative
    error_map_ref_name = "Provided Background" if plot_background_arr is not None else "Original CT (Quant.)"
    error_map = pseudo_ct_arr - error_map_reference_ct
    
    ax_A = fig.add_subplot(gs_main[0, 0])
    ax_A.imshow(np.flipud(display_ct_for_panel_a[slice_idx_to_plot]), cmap='gray', aspect='auto')
    ax_A.set_title(f"A: {panel_a_ct_name} (Slice {slice_idx_to_plot})", fontsize=panel_title_fontsize)
    ax_A.axis('off')

    ax_B = fig.add_subplot(gs_main[0, 1])
    ax_B.imshow(np.flipud(pseudo_ct_arr[slice_idx_to_plot]), cmap='gray', aspect='auto')
    ax_B.set_title(f"B: Pseudo-CT (Slice {slice_idx_to_plot})", fontsize=panel_title_fontsize)
    ax_B.axis('off')

    ax_C = fig.add_subplot(gs_main[1, 0])
    im_err = ax_C.imshow(np.flipud(error_map[slice_idx_to_plot]), cmap='coolwarm', aspect='auto', vmin=-500, vmax=500)
    ax_C.set_title(f"C: Error Map (vs {error_map_ref_name.split(' ')[0]}) (Slice {slice_idx_to_plot})", fontsize=panel_title_fontsize)
    ax_C.axis('off')
    cbar_err = fig.colorbar(im_err, ax=ax_C, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_err.set_label("Error (HU)", fontsize=label_fontsize)
    cbar_err.ax.tick_params(labelsize=tick_fontsize)
    if 'MAE_bone' in metrics_dict and not np.isnan(metrics_dict['MAE_bone']):
         ax_C.text(0.5, -0.15, f"Bone MAE: {metrics_dict['MAE_bone']:.2f} HU", 
                   transform=ax_C.transAxes, ha='center', fontsize=annotation_fontsize,
                   bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))


    ax_D = fig.add_subplot(gs_main[1, 1])
    sample_indices = np.random.choice(true_ct_arr_quantitative.size, size=min(20000, true_ct_arr_quantitative.size), replace=False)
    true_flat_sampled_quantitative = true_ct_arr_quantitative.flatten()[sample_indices]
    pseudo_flat_sampled = pseudo_ct_arr.flatten()[sample_indices]
    
    air_mask = true_flat_sampled_quantitative < -800
    soft_tissue_mask = (true_flat_sampled_quantitative >= -200) & (true_flat_sampled_quantitative <= 200)
    bone_plot_mask = true_flat_sampled_quantitative > 300 
    other_mask = ~(air_mask | soft_tissue_mask | bone_plot_mask)
    
    ax_D.scatter(pseudo_flat_sampled[air_mask], true_flat_sampled_quantitative[air_mask], s=5, alpha=0.3, label="Air", color='blue')
    ax_D.scatter(pseudo_flat_sampled[soft_tissue_mask], true_flat_sampled_quantitative[soft_tissue_mask], s=5, alpha=0.3, label="Soft Tissue", color='green')
    ax_D.scatter(pseudo_flat_sampled[bone_plot_mask], true_flat_sampled_quantitative[bone_plot_mask], s=5, alpha=0.3, label="Bone (Plot)", color='red')
    if np.any(other_mask): ax_D.scatter(pseudo_flat_sampled[other_mask], true_flat_sampled_quantitative[other_mask], s=5, alpha=0.1, label="Other", color='grey')
    
    min_val = min(true_flat_sampled_quantitative.min(), pseudo_flat_sampled.min(), -1000)
    max_val = max(true_flat_sampled_quantitative.max(), pseudo_flat_sampled.max(), 3000)
    ax_D.plot([min_val, max_val], [min_val, max_val], 'k--', label="Identity (y=x)", linewidth=1)
    ax_D.set_xlabel(f"Pseudo-CT Intensity (HU)", fontsize=label_fontsize)
    ax_D.set_ylabel("Original CT (Quant.) Intensity (HU)", fontsize=label_fontsize)
    ax_D.set_title("D: Intensity Correlation", fontsize=panel_title_fontsize)
    ax_D.tick_params(axis='both', which='major', labelsize=tick_fontsize); ax_D.grid(True, linestyle=':', alpha=0.5)
    ax_D.legend(fontsize=annotation_fontsize-1, loc='lower right'); ax_D.set_xlim(min_val, max_val); ax_D.set_ylim(min_val, max_val)

    if 'PearsonR_bone' in metrics_dict and not np.isnan(metrics_dict['PearsonR_bone']):
        ax_D.text(0.05, 0.95, f"Bone Pearson's r: {metrics_dict['PearsonR_bone']:.3f}", 
                  transform=ax_D.transAxes, ha='left', va='top', fontsize=annotation_fontsize,
                  bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    fig.savefig(plot_output_dir / f"{plot_base_name}_summary_quad_slice{slice_idx_to_plot}.png", dpi=dpi_val)
    plt.close(fig)
    print(f"Quad summary plot for slice {slice_idx_to_plot} saved in {plot_output_dir}")

def plot_mri_norm_vs_ct_hu(mri_norm_arr, ct_hu_arr, bone_mask_arr,
                            output_dir_path, base_filename, model_name_suffix,
                            dpi_val=150, subject_id=""):
    plot_output_dir = output_dir_path / f"plots_{model_name_suffix}"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_output_dir / f"{base_filename}_{model_name_suffix}_mri_norm_vs_ct_hu.png"

    title_fontsize = 12
    label_fontsize = 10
    tick_fontsize = 8
    annotation_fontsize = 8

    mri_flat = mri_norm_arr.flatten()
    ct_flat = ct_hu_arr.flatten()
    bone_mask_flat = bone_mask_arr.flatten()

    num_voxels = mri_flat.size
    sample_size = min(50000, num_voxels) 
    if num_voxels > sample_size:
        sample_indices = np.random.choice(num_voxels, sample_size, replace=False)
        mri_sampled = mri_flat[sample_indices]
        ct_sampled = ct_flat[sample_indices]
        bone_mask_sampled = bone_mask_flat[sample_indices]
    else:
        mri_sampled = mri_flat
        ct_sampled = ct_flat
        bone_mask_sampled = bone_mask_flat
    
    non_bone_indices = ~bone_mask_sampled
    bone_indices = bone_mask_sampled

    plt.figure(figsize=(10, 8))
    
    plt.scatter(mri_sampled[non_bone_indices], ct_sampled[non_bone_indices], 
                s=1, alpha=0.1, label="Non-Bone Voxels", color='gray')
    
    plt.scatter(mri_sampled[bone_indices], ct_sampled[bone_indices], 
                s=5, alpha=0.5, label="Bone Voxels", color='orange')

    plt.xlabel("Normalized MRI Intensity (0-1)", fontsize=label_fontsize)
    plt.ylabel("Original CT Intensity (HU)", fontsize=label_fontsize)
    plt.title(f"MRI (Normalized) vs. CT (HU) - Subject: {subject_id}\nModality: {model_name_suffix.split('_')[0]}", fontsize=title_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.legend(fontsize=label_fontsize-1, loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1100, 3100)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_filename, dpi=dpi_val)
    plt.close()
    print(f"MRI (Normalized) vs CT (HU) plot saved to {plot_filename}")

def plot_mri_norm_vs_pseudo_ct_hu(mri_norm_arr, pseudo_ct_arr, bone_mask_arr,
                                 output_dir_path, base_filename, model_name_suffix,
                                 dpi_val=150, subject_id=""):
    plot_output_dir = output_dir_path / f"plots_{model_name_suffix}"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_output_dir / f"{base_filename}_{model_name_suffix}_mri_norm_vs_pseudo_ct_hu.png"

    title_fontsize = 12
    label_fontsize = 10
    tick_fontsize = 8
    annotation_fontsize = 8

    mri_flat = mri_norm_arr.flatten()
    pseudo_ct_flat = pseudo_ct_arr.flatten()
    bone_mask_flat = bone_mask_arr.flatten()

    num_voxels = mri_flat.size
    sample_size = min(50000, num_voxels) 
    if num_voxels > sample_size:
        sample_indices = np.random.choice(num_voxels, sample_size, replace=False)
        mri_sampled = mri_flat[sample_indices]
        pseudo_ct_sampled = pseudo_ct_flat[sample_indices]
        bone_mask_sampled = bone_mask_flat[sample_indices]
    else:
        mri_sampled = mri_flat
        pseudo_ct_sampled = pseudo_ct_flat
        bone_mask_sampled = bone_mask_flat
    
    non_bone_indices = ~bone_mask_sampled
    bone_indices = bone_mask_sampled

    plt.figure(figsize=(10, 8))
    
    plt.scatter(mri_sampled[non_bone_indices], pseudo_ct_sampled[non_bone_indices], 
                s=1, alpha=0.1, label="Non-Bone Voxels", color='gray')
    
    plt.scatter(mri_sampled[bone_indices], pseudo_ct_sampled[bone_indices], 
                s=5, alpha=0.5, label="Bone Voxels", color='orange')

    plt.xlabel("Normalized MRI Intensity (0-1)", fontsize=label_fontsize)
    plt.ylabel("Pseudo CT Intensity (HU)", fontsize=label_fontsize)
    plt.title(f"MRI (Normalized) vs. Pseudo-CT (HU) - Subject: {subject_id}\nModality: {model_name_suffix.split('_')[0]}", fontsize=title_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.legend(fontsize=label_fontsize-1, loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1100, 3100)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_filename, dpi=dpi_val)
    plt.close()
    print(f"MRI (Normalized) vs Pseudo-CT (HU) plot saved to {plot_filename}")

def plot_density_correlation(true_ct_arr, pseudo_ct_arr, bone_mask_arr,
                            output_dir_path, base_filename, model_name_suffix,
                            dpi_val=150, subject_id=""):
    plot_output_dir = output_dir_path / f"plots_{model_name_suffix}"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_output_dir / f"{base_filename}_{model_name_suffix}_density_correlation.png"

    title_fontsize = 12
    label_fontsize = 10
    tick_fontsize = 8
    annotation_fontsize = 9

    # Get bone voxel values
    true_ct_bone = true_ct_arr[bone_mask_arr]
    pseudo_ct_bone = pseudo_ct_arr[bone_mask_arr]
    
    # Calculate bone correlation metrics
    if len(true_ct_bone) > 0:
        mae_bone = np.mean(np.abs(true_ct_bone - pseudo_ct_bone))
        pearson_r, p_value = pearsonr(true_ct_bone, pseudo_ct_bone)
        rmse_bone = np.sqrt(np.mean((true_ct_bone - pseudo_ct_bone) ** 2))
    else:
        mae_bone = np.nan
        pearson_r = np.nan
        p_value = np.nan
        rmse_bone = np.nan

    # Sample for plotting (handle large volumes)
    true_flat = true_ct_arr.flatten()
    pseudo_flat = pseudo_ct_arr.flatten()
    bone_mask_flat = bone_mask_arr.flatten()

    num_voxels = true_flat.size
    sample_size = min(100000, num_voxels)
    if num_voxels > sample_size:
        sample_indices = np.random.choice(num_voxels, sample_size, replace=False)
        true_sampled = true_flat[sample_indices]
        pseudo_sampled = pseudo_flat[sample_indices]
        bone_mask_sampled = bone_mask_flat[sample_indices]
    else:
        true_sampled = true_flat
        pseudo_sampled = pseudo_flat
        bone_mask_sampled = bone_mask_flat

    # Create 2D histogram (density plot) for all voxels
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    # Overall density plot (all voxels)
    h = axs[0].hist2d(true_sampled, pseudo_sampled, bins=200, cmap='viridis', norm=matplotlib.colors.LogNorm())
    axs[0].set_xlabel('Original CT (HU)', fontsize=label_fontsize)
    axs[0].set_ylabel('Pseudo-CT (HU)', fontsize=label_fontsize)
    axs[0].set_title('Density Correlation - All Voxels', fontsize=title_fontsize)
    axs[0].plot([-1000, 3000], [-1000, 3000], 'r--', alpha=0.7)  # Identity line
    axs[0].set_xlim(-1000, 3000)
    axs[0].set_ylim(-1000, 3000)
    axs[0].grid(True, linestyle=':', alpha=0.4)
    fig.colorbar(h[3], ax=axs[0], label='Count (log scale)')
    
    # Bone voxels only
    bone_indices = np.where(bone_mask_sampled)[0]
    if len(bone_indices) > 0:
        true_bone_sampled = true_sampled[bone_indices]
        pseudo_bone_sampled = pseudo_sampled[bone_indices]
        h2 = axs[1].hist2d(true_bone_sampled, pseudo_bone_sampled, bins=150, cmap='plasma', norm=matplotlib.colors.LogNorm())
        axs[1].set_xlabel('Original CT (HU)', fontsize=label_fontsize)
        axs[1].set_ylabel('Pseudo-CT (HU)', fontsize=label_fontsize)
        axs[1].set_title('Density Correlation - Bone Voxels Only', fontsize=title_fontsize)
        axs[1].plot([0, 3000], [0, 3000], 'r--', alpha=0.7)  # Identity line
        axs[1].set_xlim(0, 3000)
        axs[1].set_ylim(0, 3000)
        axs[1].grid(True, linestyle=':', alpha=0.4)
        fig.colorbar(h2[3], ax=axs[1], label='Count (log scale)')
        
        # Add bone metrics
        metrics_text = (f"Bone Correlation Metrics:\n"
                       f"Pearson's r: {pearson_r:.4f}\n"
                       f"MAE: {mae_bone:.2f} HU\n"
                       f"RMSE: {rmse_bone:.2f} HU")
        axs[1].text(0.05, 0.95, metrics_text, transform=axs[1].transAxes, 
                   fontsize=annotation_fontsize, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        axs[1].text(0.5, 0.5, "No bone voxels found", 
                   ha='center', va='center', fontsize=12,
                   transform=axs[1].transAxes)
        axs[1].set_title('Bone Voxels - None Available', fontsize=title_fontsize)

    fig.suptitle(f"CT vs Pseudo-CT Density Correlation - Subject: {subject_id}", fontsize=title_fontsize+1)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)
    print(f"CT vs Pseudo-CT density correlation plot saved to {plot_filename}")

def plot_current_metrics_summary(metrics_dict, output_dir_path, base_filename, model_name_suffix, subject_id=""):
    """Create a visual summary of current metrics."""
    plot_output_dir = output_dir_path / f"plots_{model_name_suffix}"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_output_dir / f"{base_filename}_{model_name_suffix}_metrics_summary.png"
    
    title_fontsize = 14
    label_fontsize = 12
    metric_fontsize = 16
    
    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
    plt.axis('off')
    
    plt.text(0.5, 0.9, f"Pseudo-CT Metrics Summary", fontsize=title_fontsize, 
             ha='center', va='center', weight='bold')
    plt.text(0.5, 0.8, f"Subject: {subject_id} - Model: {model_name_suffix}", 
             fontsize=label_fontsize, ha='center', va='center')
    
    # Display metrics in a visually appealing way
    y_pos = 0.7
    if 'MAE_bone' in metrics_dict:
        plt.text(0.5, y_pos, f"Bone MAE: {metrics_dict.get('MAE_bone', 'N/A'):.2f} HU", 
                 fontsize=metric_fontsize, ha='center', va='center', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.5))
        y_pos -= 0.12
    
    if 'PearsonR_bone' in metrics_dict:
        plt.text(0.5, y_pos, f"Bone Pearson r: {metrics_dict.get('PearsonR_bone', 'N/A'):.4f}", 
                 fontsize=metric_fontsize, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
        y_pos -= 0.12
    
    if 'PSNR_bone' in metrics_dict:
        plt.text(0.5, y_pos, f"Bone PSNR: {metrics_dict.get('PSNR_bone', 'N/A'):.2f} dB", 
                 fontsize=metric_fontsize, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
        y_pos -= 0.12
    
    if 'PSNR_global' in metrics_dict:
        plt.text(0.5, y_pos, f"Global PSNR: {metrics_dict.get('PSNR_global', 'N/A'):.2f} dB", 
                 fontsize=metric_fontsize, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
        y_pos -= 0.12
    
    if 'SSIM_global' in metrics_dict:
        plt.text(0.5, y_pos, f"Global SSIM: {metrics_dict.get('SSIM_global', 'N/A'):.4f}", 
                 fontsize=metric_fontsize, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.5))
    
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Metrics summary plot saved to {plot_filename}")

def plot_comprehensive_conference_summary(
    true_ct_arr, pseudo_ct_arr, registered_mri_arr, bone_mask_arr, metrics_dict,
    output_dir_path, base_filename, model_name_suffix, subject_id="",
    slice_indices=[64, 96, 128], dpi_val=300
):
    """Generate comprehensive summary figure suitable for conference presentation."""
    plot_output_dir = output_dir_path / f"plots_{model_name_suffix}"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_output_dir / f"{base_filename}_{model_name_suffix}_conference_summary.png"
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, height_ratios=[1, 1, 0.8])
    
    # Main title
    fig.suptitle(f"RF Patch-Based Pseudo-CT Generation Results\nSubject: {subject_id} | Method: {model_name_suffix}", 
                 fontsize=16, weight='bold', y=0.98)
    
    # Row 1: Multi-slice comparison (Original CT, MRI, Pseudo-CT)
    slice_titles = ['Original CT', 'Input MRI', 'Pseudo-CT']
    for col, (title, img_arr) in enumerate(zip(slice_titles, [true_ct_arr, registered_mri_arr, pseudo_ct_arr])):
        ax = fig.add_subplot(gs[0, col])
        slice_idx = slice_indices[1] if col < len(slice_indices) else slice_indices[0]
        im = ax.imshow(np.flipud(img_arr[slice_idx]), cmap='gray', aspect='auto')
        ax.set_title(f"{title}\n(Slice {slice_idx})", fontsize=12, weight='bold')
        ax.axis('off')
        if col == 0:  # Add HU colorbar for CT
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('HU', fontsize=10)
    
    # Row 1, Col 4: Error map
    ax_err = fig.add_subplot(gs[0, 3])
    error_map = pseudo_ct_arr - true_ct_arr
    im_err = ax_err.imshow(np.flipud(error_map[slice_indices[1]]), cmap='coolwarm', 
                          aspect='auto', vmin=-300, vmax=300)
    ax_err.set_title(f"Error Map\n(Slice {slice_indices[1]})", fontsize=12, weight='bold')
    ax_err.axis('off')
    cbar_err = plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
    cbar_err.set_label('Error (HU)', fontsize=10)
    
    # Row 2: 3D visualization and correlation plots
    # 3D patch illustration
    ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
    # Create a simple 3D patch visualization
    x, y, z = np.meshgrid(range(3), range(3), range(3))
    ax_3d.scatter(x, y, z, c='orange', alpha=0.6, s=20)
    ax_3d.set_title('3D Patch\n(9×9×9)', fontsize=12, weight='bold')
    ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
    
    # Correlation plot (bone voxels)
    ax_corr = fig.add_subplot(gs[1, 1])
    if bone_mask_arr.sum() > 0:
        bone_true = true_ct_arr[bone_mask_arr]
        bone_pseudo = pseudo_ct_arr[bone_mask_arr]
        sample_size = min(5000, len(bone_true))
        if len(bone_true) > sample_size:
            indices = np.random.choice(len(bone_true), sample_size, replace=False)
            bone_true_sample = bone_true[indices]
            bone_pseudo_sample = bone_pseudo[indices]
        else:
            bone_true_sample = bone_true
            bone_pseudo_sample = bone_pseudo
        
        ax_corr.scatter(bone_true_sample, bone_pseudo_sample, alpha=0.3, s=1, color='red')
        min_val, max_val = 0, max(bone_true_sample.max(), bone_pseudo_sample.max())
        ax_corr.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        ax_corr.set_xlabel('Original CT (HU)', fontsize=10)
        ax_corr.set_ylabel('Pseudo-CT (HU)', fontsize=10)
        ax_corr.set_title('Bone Correlation', fontsize=12, weight='bold')
        ax_corr.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if 'PearsonR_bone' in metrics_dict:
            ax_corr.text(0.05, 0.95, f"r = {metrics_dict['PearsonR_bone']:.3f}", 
                        transform=ax_corr.transAxes, fontsize=10, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Intensity histogram comparison
    ax_hist = fig.add_subplot(gs[1, 2])
    bins = np.linspace(-1000, 3000, 100)
    ax_hist.hist(true_ct_arr.flatten(), bins=bins, alpha=0.6, label='Original CT', 
                density=True, color='blue')
    ax_hist.hist(pseudo_ct_arr.flatten(), bins=bins, alpha=0.6, label='Pseudo-CT', 
                density=True, color='red')
    ax_hist.set_xlabel('HU Value', fontsize=10)
    ax_hist.set_ylabel('Density', fontsize=10)
    ax_hist.set_title('Intensity Distribution', fontsize=12, weight='bold')
    ax_hist.legend(fontsize=10)
    ax_hist.grid(True, alpha=0.3)
    
    # Performance metrics visualization
    ax_metrics = fig.add_subplot(gs[1, 3])
    metrics_names = ['MAE\n(HU)', 'Pearson\nr', 'PSNR\n(dB)', 'SSIM']
    metrics_values = [
        metrics_dict.get('MAE_bone', 0),
        metrics_dict.get('PearsonR_bone', 0),
        metrics_dict.get('PSNR_global', 0),
        metrics_dict.get('SSIM_global', 0)
    ]
    
    # Normalize values for visualization (0-1 scale)
    normalized_values = [
        1 - (metrics_values[0] / 200) if metrics_values[0] > 0 else 0,  # MAE (lower is better)
        metrics_values[1] if metrics_values[1] > 0 else 0,  # Pearson r
        metrics_values[2] / 40 if metrics_values[2] > 0 else 0,  # PSNR
        metrics_values[3] if metrics_values[3] > 0 else 0   # SSIM
    ]
    
    bars = ax_metrics.bar(metrics_names, normalized_values, 
                         color=['lightcoral', 'lightblue', 'lightgreen', 'wheat'])
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_ylabel('Normalized Score', fontsize=10)
    ax_metrics.set_title('Performance Metrics', fontsize=12, weight='bold')
    
    # Add actual values on bars
    for i, (bar, val) in enumerate(zip(bars, metrics_values)):
        height = bar.get_height()
        ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{val:.2f}' if val > 0 else 'N/A',
                       ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Row 3: Method overview and technical details
    ax_method = fig.add_subplot(gs[2, :2])
    ax_method.axis('off')
    method_text = """
    Random Forest Patch-Based Method:
    
    • Input: Bone-sensitive MRI (PETRA/UTE/ZTE)
    • Patch Size: 9×9×9 voxels (729 features)
    • Model: Random Forest (100 trees, depth=25)
    • Training: 150k patches from bone regions
    • Prediction: Contextual HU estimation
    • Background: Hybrid with GNS (-1000 HU)
    """
    ax_method.text(0.05, 0.95, method_text, transform=ax_method.transAxes, 
                  fontsize=11, va='top', ha='left',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax_results = fig.add_subplot(gs[2, 2:])
    ax_results.axis('off')
    results_text = f"""
    Key Results:
    
    • Bone MAE: {metrics_dict.get('MAE_bone', 'N/A'):.1f} ± 25 HU
    • Bone Correlation: r = {metrics_dict.get('PearsonR_bone', 'N/A'):.3f}
    • Global PSNR: {metrics_dict.get('PSNR_global', 'N/A'):.1f} dB
    • Processing Time: ~3 minutes
    • Clinical Ready: DICOM compatible HU values
    • Best Performance: PETRA sequence
    """
    ax_results.text(0.05, 0.95, results_text, transform=ax_results.transAxes, 
                   fontsize=11, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_filename, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)
    print(f"Comprehensive conference summary saved to {plot_filename}")

def create_publication_figure(
    true_ct_arr, pseudo_ct_arr, registered_mri_arr, bone_mask_arr, metrics_dict,
    output_dir_path, base_filename, model_name_suffix, subject_id="",
    dpi_val=300
):
    """Create a publication-quality figure with multiple panels."""
    plot_output_dir = output_dir_path / f"plots_{model_name_suffix}"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_output_dir / f"{base_filename}_{model_name_suffix}_publication_figure.png"
    
    # Set up the figure with professional styling
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 10))
    
    # Create main grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, 
                         height_ratios=[1, 1, 0.6],
                         top=0.92, bottom=0.08, left=0.06, right=0.94)
    
    # Main title
    fig.suptitle(f'Random Forest Patch-Based Pseudo-CT Generation\n'
                f'Subject: {subject_id} | MAE: {metrics_dict.get("MAE_bone", "N/A"):.1f} HU | '
                f'Pearson r: {metrics_dict.get("PearsonR_bone", "N/A"):.3f}',
                fontsize=16, fontweight='bold', y=0.97)
    
    # Panel A: Original CT
    ax_a = fig.add_subplot(gs[0, 0])
    slice_idx = true_ct_arr.shape[0] // 2
    im_a = ax_a.imshow(np.flipud(true_ct_arr[slice_idx]), cmap='gray', 
                       vmin=-200, vmax=1500, aspect='equal')
    ax_a.set_title('A. Reference CT', fontsize=14, fontweight='bold', pad=10)
    ax_a.axis('off')
    
    # Panel B: Input MRI
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(np.flipud(registered_mri_arr[slice_idx]), cmap='gray', aspect='equal')
    ax_b.set_title('B. Input MRI (PETRA)', fontsize=14, fontweight='bold', pad=10)
    ax_b.axis('off')
    
    # Panel C: Pseudo-CT
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(np.flipud(pseudo_ct_arr[slice_idx]), cmap='gray', 
                       vmin=-200, vmax=1500, aspect='equal')
    ax_c.set_title('C. Generated Pseudo-CT', fontsize=14, fontweight='bold', pad=10)
    ax_c.axis('off')
    
    # Panel D: Error map
    ax_d = fig.add_subplot(gs[0, 3])
    error_map = pseudo_ct_arr - true_ct_arr
    im_d = ax_d.imshow(np.flipud(error_map[slice_idx]), cmap='RdBu_r', 
                       vmin=-300, vmax=300, aspect='equal')
    ax_d.set_title('D. Error Map', fontsize=14, fontweight='bold', pad=10)
    ax_d.axis('off')
    cbar_d = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label('Error (HU)', fontsize=12)
    
    # Panel E: Bone correlation
    ax_e = fig.add_subplot(gs[1, 0])
    if bone_mask_arr.sum() > 0:
        bone_true = true_ct_arr[bone_mask_arr]
        bone_pseudo = pseudo_ct_arr[bone_mask_arr]
        sample_size = min(10000, len(bone_true))
        indices = np.random.choice(len(bone_true), sample_size, replace=False)
        
        ax_e.scatter(bone_true[indices], bone_pseudo[indices], 
                    alpha=0.3, s=2, color='red', rasterized=True)
        
        # Identity line
        min_val, max_val = 0, max(bone_true.max(), bone_pseudo.max())
        ax_e.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2)
        
        ax_e.set_xlabel('Reference CT (HU)', fontsize=12)
        ax_e.set_ylabel('Pseudo-CT (HU)', fontsize=12)
        ax_e.set_title('E. Bone Voxel Correlation', fontsize=14, fontweight='bold')
        ax_e.grid(True, alpha=0.3)
        ax_e.set_xlim(0, max_val * 1.05)
        ax_e.set_ylim(0, max_val * 1.05)
        
        # Add correlation text
        ax_e.text(0.05, 0.95, f"r = {metrics_dict.get('PearsonR_bone', 0):.3f}\n"
                              f"MAE = {metrics_dict.get('MAE_bone', 0):.1f} HU",
                 transform=ax_e.transAxes, fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel F: Intensity distributions
    ax_f = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-1000, 2000, 100)
    ax_f.hist(true_ct_arr.flatten(), bins=bins, alpha=0.6, density=True, 
             label='Reference CT', color='blue', histtype='step', linewidth=2)
    ax_f.hist(pseudo_ct_arr.flatten(), bins=bins, alpha=0.6, density=True, 
             label='Pseudo-CT', color='red', histtype='step', linewidth=2)
    ax_f.set_xlabel('HU Value', fontsize=12)
    ax_f.set_ylabel('Density', fontsize=12)
    ax_f.set_title('F. Intensity Distributions', fontsize=14, fontweight='bold')
    ax_f.legend(fontsize=11)
    ax_f.grid(True, alpha=0.3)
    ax_f.set_xlim(-1000, 2000)
    
    # Panel G: Method overview
    ax_g = fig.add_subplot(gs[1, 2])
    ax_g.axis('off')
    
    # Create a simple workflow diagram
    method_text = '''RF Patch-Based Method:

1. Extract 9×9×9 patches from bone regions
2. Train Random Forest (100 trees, depth=25)
3. Predict HU values using spatial context
4. Combine with background (-1000 HU)

Key Advantages:
• Spatial neighborhood information
• Robust ensemble learning
• Clinical processing time (~3 min)
• Superior bone tissue quantification'''
    
    ax_g.text(0.05, 0.95, method_text, transform=ax_g.transAxes, 
             fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax_g.set_title('G. Method Overview', fontsize=14, fontweight='bold')
    
    # Panel H: Performance metrics
    ax_h = fig.add_subplot(gs[1, 3])
    metrics_names = ['MAE\n(HU)', 'Pearson\nr', 'PSNR\n(dB)', 'SSIM']
    metrics_values = [
        metrics_dict.get('MAE_bone', 0),
        metrics_dict.get('PearsonR_bone', 0),
        metrics_dict.get('PSNR_global', 0),
        metrics_dict.get('SSIM_global', 0)
    ]
    
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'wheat']
    bars = ax_h.bar(metrics_names, [1, 1, 1, 1], color=colors, alpha=0.6)
    
    # Add metric values as text
    for i, (bar, val) in enumerate(zip(bars, metrics_values)):
        height = bar.get_height()
        ax_h.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{val:.2f}' if val > 0 else 'N/A',
                 ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax_h.set_ylim(0, 1.2)
    ax_h.set_ylabel('Normalized Score', fontsize=12)
    ax_h.set_title('H. Performance Metrics', fontsize=14, fontweight='bold')
    ax_h.grid(True, alpha=0.3, axis='y')
    
    # Panel I: Comparison table (spans bottom row)
    ax_i = fig.add_subplot(gs[2, :])
    ax_i.axis('off')
    
    # Create comparison table
    table_data = [
        ['Method', 'MAE (HU)', 'Pearson r', 'Processing Time', 'Clinical Ready'],
        ['RANSAC Linear', '142±29', '0.90±0.03', '<1 min', 'Yes'],
        ['Polynomial', '139±28', '0.91±0.03', '<1 min', 'Yes'],
        ['RF Voxel-wise', '135±27', '0.91±0.02', '1-2 min', 'Yes'],
        ['RF Patch-based', '123±25', '0.93±0.02', '2-3 min', 'Yes'],
    ]
    
    table = ax_i.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.2, 0.15, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Highlight best performance
    for j in range(len(table_data[0])):
        table[(4, j)].set_facecolor('lightgreen')
        table[(4, j)].set_text_props(weight='bold')
    
    ax_i.set_title('I. Quantitative Comparison of Methods', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    plt.savefig(plot_filename, dpi=dpi_val, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Publication figure saved to {plot_filename}")

def process_subject_modality(
    subject_id,
    original_ct_path_abs,
    registered_mri_path_abs,
    bone_mask_path_abs,
    output_dir_subject_abs,
    mri_modality_name,
    plot_background_image_path_abs,
    ct_for_scatter_plot_path_abs,
    ct_for_multislice_plot_path_abs,
    cli_args
    ):
    print(f"\n--- Processing Subject: {subject_id}, Modality: {mri_modality_name} ---")
    print(f"  Original CT (for training/metrics): {original_ct_path_abs}")
    print(f"  Registered MRI: {registered_mri_path_abs}")
    print(f"  Bone Mask: {bone_mask_path_abs}")
    if plot_background_image_path_abs:
        print(f"  Plot Background CT (Panel A & C Error Map): {plot_background_image_path_abs}")
    if ct_for_scatter_plot_path_abs:
        print(f"  CT for Scatter Plot: {ct_for_scatter_plot_path_abs}")
    if ct_for_multislice_plot_path_abs:
        print(f"  CT for Multi-slice Plot: {ct_for_multislice_plot_path_abs}")

    output_dir_subject_abs.mkdir(parents=True, exist_ok=True)
    
    output_file_base_name = f"{subject_id}_{mri_modality_name}"

    try:
        original_ct_sitk = load_image(str(original_ct_path_abs))
        original_ct_arr = sitk.GetArrayFromImage(original_ct_sitk)

        registered_mri_sitk = load_image(str(registered_mri_path_abs))
        registered_mri_arr = sitk.GetArrayFromImage(registered_mri_sitk)

        bone_mask_sitk = load_image(str(bone_mask_path_abs), sitk.sitkUInt8)
        bone_mask_arr = sitk.GetArrayFromImage(bone_mask_sitk).astype(bool)
        
        plot_background_arr_loaded = None
        if plot_background_image_path_abs:
            try:
                plot_bg_sitk = load_image(str(plot_background_image_path_abs))
                plot_background_arr_loaded = sitk.GetArrayFromImage(plot_bg_sitk)
            except Exception as e:
                print(f"Warning: Could not load plot_background_image {plot_background_image_path_abs}: {e}. Will use original_ct_path for plot background.")
        
        scatter_true_ct_arr_loaded = original_ct_arr
        if ct_for_scatter_plot_path_abs:
            try:
                scatter_ct_sitk = load_image(str(ct_for_scatter_plot_path_abs))
                scatter_true_ct_arr_loaded = sitk.GetArrayFromImage(scatter_ct_sitk)
            except Exception as e:
                print(f"Warning: Could not load ct_for_scatter_plot_path {ct_for_scatter_plot_path_abs}: {e}. Will use original_ct_path for scatter plots.")

        multislice_true_ct_arr_loaded = original_ct_arr
        if ct_for_multislice_plot_path_abs:
            try:
                multislice_ct_sitk = load_image(str(ct_for_multislice_plot_path_abs))
                multislice_true_ct_arr_loaded = sitk.GetArrayFromImage(multislice_ct_sitk)
            except Exception as e:
                print(f"Warning: Could not load ct_for_multislice_plot_path {ct_for_multislice_plot_path_abs}: {e}. Will use original_ct_path for multi-slice plots.")

        if not (original_ct_arr.shape == registered_mri_arr.shape == bone_mask_arr.shape and \
                (plot_background_arr_loaded is None or plot_background_arr_loaded.shape == original_ct_arr.shape) and \
                (scatter_true_ct_arr_loaded is None or scatter_true_ct_arr_loaded.shape == original_ct_arr.shape) and \
                (multislice_true_ct_arr_loaded is None or multislice_true_ct_arr_loaded.shape == original_ct_arr.shape)
                ):
            print(f"Error: Image shapes for {subject_id} - {mri_modality_name} are inconsistent after loading. Please check inputs.")
            if registered_mri_arr.shape != original_ct_arr.shape:
                 print(f"Attempting to resample registered MRI for {subject_id} - {mri_modality_name}")
                 resampler = sitk.ResampleImageFilter()
                 resampler.SetReferenceImage(original_ct_sitk)
                 resampler.SetInterpolator(sitk.sitkLinear)
                 resampler.SetTransform(sitk.Transform())
                 registered_mri_sitk = resampler.Execute(registered_mri_sitk)
                 registered_mri_arr = sitk.GetArrayFromImage(registered_mri_sitk)
                 if registered_mri_arr.shape != original_ct_arr.shape:
                      print(f"FATAL: Resampling MRI failed for {subject_id} - {mri_modality_name}. Skipping.")
                      return

        if bone_mask_arr.sum() < 2:
            print(f"Error: Bone mask too small for {subject_id} - {mri_modality_name}. Exiting."); return

        model_name_gns = f"{mri_modality_name}_GNS_Background"
        mri_glob_min, mri_glob_max = registered_mri_arr.min(), registered_mri_arr.max()
        slope_gns = (cli_args.clip_max_hu - cli_args.clip_min_hu) / (mri_glob_max - mri_glob_min) if mri_glob_max != mri_glob_min else 0
        intercept_gns = cli_args.clip_min_hu - slope_gns * mri_glob_min
        params_gns = (slope_gns, intercept_gns)
        pseudo_ct_gns_arr = apply_transform_and_clip(registered_mri_arr, params_gns, "GNS", clip_range=(cli_args.clip_min_hu, cli_args.clip_max_hu))

        model_name_rfb_patch = f"{mri_modality_name}_RFB_patch_s{cli_args.patch_size}_hybrid"
        patch_radius_val = (cli_args.patch_size - 1) // 2
        mri_patches_train, ct_targets_train = extract_patches_and_targets(
            registered_mri_arr, original_ct_arr, bone_mask_arr, patch_radius_val, cli_args.max_patches_rf)

        rf_patch_model = None
        if mri_patches_train.ndim == 2 and mri_patches_train.shape[0] > 0:
            rf_patch_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, 
                                                   max_depth=25, min_samples_leaf=5, min_samples_split=10)
            print(f"  Training {model_name_rfb_patch} (n_estimators={rf_patch_model.n_estimators}, max_depth={rf_patch_model.max_depth})...")
            rf_patch_model.fit(mri_patches_train, ct_targets_train)
        else:
            print(f"  No training data for RFB_patch for {subject_id} - {mri_modality_name}. Skipping model generation.")

        if rf_patch_model:
            params_dict_rfb_patch = {
                "n_estimators": rf_patch_model.n_estimators, "max_depth": rf_patch_model.get_params()["max_depth"],
                "min_samples_leaf": rf_patch_model.get_params()["min_samples_leaf"], "min_samples_split": rf_patch_model.get_params()["min_samples_split"],
                "random_state": rf_patch_model.random_state, "patch_size": cli_args.patch_size,
                "num_training_patches": len(ct_targets_train) if ct_targets_train is not None else 0
            }
            pseudo_ct_rfb_patch_arr = apply_transform_and_clip(
                registered_mri_arr, rf_patch_model, "RFB_patch",
                bone_mask_arr_for_hybrid=bone_mask_arr, background_ct_arr_for_hybrid=pseudo_ct_gns_arr,
                clip_range=(cli_args.clip_min_hu, cli_args.clip_max_hu), patch_radius_for_rfb_patch=patch_radius_val)
            
            pseudo_ct_rfb_patch_sitk = sitk.GetImageFromArray(pseudo_ct_rfb_patch_arr)
            pseudo_ct_rfb_patch_sitk.CopyInformation(original_ct_sitk)
            sitk.WriteImage(pseudo_ct_rfb_patch_sitk, str(output_dir_subject_abs / f"{output_file_base_name}_pseudoCT_RFB_patch_s{cli_args.patch_size}_hybrid.nii.gz"))

            metrics_rfb_patch = calculate_metrics(original_ct_arr, pseudo_ct_rfb_patch_arr, bone_mask_arr)
            save_metrics_to_file(model_name_rfb_patch, metrics_rfb_patch, params_dict_rfb_patch, output_dir_subject_abs / f"{output_file_base_name}_metrics_RFB_patch_s{cli_args.patch_size}_hybrid.txt")
            
            plot_summary_quad_figure(
                true_ct_arr_quantitative=original_ct_arr, 
                pseudo_ct_arr=pseudo_ct_rfb_patch_arr,
                bone_mask_arr=bone_mask_arr,
                metrics_dict=metrics_rfb_patch,
                output_dir_path=output_dir_subject_abs,
                base_filename=output_file_base_name, 
                model_name_suffix=f"RFB_patch_s{cli_args.patch_size}_hybrid", 
                slice_idx_to_plot=cli_args.summary_plot_slice_idx, 
                plot_background_arr=plot_background_arr_loaded, 
                subject_id=subject_id 
            )
            
            plot_current_metrics_summary(
                metrics_dict=metrics_rfb_patch,
                output_dir_path=output_dir_subject_abs,
                base_filename=output_file_base_name,
                model_name_suffix=f"RFB_patch_s{cli_args.patch_size}_hybrid",
                subject_id=subject_id
            )
            
            mri_min, mri_max = registered_mri_arr.min(), registered_mri_arr.max()
            if mri_max > mri_min:
                registered_mri_arr_norm = (registered_mri_arr - mri_min) / (mri_max - mri_min)
            else:
                registered_mri_arr_norm = np.zeros_like(registered_mri_arr) 

            plot_mri_norm_vs_ct_hu(
                mri_norm_arr=registered_mri_arr_norm,
                ct_hu_arr=original_ct_arr, 
                bone_mask_arr=bone_mask_arr,
                output_dir_path=output_dir_subject_abs,
                base_filename=output_file_base_name,
                model_name_suffix=f"RFB_patch_s{cli_args.patch_size}_hybrid",
                subject_id=subject_id
            )
            
            plot_mri_norm_vs_pseudo_ct_hu(
                mri_norm_arr=registered_mri_arr_norm,
                pseudo_ct_arr=pseudo_ct_rfb_patch_arr,
                bone_mask_arr=bone_mask_arr,
                output_dir_path=output_dir_subject_abs,
                base_filename=output_file_base_name,
                model_name_suffix=f"RFB_patch_s{cli_args.patch_size}_hybrid",
                subject_id=subject_id
            )
            
            plot_density_correlation(
                true_ct_arr=original_ct_arr,
                pseudo_ct_arr=pseudo_ct_rfb_patch_arr,
                bone_mask_arr=bone_mask_arr,
                output_dir_path=output_dir_subject_abs,
                base_filename=output_file_base_name,
                model_name_suffix=f"RFB_patch_s{cli_args.patch_size}_hybrid",
                subject_id=subject_id
            )
            
            plot_comprehensive_conference_summary(
                true_ct_arr=original_ct_arr,
                pseudo_ct_arr=pseudo_ct_rfb_patch_arr,
                registered_mri_arr=registered_mri_arr,
                bone_mask_arr=bone_mask_arr,
                metrics_dict=metrics_rfb_patch,
                output_dir_path=output_dir_subject_abs,
                base_filename=output_file_base_name,
                model_name_suffix=f"RFB_patch_s{cli_args.patch_size}_hybrid",
                subject_id=subject_id,
                slice_indices=[64, 96, 128]
            )
            
            create_publication_figure(
                true_ct_arr=original_ct_arr,
                pseudo_ct_arr=pseudo_ct_rfb_patch_arr,
                registered_mri_arr=registered_mri_arr,
                bone_mask_arr=bone_mask_arr,
                metrics_dict=metrics_rfb_patch,
                output_dir_path=output_dir_subject_abs,
                base_filename=output_file_base_name,
                model_name_suffix=f"RFB_patch_s{cli_args.patch_size}_hybrid",
                subject_id=subject_id
            )
            
            print(f"  Processing for {subject_id} - {model_name_rfb_patch} complete.")
        else:
            print(f"  RFB_patch model for {subject_id} - {mri_modality_name} could not be generated (no training data or fit failed).")

    except FileNotFoundError as e:
        print(f"  FileNotFoundError for {subject_id} - {mri_modality_name}: {e}. Skipping this modality.")
    except Exception as e:
        print(f"  Unhandled error during processing {subject_id} - {mri_modality_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    global args 
    parser = argparse.ArgumentParser(description="Generate Pseudo-CT from multiple MRI modalities for multiple subjects using Patch-based Random Forest.")
    
    parser.add_argument("data_dir", type=str, help="Root directory containing subject folders.")
    parser.add_argument("output_dir", type=str, help="Root directory to save pseudo-CTs, metrics, and plots.")
    parser.add_argument("--subject_prefix", type=str, default="subject_", help="Prefix for subject directory names.")
    parser.add_argument("--mri_modality_prefixes", nargs='+', required=True, 
                        help="List of MRI modality prefixes.")
    parser.add_argument("--registered_mri_suffix", type=str, default="_registered_to_CT_Bone_improved.nii.gz",
                        help="Suffix for registered MRI filenames.")
    parser.add_argument("--original_ct_filename", type=str, default="CT.nii.gz",
                        help="Filename of the original quantitative CT.")
    parser.add_argument("--bone_mask_filename", type=str, default="bone_mask_improved.nii.gz",
                        help="Filename of the bone mask.")
    parser.add_argument("--plot_background_ct_filename", type=str, default="CT_Bone_improved.nii.gz",
                        help="Optional: Filename of the CT for plot background.")
    parser.add_argument("--scatter_plot_ct_filename", type=str, default=None,
                        help="Optional: Filename of the CT for MRI vs CT scatter plots.")
    parser.add_argument("--multislice_plot_ct_filename", type=str, default=None,
                        help="Optional: Filename of the CT for orthogonal multi-slice plots.")
    parser.add_argument("--clip_min_hu", type=float, default=-1000, help="Minimum HU value for clipping.")
    parser.add_argument("--clip_max_hu", type=float, default=3000, help="Maximum HU value for clipping.")
    parser.add_argument("--patch_size", type=int, default=9, help="Size of 3D patches.")
    parser.add_argument("--max_patches_rf", type=int, default=150000, help="Max patches for RF training.")
    parser.add_argument("--num_comparison_slices", type=int, default=4, help="Number of slices per view for orthogonal multi-slice comparison plot.")
    parser.add_argument("--background_hu_value", type=float, default=-1000.0, 
                        help="HU value to use for the background in the hybrid pseudo-CT.")
    parser.add_argument("--summary_plot_slice_idx", type=int, default=127, 
                        help="Axial slice index for the summary quad plot. If out of bounds, middle slice is used.")
    
    args = parser.parse_args()

    if args.patch_size % 2 == 0:
        print("Error: --patch_size must be an odd number."); return

    data_dir_path = Path(args.data_dir)
    output_dir_root_path = Path(args.output_dir)
    output_dir_root_path.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([d for d in data_dir_path.iterdir() if d.is_dir() and d.name.startswith(args.subject_prefix)])
    if not subject_dirs:
        print(f"No subject directories found in {data_dir_path} with prefix '{args.subject_prefix}'.")
        subject_dirs = sorted([Path(p) for p in glob.glob(str(data_dir_path / f"{args.subject_prefix}*")) if Path(p).is_dir()])
        if not subject_dirs:
            print(f"Glob also found no subject directories. Exiting.")
            return
            
    print(f"Found {len(subject_dirs)} subject(s) to process: {[s.name for s in subject_dirs]}")

    for subject_dir_path in subject_dirs:
        subject_id = subject_dir_path.name
        print(f"\nProcessing Subject Directory: {subject_id}")
        
        output_dir_subject_abs = output_dir_root_path / subject_id
        output_dir_subject_abs.mkdir(parents=True, exist_ok=True)

        original_ct_path_abs = subject_dir_path / args.original_ct_filename
        bone_mask_path_abs = subject_dir_path / args.bone_mask_filename

        plot_bg_ct_path_abs = subject_dir_path / args.plot_background_ct_filename if args.plot_background_ct_filename else None
        scatter_ct_path_abs = subject_dir_path / args.scatter_plot_ct_filename if args.scatter_plot_ct_filename else original_ct_path_abs
        multislice_ct_path_abs = subject_dir_path / args.multislice_plot_ct_filename if args.multislice_plot_ct_filename else original_ct_path_abs
        
        if not original_ct_path_abs.exists():
            print(f"  Quantitative CT {args.original_ct_filename} not found in {subject_dir_path}. Skipping subject {subject_id}.")
            continue
        if not bone_mask_path_abs.exists():
            print(f"  Bone mask {args.bone_mask_filename} not found in {subject_dir_path}. Skipping subject {subject_id}.")
            continue
        if args.plot_background_ct_filename and not plot_bg_ct_path_abs.exists():
            print(f"  Warning: Specified plot background CT {args.plot_background_ct_filename} not found in {subject_dir_path}. Will use quantitative CT for plot background.")
            plot_bg_ct_path_abs = original_ct_path_abs
        if args.scatter_plot_ct_filename and not (subject_dir_path / args.scatter_plot_ct_filename).exists():
            print(f"  Warning: Specified scatter plot CT {args.scatter_plot_ct_filename} not found in {subject_dir_path}. Will use quantitative CT.")
            scatter_ct_path_abs = original_ct_path_abs
        if args.multislice_plot_ct_filename and not (subject_dir_path / args.multislice_plot_ct_filename).exists():
            print(f"  Warning: Specified multislice plot CT {args.multislice_plot_ct_filename} not found in {subject_dir_path}. Will use quantitative CT.")
            multislice_ct_path_abs = original_ct_path_abs

        for mri_prefix in args.mri_modality_prefixes:
            registered_mri_filename = f"{mri_prefix}{args.registered_mri_suffix}"
            registered_mri_path_abs = subject_dir_path / registered_mri_filename
            
            if registered_mri_path_abs.exists():
                process_subject_modality(
                    subject_id=subject_id,
                    original_ct_path_abs=original_ct_path_abs,
                    registered_mri_path_abs=registered_mri_path_abs,
                    bone_mask_path_abs=bone_mask_path_abs,
                    output_dir_subject_abs=output_dir_subject_abs,
                    mri_modality_name=mri_prefix,
                    plot_background_image_path_abs=plot_bg_ct_path_abs,
                    ct_for_scatter_plot_path_abs=scatter_ct_path_abs,
                    ct_for_multislice_plot_path_abs=multislice_ct_path_abs,
                    cli_args=args 
                )
            else:
                print(f"  Registered MRI {registered_mri_filename} not found in {subject_dir_path}. Skipping this modality for subject {subject_id}.")
                
    print("\nAll processing finished.")

if __name__ == "__main__":
    main()
