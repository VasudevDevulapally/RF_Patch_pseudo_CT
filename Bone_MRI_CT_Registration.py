#!/usr/bin/env python3
"""
MRI to CT Registration Pipeline

This script performs a complete registration pipeline for MRI modalities to CT space:
1. Bias correction of MRI modalities using N4ITK
2. Denoising MRI images (salt and pepper noise)
3. Normalizing MRI images
4. Registering MRI to corresponding CT space
5. Plotting registered MRI image overlaid on CT_Bone_improved.nii.gz
6. Computing and plotting registration metrics
 
"""

import os
import sys
import glob
import time
import logging
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, normalized_mutual_information

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("registration_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define MRI modalities
MRI_MODALITIES = ["T1w", "PETRA", "UTE_SE", "UTE_IR", "ute_pCT"]

class RegistrationPipeline:
    """Main class for the MRI to CT registration pipeline."""
    
    def __init__(self, data_dir, output_dir, subjects=None):
        """
        Initialize the registration pipeline.
        
        Args:
            data_dir (str): Directory containing the subject data
            output_dir (str): Directory to save the output files
            subjects (list, optional): List of subject IDs to process. If None, all subjects in data_dir will be processed.
        """
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.subjects = subjects
        self.subject_data = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for each processing step
        self.bias_corrected_dir = os.path.join(self.output_dir, "bias_corrected")
        self.denoised_dir = os.path.join(self.output_dir, "denoised")
        self.normalized_dir = os.path.join(self.output_dir, "normalized")
        self.registered_dir = os.path.join(self.output_dir, "registered")
        self.overlay_dir = os.path.join(self.output_dir, "overlays")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        
        for directory in [self.bias_corrected_dir, self.denoised_dir, self.normalized_dir, 
                         self.registered_dir, self.overlay_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Initialized registration pipeline with data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def collect_subject_data(self):
        """
        Collect file paths and metadata for all subjects.
        
        Returns:
            dict: Dictionary containing file paths for each subject
        """
        logger.info("Collecting subject data...")
        
        # If subjects not specified, find all subject directories
        if self.subjects is None:
            self.subjects = [d for d in os.listdir(self.data_dir) 
                           if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for subject_id in self.subjects:
            subject_dir = os.path.join(self.data_dir, subject_id)
            
            if not os.path.isdir(subject_dir):
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue
            
            self.subject_data[subject_id] = {
                "ct": None,
                "ct_bone_improved": None,
                "bone_mask_improved": None,
                "mri": {}
            }
            
            # Find CT files
            ct_file = os.path.join(subject_dir, "CT.nii.gz")
            if os.path.exists(ct_file):
                self.subject_data[subject_id]["ct"] = ct_file
            else:
                logger.warning(f"CT file not found for subject {subject_id}")
            
            # Find CT bone improved file
            ct_bone_file = os.path.join(subject_dir, "CT_bone_improved.nii.gz")
            if os.path.exists(ct_bone_file):
                self.subject_data[subject_id]["ct_bone_improved"] = ct_bone_file
            else:
                logger.warning(f"CT bone improved file not found for subject {subject_id}")
            
            # Find bone mask improved file
            bone_mask_file = os.path.join(subject_dir, "bone_mask_improved.nii.gz")
            if os.path.exists(bone_mask_file):
                self.subject_data[subject_id]["bone_mask_improved"] = bone_mask_file
            else:
                logger.warning(f"Bone mask improved file not found for subject {subject_id}")
            
            # Find MRI files for each modality
            for modality in MRI_MODALITIES:
                mri_file = os.path.join(subject_dir, f"{modality}.nii.gz")
                if os.path.exists(mri_file):
                    self.subject_data[subject_id]["mri"][modality] = mri_file
                else:
                    logger.warning(f"{modality} file not found for subject {subject_id}")
        
        # Log summary of collected data
        logger.info(f"Collected data for {len(self.subject_data)} subjects")
        for subject_id, data in self.subject_data.items():
            logger.info(f"Subject {subject_id}: CT: {data['ct'] is not None}, "
                       f"CT bone: {data['ct_bone_improved'] is not None}, "
                       f"Bone mask: {data['bone_mask_improved'] is not None}, "
                       f"MRI modalities: {list(data['mri'].keys())}")
        
        return self.subject_data
    
    def apply_n4_bias_correction(self, input_image_path, output_image_path):
        """
        Apply N4ITK bias correction to an MRI image.
        
        Args:
            input_image_path (str): Path to the input MRI image
            output_image_path (str): Path to save the bias-corrected image
            
        Returns:
            str: Path to the bias-corrected image
        """
        logger.info(f"Applying N4ITK bias correction to {input_image_path}")
        
        try:
            # Read the input image
            input_image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
            
            # Create a mask for the N4 filter (optional)
            # If no mask is provided, the filter will create one internally
            mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
            
            # Create the N4 bias field correction filter
            n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
            
            # Set parameters
            n4_filter.SetMaximumNumberOfIterations([50, 50, 50, 50])
            n4_filter.SetConvergenceThreshold(0.0001)
            
            # Apply the filter
            corrected_image = n4_filter.Execute(input_image, mask_image)
            
            # Save the bias-corrected image
            sitk.WriteImage(corrected_image, output_image_path)
            
            logger.info(f"Bias correction completed. Output saved to {output_image_path}")
            return output_image_path
            
        except Exception as e:
            logger.error(f"Error applying bias correction: {str(e)}")
            return None
    
    def denoise_image(self, input_image_path, output_image_path):
        """
        Denoise an MRI image to remove salt and pepper noise.
        
        Args:
            input_image_path (str): Path to the input MRI image
            output_image_path (str): Path to save the denoised image
            
        Returns:
            str: Path to the denoised image
        """
        logger.info(f"Denoising image {input_image_path}")
        
        try:
            # Load the image using nibabel
            img = nib.load(input_image_path)
            data = img.get_fdata()
            
            # Apply non-local means denoising
            # This is effective for salt and pepper noise while preserving edges
            # Handle API changes in newer versions of scikit-image
            try:
                # Try new API (scikit-image >= 0.19.0)
                denoised_data = denoise_nl_means(data, patch_size=5, patch_distance=6, h=0.05, 
                                              channel_axis=None, preserve_range=True)
            except TypeError:
                # Fall back to old API (scikit-image < 0.19.0)
                denoised_data = denoise_nl_means(data, patch_size=5, patch_distance=6, h=0.05, 
                                              multichannel=False, preserve_range=True)
            
            # Create a new nifti image with the denoised data
            denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)
            
            # Save the denoised image
            nib.save(denoised_img, output_image_path)
            
            logger.info(f"Denoising completed. Output saved to {output_image_path}")
            return output_image_path
            
        except Exception as e:
            logger.error(f"Error denoising image: {str(e)}")
            return None
    
    def normalize_image(self, input_image_path, output_image_path):
        """
        Normalize an MRI image.
        
        Args:
            input_image_path (str): Path to the input MRI image
            output_image_path (str): Path to save the normalized image
            
        Returns:
            str: Path to the normalized image
        """
        logger.info(f"Normalizing image {input_image_path}")
        
        try:
            # Load the image using nibabel
            img = nib.load(input_image_path)
            data = img.get_fdata()
            
            # Apply intensity normalization (z-score)
            # Exclude zero values (background) from normalization
            mask = data > 0
            if np.sum(mask) > 0:  # Check if there are non-zero voxels
                mean = np.mean(data[mask])
                std = np.std(data[mask])
                if std > 0:  # Avoid division by zero
                    normalized_data = np.zeros_like(data)
                    normalized_data[mask] = (data[mask] - mean) / std
                else:
                    normalized_data = data
            else:
                normalized_data = data
            
            # Create a new nifti image with the normalized data
            normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
            
            # Save the normalized image
            nib.save(normalized_img, output_image_path)
            
            logger.info(f"Normalization completed. Output saved to {output_image_path}")
            return output_image_path
            
        except Exception as e:
            logger.error(f"Error normalizing image: {str(e)}")
            return None
    
    def register_mri_to_ct(self, mri_image_path, ct_image_path, output_image_path):
        """
        Register an MRI image to CT space.
        
        Args:
            mri_image_path (str): Path to the input MRI image
            ct_image_path (str): Path to the reference CT image
            output_image_path (str): Path to save the registered image
            
        Returns:
            tuple: (Path to the registered image, transformation parameters)
        """
        logger.info(f"Registering {mri_image_path} to {ct_image_path}")
        
        try:
            # Read the fixed (CT) and moving (MRI) images
            fixed_image = sitk.ReadImage(ct_image_path, sitk.sitkFloat32)
            moving_image = sitk.ReadImage(mri_image_path, sitk.sitkFloat32)
            
            # Initialize the registration method
            registration_method = sitk.ImageRegistrationMethod()
            
            # Set up similarity metric
            # Mutual information is often good for multi-modal registration
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.2)
            
            # Set up optimizer
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0, 
                numberOfIterations=100, 
                convergenceMinimumValue=1e-6, 
                convergenceWindowSize=10
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # Set up interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Set up initial transformation
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, 
                moving_image, 
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            
            # Execute the registration
            final_transform = registration_method.Execute(fixed_image, moving_image)
            
            # Resample the moving image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)
            
            registered_image = resampler.Execute(moving_image)
            
            # Save the registered image
            sitk.WriteImage(registered_image, output_image_path)
            
            # Get registration metrics
            metrics = {
                "final_metric_value": registration_method.GetMetricValue(),
                "iterations": registration_method.GetOptimizerIteration(),
                "convergence_value": registration_method.GetOptimizerConvergenceValue(),
                "transform_parameters": final_transform.GetParameters()
            }
            
            logger.info(f"Registration completed. Output saved to {output_image_path}")
            logger.info(f"Registration metrics: {metrics}")
            
            return output_image_path, metrics
            
        except Exception as e:
            logger.error(f"Error registering image: {str(e)}")
            return None, None
    
    def create_overlay(self, registered_mri_path, ct_bone_path, output_image_path):
        """
        Create an overlay of the registered MRI on the CT bone image.
        
        Args:
            registered_mri_path (str): Path to the registered MRI image
            ct_bone_path (str): Path to the CT bone image
            output_image_path (str): Path to save the overlay image
            
        Returns:
            str: Path to the overlay image
        """
        logger.info(f"Creating overlay of {registered_mri_path} on {ct_bone_path}")
        
        try:
            # Load the images
            mri_img = nib.load(registered_mri_path)
            ct_img = nib.load(ct_bone_path)
            
            mri_data = mri_img.get_fdata()
            ct_data = ct_img.get_fdata()
            
            # Ensure the images have the same dimensions
            if mri_data.shape != ct_data.shape:
                logger.error(f"Image dimensions do not match: MRI {mri_data.shape}, CT {ct_data.shape}")
                return None
            
            # Create the overlay
            # Select a middle slice for visualization
            slice_idx = mri_data.shape[2] // 2
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot CT bone
            axes[0].imshow(ct_data[:, :, slice_idx].T, cmap='gray')
            axes[0].set_title('CT Bone')
            axes[0].axis('off')
            
            # Plot registered MRI
            axes[1].imshow(mri_data[:, :, slice_idx].T, cmap='hot')
            axes[1].set_title('Registered MRI')
            axes[1].axis('off')
            
            # Plot overlay
            axes[2].imshow(ct_data[:, :, slice_idx].T, cmap='gray')
            axes[2].imshow(mri_data[:, :, slice_idx].T, cmap='hot', alpha=0.5)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Overlay created and saved to {output_image_path}")
            return output_image_path
            
        except Exception as e:
            logger.error(f"Error creating overlay: {str(e)}")
            return None
    
    def compute_registration_metrics(self, registered_mri_path, ct_bone_path, output_metrics_path):
        """
        Compute metrics to evaluate the registration quality.
        
        Args:
            registered_mri_path (str): Path to the registered MRI image
            ct_bone_path (str): Path to the CT bone image
            output_metrics_path (str): Path to save the metrics
            
        Returns:
            dict: Dictionary containing the computed metrics
        """
        logger.info(f"Computing registration metrics for {registered_mri_path}")
        
        try:
            # Load the images
            mri_img = nib.load(registered_mri_path)
            ct_img = nib.load(ct_bone_path)
            
            mri_data = mri_img.get_fdata()
            ct_data = ct_img.get_fdata()
            
            # Ensure the images have the same dimensions
            if mri_data.shape != ct_data.shape:
                logger.error(f"Image dimensions do not match: MRI {mri_data.shape}, CT {ct_data.shape}")
                return None
            
            # Compute metrics
            # 1. Mean Squared Error (MSE)
            mse = mean_squared_error(ct_data.flatten(), mri_data.flatten())
            
            # 2. Normalized Mutual Information (NMI)
            nmi = normalized_mutual_information(ct_data, mri_data)
            
            # 3. Structural Similarity Index (SSIM)
            # Compute SSIM for each slice and take the average
            ssim_values = []
            for i in range(ct_data.shape[2]):
                ssim_val = ssim(ct_data[:, :, i], mri_data[:, :, i], 
                              data_range=max(ct_data.max(), mri_data.max()) - min(ct_data.min(), mri_data.min()))
                ssim_values.append(ssim_val)
            
            avg_ssim = np.mean(ssim_values)
            
            # Compile metrics
            metrics = {
                "mse": mse,
                "nmi": nmi,
                "ssim": avg_ssim
            }
            
            # Save metrics to file
            with open(output_metrics_path, 'w') as f:
                f.write(f"Registration Metrics for {os.path.basename(registered_mri_path)}:\n")
                f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
                f.write(f"Normalized Mutual Information (NMI): {nmi:.6f}\n")
                f.write(f"Structural Similarity Index (SSIM): {avg_ssim:.6f}\n")
            
            # Create a visualization of the metrics
            metrics_plot_path = output_metrics_path.replace('.txt', '.png')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_names = list(metrics.keys())
            metrics_values = list(metrics.values())
            
            ax.bar(metrics_names, metrics_values)
            ax.set_title(f'Registration Metrics for {os.path.basename(registered_mri_path)}')
            ax.set_ylabel('Value')
            
            plt.tight_layout()
            plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Metrics computed and saved to {output_metrics_path}")
            logger.info(f"Metrics plot saved to {metrics_plot_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return None
    
    def process_subject(self, subject_id):
        """
        Process a single subject through the entire pipeline.
        
        Args:
            subject_id (str): Subject ID to process
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        logger.info(f"Processing subject {subject_id}")
        
        subject_data = self.subject_data.get(subject_id)
        if subject_data is None:
            logger.error(f"No data found for subject {subject_id}")
            return False
        
        # Check if CT and CT bone improved files exist
        if subject_data["ct"] is None or subject_data["ct_bone_improved"] is None:
            logger.error(f"CT or CT bone improved file not found for subject {subject_id}")
            return False
        
        # Process each MRI modality
        for modality, mri_path in subject_data["mri"].items():
            logger.info(f"Processing {modality} for subject {subject_id}")
            
            # Create output directories for this subject
            subject_bias_dir = os.path.join(self.bias_corrected_dir, subject_id)
            subject_denoised_dir = os.path.join(self.denoised_dir, subject_id)
            subject_normalized_dir = os.path.join(self.normalized_dir, subject_id)
            subject_registered_dir = os.path.join(self.registered_dir, subject_id)
            subject_overlay_dir = os.path.join(self.overlay_dir, subject_id)
            subject_metrics_dir = os.path.join(self.metrics_dir, subject_id)
            
            for directory in [subject_bias_dir, subject_denoised_dir, subject_normalized_dir, 
                             subject_registered_dir, subject_overlay_dir, subject_metrics_dir]:
                os.makedirs(directory, exist_ok=True)
            
            # Define output file paths
            bias_corrected_path = os.path.join(subject_bias_dir, f"{modality}_bias_corrected.nii.gz")
            denoised_path = os.path.join(subject_denoised_dir, f"{modality}_denoised.nii.gz")
            normalized_path = os.path.join(subject_normalized_dir, f"{modality}_normalized.nii.gz")
            registered_path = os.path.join(subject_registered_dir, f"{modality}_registered.nii.gz")
            overlay_path = os.path.join(subject_overlay_dir, f"{modality}_overlay.png")
            metrics_path = os.path.join(subject_metrics_dir, f"{modality}_metrics.txt")
            
            # 1. Apply N4ITK bias correction
            bias_corrected_path = self.apply_n4_bias_correction(mri_path, bias_corrected_path)
            if bias_corrected_path is None:
                logger.error(f"Bias correction failed for {modality}, subject {subject_id}")
                continue
            
            # 2. Denoise the bias-corrected image
            denoised_path = self.denoise_image(bias_corrected_path, denoised_path)
            if denoised_path is None:
                logger.error(f"Denoising failed for {modality}, subject {subject_id}")
                continue
            
            # 3. Normalize the denoised image
            normalized_path = self.normalize_image(denoised_path, normalized_path)
            if normalized_path is None:
                logger.error(f"Normalization failed for {modality}, subject {subject_id}")
                continue
            
            # 4. Register the normalized MRI to CT space
            registered_path, reg_metrics = self.register_mri_to_ct(
                normalized_path, subject_data["ct"], registered_path
            )
            if registered_path is None:
                logger.error(f"Registration failed for {modality}, subject {subject_id}")
                continue
            
            # 5. Create overlay of registered MRI on CT bone
            overlay_path = self.create_overlay(
                registered_path, subject_data["ct_bone_improved"], overlay_path
            )
            if overlay_path is None:
                logger.error(f"Overlay creation failed for {modality}, subject {subject_id}")
                continue
            
            # 6. Compute registration metrics
            metrics = self.compute_registration_metrics(
                registered_path, subject_data["ct_bone_improved"], metrics_path
            )
            if metrics is None:
                logger.error(f"Metrics computation failed for {modality}, subject {subject_id}")
                continue
            
            logger.info(f"Successfully processed {modality} for subject {subject_id}")
        
        return True
    
    def run_pipeline(self):
        """
        Run the complete registration pipeline for all subjects.
        
        Returns:
            bool: True if pipeline execution was successful, False otherwise
        """
        logger.info("Starting registration pipeline")
        
        # Collect subject data
        self.collect_subject_data()
        
        if not self.subject_data:
            logger.error("No subject data found")
            return False
        
        # Process each subject
        success_count = 0
        for subject_id in self.subject_data:
            if self.process_subject(subject_id):
                success_count += 1
        
        logger.info(f"Registration pipeline completed. Successfully processed {success_count}/{len(self.subject_data)} subjects")
        return success_count > 0


def main():
    """Main function to run the registration pipeline."""
    parser = argparse.ArgumentParser(description="MRI to CT Registration Pipeline")
    parser.add_argument("--data_dir", required=True, help="Directory containing the subject data")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output files")
    parser.add_argument("--subjects", nargs="+", help="List of subject IDs to process (optional)")
    
    args = parser.parse_args()
    
    # Create and run the registration pipeline
    pipeline = RegistrationPipeline(args.data_dir, args.output_dir, args.subjects)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
