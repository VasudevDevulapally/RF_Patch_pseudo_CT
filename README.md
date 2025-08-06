 This method involves training a random forest model on a dataset of paired MRI and CT images, using image patches to capture the relationship between the two modalities.
 Following training, the model may be used to predict pCT images from MRI data. The algorithm is evaluated using bone-specific MR sequences, such as UTE, PETRA, and ZTE, that were taken from 25 ex-vivo skulls, and the MAE_Bone evaluate the discrepancies between the actual CT and pseudo-CT generated from MRI sequences
 
 
 MRI to CT (Pseudo-CT) Translation : <img width="1080" height="761" alt="13534_2024_430_Fig1_HTML" src="https://github.com/user-attachments/assets/1c04be7f-33da-445f-892c-e694a28c4a9d" />

 RF_Patch_approach for Pseudo-CT Generation :<img width="2463" height="2345" alt="RF_Patch_method" src="https://github.com/user-attachments/assets/46b6c6c1-3352-4e64-ac22-67199ff8595c" />


 Comparative Assessment of CT with MR-Bone-Specific sequences

 Original CT of an ex-vivo skull<img width="2250" height="766" alt="CT_original_Axial" src="https://github.com/user-attachments/assets/5aa6e749-1d0a-45b4-b0a5-7ae4e38f9410" />

 Example UTE sequence acquired from ex-vivo skull : <img width="2250" height="705" alt="UTE_pCT_ex-vivo_Raw" src="https://github.com/user-attachments/assets/3189c3dc-e200-49ce-adc7-aa6fc8f5c227" />

 Example of Pseudo-CT Generated from UTE_Sequence : <img width="2250" height="766" alt="UTE_pCT_Axial" src="https://github.com/user-attachments/assets/ba489e88-3c89-41b1-9a2f-3e1829e8494d" />
