How to fully reprocude the analysis.

1. Create the MRI data in .npy format.
python data_handling.py 
Adjust pathes in data_handling as needed

2. Create the clinical data. In the console:
python create_clinical_data_file.py /data/shared/bvFTD/paper_final/data/

3. Create combined MRI + clinical data.
python create_combined_data.py /data/shared/bvFTD/paper_final/data/

4. Run classification. (Should be actually split up and ran across different servers, i.e. things need to be adjusted in run_classification.py)
python run_classification.py

5. Create result tables.
python create_result_table.py /data/shared/bvFTD/paper_final/analysis/

6. Create table for data type comparisons.
python create_p_values_data_types_comparison.py /data/shared/bvFTD/paper_final/analysis/

7. Create p-values maps for SVM classifications.
python create_svm_group_association_maps.py --clf ['FTDvsRest' | 'FTDvsNeurol' | 'FTDvsPsych']

8. Extract significant clusters + region names out of the p-maps
python create_cluster_coord_files.py /data/shared/bvFTD/paper_final/analysis/

9. Create figures for paper.
python create_figures_paper.py /data/shared/bvFTD/paper_final/analysis/
