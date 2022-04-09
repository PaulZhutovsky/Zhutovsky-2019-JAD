from create_atlas import main

kwargs = {'--prob': '25',
          '--res': '1_5',
          'SAVE_DIR': '/data/shared/bvFTD/paper_final/atlases',
          'DOWNSAMPLE_TO': '/data/shared/bvFTD/VBM/vbm_data_baseline/bvFTD/4908/structural/mri/smwp1CAT4908_T1_reoriented_time01.nii'}

main(**kwargs)