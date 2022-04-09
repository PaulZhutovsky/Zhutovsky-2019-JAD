"""Script producing the cleaned demographics + clnical file

Usage:
    create_clinical_data_file.py DATA_FILE

Arguments:
    DATA_FILE   Full path to where the uncleaned file is located
"""
import pandas as pd
import numpy as np
import os.path as osp
from docopt import docopt


CLINICAL_FEATURES = ['age_Dx_T0', 'Sex', 'Education_Verhage', 'MMSE', 'FBI_tot_baseline', 'SRI_tot_baseline']
VALUES_TO_REPLACE = {'Sex': {'m': 0, 'f': 1}}


def load_file(file_path):
    return pd.read_csv(file_path)


def extract_used_subjects(df, subj_ids):
    df_columns = df.columns
    # object is necessary as datatype because the data is completely mixed in the file
    data = np.zeros((subj_ids.size, df_columns.size), dtype=np.object)

    for i, subj_id  in enumerate(subj_ids):
        id_subj = (df.ID_D == subj_id).values
        assert id_subj.sum() == 1, 'Something wrong for {}'.format(subj_id)

        data[i] = df.loc[id_subj, :].values
    return pd.DataFrame(data=data, columns=df_columns)


def clinical_data(df):
    df_clinical = df[['ID_D'] + CLINICAL_FEATURES]
    return df_clinical.rename(columns={'ID_D': 'subj_id'})


def run(kwargs):
    file_path = kwargs['DATA_FILE']
    dir_name = osp.dirname(file_path)

    df_file_messy = load_file(file_path)
    df_subj_ids = load_file(osp.join(dir_name, 'class_labels.csv'))
    subj_ids = df_subj_ids.subj_id.values

    df_file_clean = extract_used_subjects(df_file_messy, subj_ids)
    df_clinical_clean = clinical_data(df_file_clean)

    df_file_clean.to_csv(osp.join(dir_name, 'demographics_used_subjects.csv'), index=False)
    df_clinical_clean = df_clinical_clean.drop('subj_id', axis=1)
    education_median = np.median(df_clinical_clean.Education_Verhage.values)
    education_replace = {'Education_Verhage': {999: education_median}}
    values_to_replace = dict(VALUES_TO_REPLACE, **education_replace)
    df_clinical_clean = df_clinical_clean.replace(values_to_replace)
    df_clinical_clean.age_Dx_T0 = pd.to_numeric(df_clinical_clean.age_Dx_T0.str.replace(',', '.'))
    df_clinical_clean.to_csv(osp.join(dir_name, 'data_set_clinical.csv'), index=False)

    np.save(osp.join(dir_name, 'data_set_clinical.npy'), df_clinical_clean.values)


if __name__ == '__main__':
    args = docopt(__doc__)
    run(args)
