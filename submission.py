import os
import pandas as pd


def generate_submission_file(filename, observation_ids, s_pred, append=False):
    """Generate submission file for Kaggle

    Parameters
    ----------
    filename : string
        Submission filename.
    observation_ids : 1d array-like
        Test observations ids
    s_pred : list of 1d array-like
        Set predictions for test observations.
    """

    s_pred = [" ".join(map(str, pred_set)) for pred_set in s_pred]

    df = pd.DataFrame({"Id": observation_ids, "Predicted": s_pred})

    if append and os.path.isfile(filename):
        df_prev_preds = pd.read_csv(filename)
        df_all = pd.concat([df, df_prev_preds])
        df_all.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)
