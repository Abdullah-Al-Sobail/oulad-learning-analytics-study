import pandas as pd, numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

CAT = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability', 'code_module']
ENROLL_NUM = ['num_of_prev_attempts', 'studied_credits']
OTHER_NUM = ['total_clicks', 'avg_assessment_score', 'n_assessments_submitted', 'avg_days_before_deadline']

class TrainQuantileBinner(BaseEstimator, TransformerMixin):
    """Learns qcut quartile bin edges per column from whatever data it is
    fit on. Placed INSIDE an sklearn Pipeline (rather than applied to the
    whole outer training set before cross-validation), so that when
    RandomizedSearchCV performs its internal k-fold CV, each inner fold
    re-fits this transformer on only that fold's training rows -- no
    quartile-boundary information crosses from an inner validation fold
    into the edges used to transform it."""
    def __init__(self, q=4):
        self.q = q
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.edges_ = {}
        for col in self.columns_:
            vals = pd.to_numeric(X[col], errors='coerce').fillna(0)
            if vals.nunique() <= 1:
                self.edges_[col] = None
                continue
            try:
                _, edges = pd.qcut(vals, q=self.q, retbins=True, duplicates='drop')
                edges = np.array(edges, dtype=float)
                edges[0] = -np.inf
                edges[-1] = np.inf
                self.edges_[col] = edges
            except ValueError:
                self.edges_[col] = None
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            vals = pd.to_numeric(X[col], errors='coerce').fillna(0)
            edges = self.edges_.get(col)
            if edges is None:
                out[col] = 0
            else:
                binned = pd.cut(vals, bins=edges, labels=False, include_lowest=True)
                out[col] = binned.fillna(-1)
        return out.values
    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_q" for c in self.columns_])


def build_raw_features(student_info, student_vle, student_assessment, assessments, vle, cutoff_day=None):
    """Build per-enrollment raw (pre-quartile) features, including a
    CORRECTED avg_days_before_deadline = mean(deadline_date - date_submitted)
    rather than the earlier buggy mean(date_submitted) alone."""
    svle = student_vle.merge(vle[['id_site', 'code_module', 'code_presentation', 'activity_type']],
                              on=['id_site', 'code_module', 'code_presentation'], how='left')
    if cutoff_day is not None:
        svle = svle[svle['date'] <= cutoff_day]
    per_activity = svle.groupby(['id_student', 'code_module', 'code_presentation', 'activity_type'])['sum_click'].sum().reset_index()
    total_clicks = svle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index().rename(columns={'sum_click': 'total_clicks'})
    pivot = per_activity.pivot_table(index=['id_student', 'code_module', 'code_presentation'],
                                      columns='activity_type', values='sum_click', fill_value=0).reset_index()
    activity_cols = [c for c in pivot.columns if c not in ['id_student', 'code_module', 'code_presentation']]

    assess_elig = assessments.copy()
    if cutoff_day is not None:
        assess_elig = assess_elig[assess_elig['date'] <= cutoff_day]
    assess_merged = student_assessment.merge(assess_elig, on='id_assessment')  # has BOTH 'date' (deadline) and 'date_submitted'
    if cutoff_day is not None:
        assess_merged = assess_merged[assess_merged['date_submitted'] <= cutoff_day]
    # CORRECTED: actual days before deadline = deadline date - submission date
    # (positive = submitted early/on time relative to deadline, negative = late)
    assess_merged = assess_merged.copy()
    assess_merged['days_before_deadline'] = assess_merged['date'] - assess_merged['date_submitted']
    score_agg = assess_merged.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        avg_assessment_score=('score', 'mean'),
        n_assessments_submitted=('score', 'count'),
        avg_days_before_deadline=('days_before_deadline', 'mean'),
    ).reset_index()

    df = student_info.merge(pivot, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df = df.merge(total_clicks, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df = df.merge(score_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df['total_clicks'] = df['total_clicks'].fillna(0)
    for c in activity_cols:
        df[c] = df[c].fillna(0)
    return df, activity_cols


def make_pipeline(activity_cols, other_num, cat_features, model):
    """CV-safe pipeline: quartile binning of raw per-activity-type click
    counts happens INSIDE the pipeline (via TrainQuantileBinner), not before
    it, so RandomizedSearchCV's internal CV folds are properly isolated."""
    preprocess = ColumnTransformer([
        ('activity_q', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('qbin', TrainQuantileBinner(q=4)),
        ]), activity_cols),
        ('num', Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', StandardScaler())]), other_num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features),
    ])
    return Pipeline([('prep', preprocess), ('clf', model)])
