import pandas as pd, numpy as np, json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                              classification_report)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

np.random.seed(42)

CAT = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability', 'code_module']
ENROLL_NUM = ['num_of_prev_attempts', 'studied_credits']
OTHER_NUM = ['total_clicks', 'avg_assessment_score', 'n_assessments_submitted', 'avg_days_before_deadline']

class TrainQuantileBinner(BaseEstimator, TransformerMixin):
    """Learns qcut quartile bin edges per column ONLY from the data it is
    fit on. Placed inside the sklearn Pipeline (not applied before it), so
    RandomizedSearchCV's internal CV folds each learn their own edges from
    that fold's training rows only -- properly leakage-safe across folds,
    not just across the outer train/test split."""
    def __init__(self, q=4):
        self.q = q
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.edges_ = {}
        for col in self.columns_:
            vals = pd.to_numeric(X[col], errors='coerce').fillna(0)
            if vals.nunique() <= 1:
                self.edges_[col] = None; continue
            try:
                _, edges = pd.qcut(vals, q=self.q, retbins=True, duplicates='drop')
                edges = np.array(edges, dtype=float)
                edges[0] = -np.inf; edges[-1] = np.inf
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
            out[col] = 0 if edges is None else pd.cut(vals, bins=edges, labels=False, include_lowest=True).fillna(-1)
        return out.values

# 1. Load OULAD tables
student_info = pd.read_csv('oulad/studentInfo.csv')
student_vle = pd.read_csv('oulad/studentVle.csv')
student_assessment = pd.read_csv('oulad/studentAssessment.csv')
assessments = pd.read_csv('oulad/assessments.csv')
vle = pd.read_csv('oulad/vle.csv')

print('Total enrollments:', len(student_info), '| Unique students:', student_info['id_student'].nunique())

# 2. Raw per-activity-type engagement aggregates
svle = student_vle.merge(vle[['id_site', 'code_module', 'code_presentation', 'activity_type']],
                          on=['id_site', 'code_module', 'code_presentation'], how='left')
per_activity = svle.groupby(['id_student', 'code_module', 'code_presentation', 'activity_type'])['sum_click'].sum().reset_index()
total_clicks = svle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index().rename(columns={'sum_click': 'total_clicks'})
pivot = per_activity.pivot_table(index=['id_student', 'code_module', 'code_presentation'],
                                  columns='activity_type', values='sum_click', fill_value=0).reset_index()
activity_cols = [c for c in pivot.columns if c not in ['id_student', 'code_module', 'code_presentation']]

# 3. Assessment features -- CORRECTED: avg_days_before_deadline now uses the
#    actual deadline ('date' in assessments.csv) minus submission date, not
#    the mean raw submission date (an earlier version of this script had this bug).
assess_merged = student_assessment.merge(assessments, on='id_assessment')  # has BOTH 'date' (deadline) and 'date_submitted'
assess_merged['days_before_deadline'] = assess_merged['date'] - assess_merged['date_submitted']
score_agg = assess_merged.groupby(['id_student', 'code_module', 'code_presentation']).agg(
    avg_assessment_score=('score', 'mean'),
    n_assessments_submitted=('score', 'count'),
    avg_days_before_deadline=('days_before_deadline', 'mean'),  # NaN for the 11 Exam rows lacking a 'date'; excluded via pandas default skipna
).reset_index()

# 4. Merge RAW (non-leaky) features into the modeling table
df = student_info.merge(pivot, on=['id_student', 'code_module', 'code_presentation'], how='left')
df = df.merge(total_clicks, on=['id_student', 'code_module', 'code_presentation'], how='left')
df = df.merge(score_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')
df['total_clicks'] = df['total_clicks'].fillna(0)
for c in activity_cols:
    df[c] = df[c].fillna(0)

target = 'final_result'
le = LabelEncoder()
y_all = le.fit_transform(df[target])
class_names = le.classes_

# 5. Split FIRST
train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=y_all, random_state=42)
X_train = df.loc[train_idx, activity_cols + OTHER_NUM + CAT]
X_test = df.loc[test_idx, activity_cols + OTHER_NUM + CAT]
y_train, y_test = y_all[train_idx], y_all[test_idx]

# 6. CV-SAFE pipeline: quartile binning happens INSIDE the pipeline so each
#    RandomizedSearchCV inner CV fold learns its own bin edges.
preprocess = ColumnTransformer([
    ('activity_q', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                              ('qbin', TrainQuantileBinner(q=4))]), activity_cols),
    ('num', Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', StandardScaler())]), OTHER_NUM),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), CAT),
])

param_dists = {
    'LogisticRegression': (LogisticRegression(max_iter=2000), {'clf__C': [0.01, 0.1, 1, 10]}),
    'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'),
        {'clf__n_estimators': [200, 400], 'clf__max_depth': [None, 10, 20], 'clf__min_samples_leaf': [1, 2, 4]}),
    'XGBoost': (XGBClassifier(eval_metric='mlogloss', random_state=42),
        {'clf__n_estimators': [200, 400], 'clf__max_depth': [3, 6, 9], 'clf__learning_rate': [0.05, 0.1, 0.2]}),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, (model, dist) in param_dists.items():
    pipe = Pipeline([('prep', preprocess), ('clf', model)])
    search = RandomizedSearchCV(pipe, dist, n_iter=6, scoring='f1_macro', cv=cv, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    preds = best.predict(X_test)
    proba = best.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    # Explicit class ordering via label_binarize (classes=0..n-1, matching
    # LabelEncoder's output) rather than pd.get_dummies, so column order is
    # guaranteed to match proba's column order regardless of which classes
    # happen to appear in y_test.
    y_onehot = label_binarize(y_test, classes=list(range(len(class_names))))
    auc = roc_auc_score(y_onehot, proba, average='macro', multi_class='ovr')
    brier = np.mean(np.sum((proba - y_onehot) ** 2, axis=1))
    results[name] = dict(accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4),
                          f1_macro=round(f1,4), roc_auc=round(auc,4), brier=round(brier,4),
                          best_params=search.best_params_)
    print(f'\n=== {name} ===')
    print(results[name])
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

with open('results_final.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('\nSaved results_final.json')
