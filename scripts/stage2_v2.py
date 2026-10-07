import sys, time, json, pickle
sys.path.insert(0, '/home/claude')
import pandas as pd, numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from oulad_pipeline import build_raw_features, make_pipeline, CAT, ENROLL_NUM, OTHER_NUM

t0 = time.time()
with open('checkpoint_v2.pkl', 'rb') as f:
    ck = pickle.load(f)
df_full = ck['df_full']; act_cols_full = ck['act_cols_full']
train_idx = ck['train_idx']; test_idx = ck['test_idx']; y_full = ck['y_full']
y_train = ck['y_train']; y_test = ck['y_test']

RESULTS = json.load(open('validation_results_v2.json'))

def eval_model(pipe, X_test, y_test):
    preds = pipe.predict(X_test); proba = pipe.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(pd.get_dummies(y_test), proba, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')
    return dict(accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4), f1_macro=round(f1,4), roc_auc=round(auc,4))

def simple_metrics(preds, y_test):
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    return dict(accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4), f1_macro=round(f1,4))

RF_FAST = dict(n_estimators=150, max_depth=None, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=1)

# ---- Student-grouped split ----
print("Student-grouped split...", time.time()-t0)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
groups = df_full['id_student'].values
g_train_pos, g_test_pos = next(gss.split(df_full, y_full, groups=groups))
g_train_idx = df_full.index[g_train_pos]; g_test_idx = df_full.index[g_test_pos]
Xg_train = df_full.loc[g_train_idx, act_cols_full + OTHER_NUM + CAT]
Xg_test = df_full.loc[g_test_idx, act_cols_full + OTHER_NUM + CAT]
yg_train = y_full[g_train_pos]; yg_test = y_full[g_test_pos]
pipe_g = make_pipeline(act_cols_full, OTHER_NUM, CAT, RandomForestClassifier(**RF_FAST))
pipe_g.fit(Xg_train, yg_train)
metrics_group = eval_model(pipe_g, Xg_test, yg_test)
RESULTS['student_grouped_split_RF'] = metrics_group
print("STUDENT-GROUPED:", metrics_group, time.time()-t0)

# ---- Presentation/temporal holdout ----
print("Presentation holdout...", time.time()-t0)
is_test_pres = (df_full['code_presentation'] == '2014J').values
t_train_idx = df_full.index[~is_test_pres]; t_test_idx = df_full.index[is_test_pres]
Xt_train = df_full.loc[t_train_idx, act_cols_full + OTHER_NUM + CAT]
Xt_test = df_full.loc[t_test_idx, act_cols_full + OTHER_NUM + CAT]
yt_train = y_full[~is_test_pres]; yt_test = y_full[is_test_pres]
pipe_t = make_pipeline(act_cols_full, OTHER_NUM, CAT, RandomForestClassifier(**RF_FAST))
pipe_t.fit(Xt_train, yt_train)
metrics_temporal = eval_model(pipe_t, Xt_test, yt_test)
RESULTS['presentation_holdout_RF'] = metrics_temporal
RESULTS['presentation_holdout_test_n'] = int(is_test_pres.sum())
print("PRESENTATION HOLDOUT:", metrics_temporal, "n_test=", int(is_test_pres.sum()), time.time()-t0)

# ---- Baselines ----
print("Baselines...", time.time()-t0)
X_train_full = df_full.loc[train_idx, act_cols_full + OTHER_NUM + CAT]
X_test_full = df_full.loc[test_idx, act_cols_full + OTHER_NUM + CAT]
maj = DummyClassifier(strategy='most_frequent', random_state=42).fit(X_train_full, y_train)
strat = DummyClassifier(strategy='stratified', random_state=42).fit(X_train_full, y_train)
RESULTS['baseline_majority_class'] = simple_metrics(maj.predict(X_test_full), y_test)
RESULTS['baseline_stratified_dummy'] = simple_metrics(strat.predict(X_test_full), y_test)
print("BASELINES:", RESULTS['baseline_majority_class'], RESULTS['baseline_stratified_dummy'], time.time()-t0)

# ---- Feature-group ablation ----
print("Feature-group ablation...", time.time()-t0)
ablation = {}
configs = {
    'G1_demographics': ([], [], CAT),
    'G2_+enrollment': ([], ENROLL_NUM, CAT),
    'G3_+engagement': (act_cols_full, ENROLL_NUM + ['total_clicks'], CAT),
    'G4_+assessment(full)': (act_cols_full, OTHER_NUM, CAT),
}
for gname, (acts, nums, cats) in configs.items():
    feats = acts + nums + cats
    Xa_train = df_full.loc[train_idx, feats] if feats else df_full.loc[train_idx, CAT]
    Xa_test = df_full.loc[test_idx, feats] if feats else df_full.loc[test_idx, CAT]
    pipe_a = make_pipeline(acts, nums, cats, RandomForestClassifier(**RF_FAST))
    pipe_a.fit(Xa_train, y_train)
    m = eval_model(pipe_a, Xa_test, y_test)
    ablation[gname] = m
    print(f"  {gname}:", m, time.time()-t0)
RESULTS['feature_group_ablation_RF'] = ablation

with open('validation_results_v2.json', 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("DONE stage2_v2", time.time()-t0)
