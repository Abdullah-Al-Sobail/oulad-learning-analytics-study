import sys, time, json, pickle
sys.path.insert(0, '/home/claude')
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from oulad_pipeline import build_raw_features, make_pipeline, CAT, ENROLL_NUM, OTHER_NUM

t0 = time.time()
np.random.seed(42)

student_info = pd.read_csv('oulad/studentInfo.csv')
student_vle = pd.read_csv('oulad/studentVle.csv')
student_assessment = pd.read_csv('oulad/studentAssessment.csv')
assessments = pd.read_csv('oulad/assessments.csv')
vle = pd.read_csv('oulad/vle.csv')

le = LabelEncoder()
y_all = le.fit_transform(student_info['final_result'])
class_names = list(le.classes_)

print("Building full-course features (corrected days-before-deadline)...", time.time()-t0)
df_full, act_cols_full = build_raw_features(student_info, student_vle, student_assessment, assessments, vle, cutoff_day=None)
y_full = le.transform(df_full['final_result'])
train_idx, test_idx = train_test_split(df_full.index, test_size=0.2, stratify=y_full, random_state=42)
y_train, y_test = y_full[train_idx], y_full[test_idx]

print("Building early (day<=56) features...", time.time()-t0)
df_early, act_cols_early = build_raw_features(student_info, student_vle, student_assessment, assessments, vle, cutoff_day=56)
df_early = df_early.set_index(df_full.index)

def eval_model(pipe, X_test, y_test):
    preds = pipe.predict(X_test); proba = pipe.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(pd.get_dummies(y_test), proba, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')
    return dict(accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4), f1_macro=round(f1,4), roc_auc=round(auc,4)), preds, proba

RF_BEST = dict(n_estimators=400, max_depth=None, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=1)
LR_BEST = dict(max_iter=2000, C=10)

X_train_full = df_full.loc[train_idx, act_cols_full + OTHER_NUM + CAT]
X_test_full = df_full.loc[test_idx, act_cols_full + OTHER_NUM + CAT]

print("Fitting MAIN end-of-course RF (CV-safe quartile binning, 400 trees)...", time.time()-t0)
pipe_full_rf = make_pipeline(act_cols_full, OTHER_NUM, CAT, RandomForestClassifier(**RF_BEST))
pipe_full_rf.fit(X_train_full, y_train)
metrics_full_rf, preds_full_rf, proba_full_rf = eval_model(pipe_full_rf, X_test_full, y_test)
print("END-OF-COURSE RF:", metrics_full_rf, time.time()-t0)

print("Fitting MAIN end-of-course LR...", time.time()-t0)
pipe_full_lr = make_pipeline(act_cols_full, OTHER_NUM, CAT, LogisticRegression(**LR_BEST))
pipe_full_lr.fit(X_train_full, y_train)
metrics_full_lr, preds_full_lr, proba_full_lr = eval_model(pipe_full_lr, X_test_full, y_test)
print("END-OF-COURSE LR:", metrics_full_lr, time.time()-t0)

X_train_early = df_early.loc[train_idx, act_cols_early + OTHER_NUM + CAT]
X_test_early = df_early.loc[test_idx, act_cols_early + OTHER_NUM + CAT]
print("Fitting EARLY (day<=56) RF...", time.time()-t0)
pipe_early_rf = make_pipeline(act_cols_early, OTHER_NUM, CAT, RandomForestClassifier(**RF_BEST))
pipe_early_rf.fit(X_train_early, y_train)
metrics_early_rf, preds_early_rf, proba_early_rf = eval_model(pipe_early_rf, X_test_early, y_test)
print("EARLY RF:", metrics_early_rf, time.time()-t0)

with open('checkpoint_v2.pkl', 'wb') as f:
    pickle.dump(dict(
        df_full=df_full, act_cols_full=act_cols_full, train_idx=train_idx, test_idx=test_idx,
        y_full=y_full, y_train=y_train, y_test=y_test, class_names=class_names,
        pipe_full_rf=pipe_full_rf, preds_full_rf=preds_full_rf, proba_full_rf=proba_full_rf, metrics_full_rf=metrics_full_rf,
        metrics_full_lr=metrics_full_lr, metrics_early_rf=metrics_early_rf,
    ), f)

RESULTS = {'end_of_course_RF': metrics_full_rf, 'end_of_course_LR': metrics_full_lr, 'early_day56_RF': metrics_early_rf}
with open('validation_results_v2.json', 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("DONE stage1_v2", time.time()-t0)
