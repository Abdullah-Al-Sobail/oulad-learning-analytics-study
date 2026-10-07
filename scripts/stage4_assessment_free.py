import sys, time, json, pickle
sys.path.insert(0, '/home/claude')
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from oulad_pipeline import build_raw_features, make_pipeline, CAT, ENROLL_NUM, OTHER_NUM

t0 = time.time()
with open('checkpoint_v2.pkl', 'rb') as f:
    ck = pickle.load(f)
df_full = ck['df_full']; train_idx = ck['train_idx']; test_idx = ck['test_idx']
y_train = ck['y_train']; y_test = ck['y_test']

student_info = pd.read_csv('oulad/studentInfo.csv')
student_vle = pd.read_csv('oulad/studentVle.csv')
student_assessment = pd.read_csv('oulad/studentAssessment.csv')
assessments = pd.read_csv('oulad/assessments.csv')
vle = pd.read_csv('oulad/vle.csv')

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
    return dict(accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4), f1_macro=round(f1,4), roc_auc=round(auc,4))

RF_BEST = dict(n_estimators=400, max_depth=None, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=1)

# Assessment-free: demographics + enrollment + engagement only, NO assessment score/count/timing features
OTHER_NUM_NO_ASSESS = ENROLL_NUM + ['total_clicks']
X_train_noassess = df_early.loc[train_idx, act_cols_early + OTHER_NUM_NO_ASSESS + CAT]
X_test_noassess = df_early.loc[test_idx, act_cols_early + OTHER_NUM_NO_ASSESS + CAT]

print("Fitting ASSESSMENT-FREE early (day<=56) RF (demographics+enrollment+engagement only)...", time.time()-t0)
pipe_noassess = make_pipeline(act_cols_early, OTHER_NUM_NO_ASSESS, CAT, RandomForestClassifier(**RF_BEST))
pipe_noassess.fit(X_train_noassess, y_train)
metrics_noassess = eval_model(pipe_noassess, X_test_noassess, y_test)
print("ASSESSMENT-FREE EARLY RF:", metrics_noassess, time.time()-t0)

RESULTS = json.load(open('validation_results_v2.json'))
RESULTS['early_day56_assessment_free_RF'] = metrics_noassess
with open('validation_results_v2.json', 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("DONE stage4", time.time()-t0)
