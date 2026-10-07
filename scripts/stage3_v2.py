import sys, time, json, pickle
sys.path.insert(0, '/home/claude')
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
from oulad_pipeline import CAT, OTHER_NUM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
with open('checkpoint_v2.pkl', 'rb') as f:
    ck = pickle.load(f)
pipe_full_rf = ck['pipe_full_rf']; preds_full_rf = ck['preds_full_rf']; proba_full_rf = ck['proba_full_rf']
y_test = ck['y_test']; class_names = ck['class_names']
df_full = ck['df_full']; test_idx = ck['test_idx']; act_cols_full = ck['act_cols_full']

RESULTS = json.load(open('validation_results_v2.json'))

# ---- Bootstrap 95% CI ----
print("Bootstrap CI...", time.time()-t0)
rng = np.random.RandomState(42)
n_boot = 1000
y_test_arr = np.array(y_test); preds_arr = np.array(preds_full_rf)
n = len(y_test_arr)
boot_acc, boot_f1 = [], []
for _ in range(n_boot):
    idx = rng.randint(0, n, n)
    boot_acc.append(accuracy_score(y_test_arr[idx], preds_arr[idx]))
    _, _, f1b, _ = precision_recall_fscore_support(y_test_arr[idx], preds_arr[idx], average='macro', zero_division=0)
    boot_f1.append(f1b)
RESULTS['bootstrap_CI_RF'] = dict(
    accuracy_mean=round(float(np.mean(boot_acc)),4),
    accuracy_ci=[round(float(np.percentile(boot_acc,2.5)),4), round(float(np.percentile(boot_acc,97.5)),4)],
    f1_macro_mean=round(float(np.mean(boot_f1)),4),
    f1_macro_ci=[round(float(np.percentile(boot_f1,2.5)),4), round(float(np.percentile(boot_f1,97.5)),4)],
)
print("Bootstrap CI:", RESULTS['bootstrap_CI_RF'], time.time()-t0)

# ---- Permutation importance (raw-feature level, 3 reps) ----
print("Permutation importance...", time.time()-t0)
X_test = df_full.loc[test_idx, act_cols_full + OTHER_NUM + CAT]
def f1_of(Xp):
    preds = pipe_full_rf.predict(Xp)
    _, _, f1b, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    return f1b
base_f1 = f1_of(X_test)
raw_feats = act_cols_full + OTHER_NUM + CAT
importances = {}
for feat in raw_feats:
    drops = []
    for rep in range(3):
        Xp = X_test.copy()
        Xp[feat] = Xp[feat].sample(frac=1, random_state=rep).values
        drops.append(base_f1 - f1_of(Xp))
    importances[feat] = round(float(np.mean(drops)), 4)
top_importances = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])
RESULTS['permutation_importance_top10_RF'] = top_importances
print("TOP 10:", top_importances, time.time()-t0)

# ---- Calibration curve ----
print("Calibration figure...", time.time()-t0)
pass_idx = class_names.index('Pass')
y_test_bin = (np.array(y_test) == pass_idx).astype(int)
prob_pass = proba_full_rf[:, pass_idx]
frac_pos, mean_pred = calibration_curve(y_test_bin, prob_pass, n_bins=10, strategy='quantile')
plt.figure(figsize=(5,5))
plt.plot(mean_pred, frac_pos, marker='o', label='Random Forest (Pass vs. rest)')
plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel('Mean predicted probability'); plt.ylabel('Observed frequency')
plt.title('Calibration (reliability) curve \u2014 Pass class')
plt.legend(); plt.tight_layout()
plt.savefig('calibration_curve_v2.png', dpi=150)
plt.close()
print("Saved calibration_curve_v2.png", time.time()-t0)

with open('validation_results_v2.json', 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("DONE stage3_v2", time.time()-t0)
