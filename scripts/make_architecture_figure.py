import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(6.8, 9))
ax.set_xlim(0, 11)
ax.set_ylim(0.5, 15.5)
ax.axis('off')

boxes = [
    ("Learning Data Layer", "LMS interaction logs, assessment\nrecords, demographic data", 13.2, "#DCE6F7"),
    ("Feature / Representation Layer", "Leakage-safe transformation into\nmodel-ready features (Sec. 4.2-4.3)", 11.0, "#DCE6F7"),
    ("Learner-Risk Modeling Layer", "Predictive models (LR / RF / XGBoost)\nempirically evaluated in this paper (Sec. 4)", 8.8, "#CFE8CF"),
    ("Pedagogical Decision Layer", "Risk estimate + diagnosis -> specific\nintervention recommendation (proposed)", 6.6, "#F7E6C9"),
    ("Generative Feedback Layer", "LLM generates feedback constrained by\nthe decision layer's recommendation (proposed)", 4.4, "#F7E6C9"),
    ("Learner Support", "Delivered feedback / intervention\nreaches the student", 2.2, "#F0D9D9"),
]

for label, sub, y, color in boxes:
    box = FancyBboxPatch((1.0, y-0.9), 8.0, 1.8, boxstyle="round,pad=0.08,rounding_size=0.12",
                          linewidth=1.3, edgecolor="#444444", facecolor=color)
    ax.add_patch(box)
    ax.text(5.0, y+0.35, label, ha='center', va='center', fontsize=11.5, fontweight='bold')
    ax.text(5.0, y-0.28, sub, ha='center', va='center', fontsize=8.7, color="#333333")

for i in range(len(boxes)-1):
    y_top = boxes[i][2] - 0.9
    y_bot = boxes[i+1][2] + 0.9
    arr = FancyArrowPatch((5.0, y_top), (5.0, y_bot), arrowstyle='-|>', mutation_scale=16,
                           linewidth=1.4, color="#333333")
    ax.add_patch(arr)

# Feedback loop: Learner Support back up to Learning Data Layer
loop = FancyArrowPatch((9.1, 2.2), (9.1, 13.2), connectionstyle="arc3,rad=0.35",
                        arrowstyle='-|>', mutation_scale=16, linewidth=1.3, color="#8B0000", linestyle='--')
ax.add_patch(loop)
ax.text(9.75, 7.7, "new learning data\n(feedback loop)", rotation=90, va='center', ha='center',
        fontsize=8.3, color="#8B0000")

ax.text(5.0, 15.0, "Proposed Conceptual Architecture", ha='center', fontsize=13.5, fontweight='bold')
ax.text(5.0, 0.3, "Green = empirically evaluated in this paper.  Orange = proposed, not yet empirically validated.",
        ha='center', fontsize=8, style='italic', color="#555555")

plt.tight_layout()
plt.savefig('architecture_figure.png', dpi=170, bbox_inches='tight')
print("saved")
