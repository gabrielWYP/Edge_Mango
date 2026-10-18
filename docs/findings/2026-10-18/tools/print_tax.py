"""Analysis tool for the 2026-10-18 findings, not part of the training pipeline.

Prints the FP taxonomy and score tables from taxonomy_summary.json (in --out-dir).
"""
import json
from _paths import resolve_roots
_,_,_OUT,_=resolve_roots(__doc__,repo=False,data=False)
OUT=str(_OUT)
S=json.load(open(OUT+"/taxonomy_summary.json"))
cats=["a_duplicate","b_localization","c_background_on_fruit","d_off_fruit"]
print("tag      imgs  GT  preds  TP   FP | a_dup  b_loc  c_bg  d_off | % a   % b   % c   % d | preds/img (GT/img)")
for t,s in S.items():
    c=s["counts"]; n=s["n_fp"]
    print(f"{t:8s} {s['n_images']:4d} {s['n_gt']:4d} {s['n_pred_025']:5d} {s['n_tp']:4d} {n:5d} | "+" ".join(f"{c[k]:5d}" for k in cats)+" | "+" ".join(f"{100*c[k]/n:5.1f}" for k in cats)+f" | {s['preds_per_image_mean']:.1f} (med {s['preds_per_image_median']:.0f}) ({s['gt_per_image_mean']:.2f})")
print()
print("tag      FPscore med  p90  max | TPscore med p90 | FP>=.5  FP>=.7  TP>=.5 | FP size px (w,h) med | TP size px med")
for t,s in S.items():
    print(f"{t:8s} {s['fp_score_median']:.3f} {s['fp_score_p90']:.3f} {s['fp_score_max']:.3f} | {s['tp_score_median']:.3f} {s['tp_score_p90']:.3f} | {s['fp_ge_0.5']:5d} {s['fp_ge_0.7']:5d} {s['tp_ge_0.5']:5d} | {s['fp_size_px_median'][0]:.0f}x{s['fp_size_px_median'][1]:.0f} | {s['tp_size_px_median'][0] or 0:.0f}x{s['tp_size_px_median'][1] or 0:.0f}")
print()
for t,s in S.items():
    print(t, {k:(round(v['median'],3),round(v['p90'],3),v['n_ge_0.5']) for k,v in s['score_by_cat'].items()})
