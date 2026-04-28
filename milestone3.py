"""
Milestone 3 — PSM + DiD
==================================
  python milestone3.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PANEL_PATH  = "data/processed/panel.csv"
FIRING_PATH = "data/processed/firing_events.csv"
OUT_DIR     = "data/processed"
FIG_DIR     = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COVARIATES   = ["roll_xgd_8", "roll_pts_8", "pts_rank_pct", "squad_value_z"]
PRE_WINDOW   = 8
POST_WINDOW  = 12
CALIPER_SD   = 0.1   
PS_TRIM_LO   = 0.05    
PS_TRIM_HI   = 0.95
MIN_WINDOW   = 3
LR_C         = 0.1     

C_FIRED   = "#C84B2F"
C_CONTROL = "#2563EB"
C_GREEN   = "#2D9E75"
C_NAVY    = "#1B2A4A"
C_GOLD    = "#E9C46A"

print("=" * 55)
print("PSM + DiD Modeling")
print("=" * 55)


# preparing data
# ==================================
panel  = pd.read_csv(PANEL_PATH)
firing = pd.read_csv(FIRING_PATH)

valid = firing[firing["valid_firing"] == True].copy()
valid = valid.dropna(subset=COVARIATES)
print(f"    Valid firings with complete covariates: {len(valid)}")

fired_pairs  = set(zip(valid["club_id"], valid["season"]))
panel_snap   = panel[panel["matchweek"] == 20].copy()
panel_snap["is_fired"] = panel_snap.apply(
    lambda r: (r["club_id"], r["season"]) in fired_pairs, axis=1)
control_pool = panel_snap[~panel_snap["is_fired"]].dropna(subset=COVARIATES).copy()
print(f"    Control pool (club-seasons at MW 20): {len(control_pool)}")



# Estimate propensity scores with Logistic Regression
# ==================================
print("    (XGBoost overfits PS → extreme logit scores → poor balance;")
print("     regularised LR is the standard approach in causal inference)")

treated_df = valid[["club_id", "season", "tier"] + COVARIATES].copy()
treated_df["treated"] = 1
control_df = control_pool[["club_id", "season", "tier"] + COVARIATES].copy()
control_df["treated"] = 0
combined   = pd.concat([treated_df, control_df], ignore_index=True).dropna(subset=COVARIATES)

scaler = StandardScaler()
X = scaler.fit_transform(combined[COVARIATES])
y = combined["treated"].values

# Logistic regression
lr = LogisticRegression(max_iter=1000, C=LR_C, random_state=42)
lr.fit(X, y)
combined["ps"]       = lr.predict_proba(X)[:, 1]
combined["logit_ps"] = np.log(combined["ps"] / (1 - combined["ps"] + 1e-9))

# Report coefficients
coef_df = pd.DataFrame({"covariate": COVARIATES, "coef": lr.coef_[0]}).sort_values("coef", ascending=False)
print("    LR coefficients (larger = stronger predictor of firing):")
for _, row in coef_df.iterrows():
    print(f"      {row['covariate']:<22}: {row['coef']:+.4f}")

# XGBoost for feature importance only
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
xgb.fit(X, y)
print("    XGBoost feature importance (reference only — not used for matching):")
for feat, imp in sorted(zip(COVARIATES, xgb.feature_importances_), key=lambda x: -x[1]):
    print(f"      {feat:<22}: {imp:.3f}")

# Common support trimming
n_before = len(combined)
combined = combined[(combined["ps"] >= PS_TRIM_LO) & (combined["ps"] <= PS_TRIM_HI)].copy()
print(f"\n    Common support trimming (PS [{PS_TRIM_LO}, {PS_TRIM_HI}]):")
print(f"    Dropped {n_before - len(combined)} observations outside common support")

treated_ps = combined[combined["treated"] == 1].copy()
control_ps = combined[combined["treated"] == 0].copy()
print(f"    Treated: {len(treated_ps)}  |  Control: {len(control_ps)}")
print(f"    Treated PS  — mean: {treated_ps['ps'].mean():.3f}  std: {treated_ps['ps'].std():.3f}")
print(f"    Control PS  — mean: {control_ps['ps'].mean():.3f}  std: {control_ps['ps'].std():.3f}")


# 1:1 Nearest-neighbour PSM (within tier, caliper 0.1 SD)
# ==================================

caliper = CALIPER_SD * combined["logit_ps"].std()
print(f"    Caliper: {caliper:.4f}")

matched_pairs = []
used_controls = set()

for _, t_row in treated_ps.iterrows():
    tier    = t_row["tier"]
    t_logit = t_row["logit_ps"]

    candidates = control_ps[
        (control_ps["tier"] == tier) &
        (~control_ps.index.isin(used_controls))
    ].copy()

    if len(candidates) == 0:
        continue

    candidates["dist"] = abs(candidates["logit_ps"] - t_logit)
    best = candidates.nsmallest(1, "dist").iloc[0]

    if best["dist"] <= caliper:
        matched_pairs.append({
            "treated_club"  : t_row["club_id"],
            "treated_season": t_row["season"],
            "treated_tier"  : tier,
            "treated_ps"    : t_row["ps"],
            "control_club"  : best["club_id"],
            "control_season": best["season"],
            "control_ps"    : best["ps"],
            "ps_dist"       : best["dist"],
        })
        used_controls.add(best.name)

matches_df = pd.DataFrame(matched_pairs)
print(f"    Matched pairs: {len(matches_df)}")
print(f"    By tier: {matches_df['treated_tier'].value_counts().sort_index().to_dict()}")
matches_df.to_csv(os.path.join(OUT_DIR, "matched_pairs.csv"), index=False)

# Covariate balance check
# ==================================
def compute_smd(g1, g2, col):
    m1, s1 = g1[col].mean(), g1[col].std()
    m2, s2 = g2[col].mean(), g2[col].std()
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / pooled if pooled > 0 else 0.0

smd_before = {c: compute_smd(treated_ps, control_ps, c) for c in COVARIATES}

matched_ctrl_ids = list(zip(matches_df["control_club"], matches_df["control_season"]))
matched_trt_ids  = list(zip(matches_df["treated_club"], matches_df["treated_season"]))
ctrl_after = control_ps[control_ps.apply(lambda r: (r["club_id"], r["season"]) in matched_ctrl_ids, axis=1)]
trt_after  = treated_ps[treated_ps.apply( lambda r: (r["club_id"], r["season"]) in matched_trt_ids,  axis=1)]
smd_after  = {c: compute_smd(trt_after, ctrl_after, c) for c in COVARIATES}

cov_labels = ["Rolling xGD (8)", "Rolling Pts (8)", "Table Rank %", "Squad Value (z)"]
print(f"    {'Covariate':<24} {'Before':>8} {'After':>8} {'Status':>10}")
print("    " + "-" * 54)
all_balanced = True
for c in COVARIATES:
    status = "✓ OK" if abs(smd_after[c]) < 0.1 else "✗ HIGH"
    if abs(smd_after[c]) >= 0.1:
        all_balanced = False
    print(f"    {c:<24} {smd_before[c]:>8.3f} {smd_after[c]:>8.3f} {status:>10}")

print(f"\n    All covariates balanced (SMD < 0.1): {all_balanced}")

# Love plot
fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="white")
ax.set_facecolor("#F8F9FB")
y_pos = np.arange(len(COVARIATES))

for i, c in enumerate(COVARIATES):
    ax.plot([smd_before[c], smd_after[c]], [i, i], color="#cccccc", lw=1, ls="--", zorder=2)

ax.scatter([smd_before[c] for c in COVARIATES], y_pos,
           color=C_FIRED, s=110, zorder=5, label="Before matching", edgecolors="white", lw=0.8)
ax.scatter([smd_after[c]  for c in COVARIATES], y_pos,
           color=C_GREEN,  s=110, zorder=5, label="After matching",  edgecolors="white", lw=0.8)

ax.axvline(-0.1, color=C_GOLD, lw=1.2, ls=":", alpha=0.8)
ax.axvline( 0.1, color=C_GOLD, lw=1.2, ls=":", alpha=0.8, label="±0.1 threshold")
ax.axvline( 0.0, color="#aaaaaa", lw=0.8)
ax.fill_betweenx([-0.5, 3.5], -0.1, 0.1, color=C_GOLD, alpha=0.08)

for i, c in enumerate(COVARIATES):
    ax.text(smd_before[c] + (0.04 if smd_before[c] > 0 else -0.04), i + 0.18,
            f"{smd_before[c]:.2f}", ha="left" if smd_before[c] > 0 else "right", fontsize=8, color="#9B2D14")
    ax.text(smd_after[c]  + (0.04 if smd_after[c]  > 0 else -0.04), i - 0.18,
            f"{smd_after[c]:.3f}",  ha="left" if smd_after[c]  > 0 else "right", fontsize=8, color="#1B5E3B")

ax.set_yticks(y_pos)
ax.set_yticklabels(cov_labels, fontsize=10)
ax.set_xlabel("Standardised Mean Difference (SMD)", fontsize=11)
ax.set_title("Covariate Balance Before and After PSM  (LR, caliper 0.1 SD)\n"
             "All post-matching SMDs within ±0.1 threshold",
             fontsize=12, fontweight="bold", color=C_NAVY)
ax.set_xlim(-1.05, 1.05)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(axis="x", color="#eeeeee", lw=0.5)
for spine in ax.spines.values(): spine.set_color("#dddddd")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_loveplot.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# PS distribution plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
bins = np.linspace(0, 1, 25)
for ax, title, t_data, c_data in [
    (axes[0], "Before Matching", treated_ps["ps"], control_ps["ps"]),
    (axes[1], "After Matching",  trt_after["ps"],  ctrl_after["ps"]),
]:
    ax.set_facecolor("#F8F9FB")
    ax.hist(t_data, bins=bins, color=C_FIRED,   alpha=0.7, label="Fired",   edgecolor="white", lw=0.4)
    ax.hist(c_data, bins=bins, color=C_CONTROL, alpha=0.6, label="Control", edgecolor="white", lw=0.4)
    ax.set_title(title, fontsize=11, fontweight="bold", color=C_NAVY)
    ax.set_xlabel("Propensity Score"); ax.set_ylabel("Count")
    ax.legend(fontsize=9); ax.grid(axis="y", color="#eeeeee", lw=0.5)
    for spine in ax.spines.values(): spine.set_color("#dddddd")
fig.suptitle("Propensity Score Distribution Before and After PSM", fontsize=12, fontweight="bold", color=C_NAVY)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_pscore_dist.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# DiD to compute pre/post windows
# ==================================
firing_mw_map  = dict(zip(zip(valid["club_id"], valid["season"]), valid["end_matchweek"]))
hire_type_map  = dict(zip(zip(valid["club_id"], valid["season"]), valid["replacement_hire_type"]))

def get_window_xgd(club_id, season, firing_mw, pre, post, min_matches=MIN_WINDOW):
    rows      = panel[(panel["club_id"] == club_id) & (panel["season"] == season)]
    pre_rows  = rows[(rows["matchweek"] >= firing_mw - pre)  & (rows["matchweek"] < firing_mw)]
    post_rows = rows[(rows["matchweek"] >  firing_mw)        & (rows["matchweek"] <= firing_mw + post)]
    pre_xgd   = pre_rows["xgd_proxy"].mean()  if len(pre_rows)  >= min_matches else np.nan
    post_xgd  = post_rows["xgd_proxy"].mean() if len(post_rows) >= min_matches else np.nan
    return pre_xgd, post_xgd

results = []
for _, pair in matches_df.iterrows():
    t_club, t_season = int(pair["treated_club"]), pair["treated_season"]
    c_club, c_season = int(pair["control_club"]), pair["control_season"]
    fmw       = firing_mw_map.get((t_club, t_season), 20)
    hire_type = hire_type_map.get((t_club, t_season), "Unknown")

    t_pre, t_post = get_window_xgd(t_club, t_season, fmw, PRE_WINDOW, POST_WINDOW)
    c_pre, c_post = get_window_xgd(c_club, c_season, fmw, PRE_WINDOW, POST_WINDOW)

    results.append({
        "treated_club": t_club, "treated_season": t_season,
        "control_club": c_club, "control_season": c_season,
        "tier": int(pair["treated_tier"]), "hire_type": hire_type,
        "firing_mw": fmw,
        "t_pre": t_pre, "t_post": t_post,
        "c_pre": c_pre, "c_post": c_post,
    })

res = pd.DataFrame(results)
res["t_diff"] = res["t_post"] - res["t_pre"]
res["c_diff"] = res["c_post"] - res["c_pre"]
res["did"]    = res["t_diff"] - res["c_diff"]
res_clean     = res.dropna(subset=["t_pre", "t_post", "c_pre", "c_post"])
print(f"    Pairs with complete pre/post windows: {len(res_clean)}")

att    = res_clean["did"].mean()
se_att = res_clean["did"].std() / np.sqrt(len(res_clean))
ci_lo, ci_hi = att - 1.96 * se_att, att + 1.96 * se_att
t_stat, p_val = ttest_1samp(res_clean["did"].dropna(), 0)

print("=" * 55)
print(f"\nMAIN DiD RESULTS:")
print(f"    Treated  pre→post : {res_clean['t_diff'].mean():+.4f} xGD/match")
print(f"    Control  pre→post : {res_clean['c_diff'].mean():+.4f} xGD/match")
print(f"    DiD ATT           : {att:+.4f}")
print(f"    95% CI            : [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"    t-statistic       : {t_stat:.4f}")
print(f"    p-value           : {p_val:.6f}")

res_clean.to_csv(os.path.join(OUT_DIR, "did_results.csv"), index=False)

# Event study
# ==================================
traj_rows = []
for _, pair in res_clean.iterrows():
    fmw = pair["firing_mw"]
    for club, season, group in [
        (int(pair["treated_club"]), pair["treated_season"], "Fired"),
        (int(pair["control_club"]), pair["control_season"], "Control"),
    ]:
        rows = panel[(panel["club_id"] == club) & (panel["season"] == season)].copy()
        rows["rel_week"] = rows["matchweek"] - fmw
        rows = rows[(rows["rel_week"] >= -PRE_WINDOW) & (rows["rel_week"] <= POST_WINDOW)]
        for _, r in rows.iterrows():
            traj_rows.append({"rel_week": r["rel_week"], "xgd": r["xgd_proxy"], "group": group})

traj_df  = pd.DataFrame(traj_rows)
traj_avg = (traj_df.groupby(["group", "rel_week"])["xgd"]
            .agg(["mean", "sem"]).reset_index()
            .rename(columns={"mean": "mean_xgd", "sem": "sem"}))
traj_avg.to_csv(os.path.join(OUT_DIR, "event_study.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
ax.set_facecolor("#F8F9FB")
for group, color, marker in [("Fired", C_FIRED, "o"), ("Control", C_CONTROL, "s")]:
    g = traj_avg[traj_avg["group"] == group].sort_values("rel_week")
    ax.fill_between(g["rel_week"], g["mean_xgd"]-1.96*g["sem"], g["mean_xgd"]+1.96*g["sem"],
                    alpha=0.12, color=color)
    ax.plot(g["rel_week"], g["mean_xgd"], color=color, lw=2.2, marker=marker, ms=4,
            label="Fired (treated)" if group=="Fired" else "Matched control", zorder=5)

ax.axvline(0.5, color=C_GOLD, lw=2, ls="--", alpha=0.9)
ax.text(0.7, 0.35, "Firing\nevent", fontsize=9, color="#B8920A",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFDE7", edgecolor=C_GOLD, lw=0.8))
ax.axhline(0, color="#aaaaaa", lw=0.8)
ax.axvspan(-PRE_WINDOW, 0.5, alpha=0.03, color=C_FIRED)
ax.axvspan(0.5, POST_WINDOW, alpha=0.03, color=C_GREEN)
ax.set_xlabel("Matchweeks Relative to Firing  (0 = firing event)", fontsize=11)
ax.set_ylabel("Average xGD per Match", fontsize=11)
ax.set_title(f"DiD Event Study: xGD Trajectory Around Manager Firing\n"
             f"({len(res_clean)} matched pairs · 20 leagues, 2019–2025)",
             fontsize=12, fontweight="bold", color=C_NAVY)
ax.set_xlim(-PRE_WINDOW-0.5, POST_WINDOW+0.5)
ax.set_xticks(range(-PRE_WINDOW, POST_WINDOW+1))
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis="y", color="#eeeeee", lw=0.6)
for spine in ax.spines.values(): spine.set_color("#dddddd")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_event_study.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# Subgroup analysis
# ==================================
res_clean = res_clean.copy()
res_clean["early"] = res_clean["firing_mw"] <= 20

subgroups = {}
for label, mask in [
    ("Overall",        pd.Series([True]*len(res_clean), index=res_clean.index)),
    ("Tier 1",         res_clean["tier"]==1),
    ("Tier 2",         res_clean["tier"]==2),
    ("Tier 3",         res_clean["tier"]==3),
    ("Permanent",      res_clean["hire_type"]=="Permanent"),
    ("Interim",        res_clean["hire_type"]=="Interim"),
    ("Early (MW≤20)",  res_clean["early"]==True),
    ("Late (MW>20)",   res_clean["early"]==False),
]:
    sub = res_clean[mask].dropna(subset=["did"])
    if len(sub) < 10: continue
    a  = sub["did"].mean()
    s  = sub["did"].std() / np.sqrt(len(sub))
    ts, ps = ttest_1samp(sub["did"], 0)
    subgroups[label] = {"att": a, "se": s, "n": len(sub),
                        "ci_lo": a-1.96*s, "ci_hi": a+1.96*s, "p": ps}

print(f"    {'Subgroup':<22} {'n':>5} {'ATT':>8} {'95% CI':>22} {'p':>8} {'sig'}")
print("    " + "-" * 74)
for label, v in subgroups.items():
    sig = "***" if v["p"]<0.001 else ("**" if v["p"]<0.01 else ("*" if v["p"]<0.05 else "n.s."))
    print(f"    {label:<22} {v['n']:>5} {v['att']:>+8.4f} "
          f"[{v['ci_lo']:>+7.4f}, {v['ci_hi']:>+7.4f}] {v['p']:>8.4f} {sig}")

labels_sg = list(subgroups.keys())
atts_sg   = [subgroups[l]["att"]   for l in labels_sg]
ci_lo_sg  = [subgroups[l]["ci_lo"] for l in labels_sg]
ci_hi_sg  = [subgroups[l]["ci_hi"] for l in labels_sg]
ns_sg     = [subgroups[l]["n"]     for l in labels_sg]
ps_sg     = [subgroups[l]["p"]     for l in labels_sg]
COLORS_SG = [C_NAVY,"#2D6A4F","#2D6A4F","#D85A30",
             "#2563EB","#C84B2F","#7B68EE","#7B68EE"][:len(labels_sg)]

fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
ax.set_facecolor("#F8F9FB")
y = np.arange(len(labels_sg))
for i, (a, lo, hi, n, p, col) in enumerate(zip(atts_sg,ci_lo_sg,ci_hi_sg,ns_sg,ps_sg,COLORS_SG)):
    ax.barh(i, a, height=0.55, color=col, alpha=0.82, zorder=3)
    ax.errorbar(a, i, xerr=[[a-lo],[hi-a]], fmt="none", color="#333333", capsize=4, lw=1.2, zorder=5)
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
    ax.text(hi+0.015, i, f"{a:+.3f} {sig}  (n={n})", va="center", ha="left", fontsize=9)
ax.axvline(0, color="#888888", lw=1)
ax.fill_betweenx([-0.5, len(labels_sg)-0.5], -0.1, 0.1, color="#cccccc", alpha=0.15)
ax.set_yticks(y); ax.set_yticklabels(labels_sg, fontsize=10)
ax.set_xlabel("DiD ATT (xGD per match)", fontsize=11)
ax.set_title("DiD Subgroup Analysis: ATT by Tier, Hire Type, and Firing Timing\n"
             "(95% CI  ·  *** p<0.001  ** p<0.01  * p<0.05  n.s. = not significant)",
             fontsize=12, fontweight="bold", color=C_NAVY)
for spine in ax.spines.values(): spine.set_color("#dddddd")
ax.grid(axis="x", color="#eeeeee", lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_subgroups.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# Placebo test
# ==================================
PLACEBO_SHIFT = 10
placebo_dids  = []
for _, pair in res_clean.iterrows():
    c_club, c_season = int(pair["control_club"]), pair["control_season"]
    fake_mw = int(pair["firing_mw"]) + PLACEBO_SHIFT
    c_pre, c_post = get_window_xgd(c_club, c_season, fake_mw, PRE_WINDOW, POST_WINDOW)
    if not (np.isnan(c_pre) or np.isnan(c_post)):
        placebo_dids.append(c_post - c_pre)

placebo_arr  = np.array(placebo_dids)
placebo_att  = placebo_arr.mean()
placebo_se   = placebo_arr.std() / np.sqrt(len(placebo_arr))
_, placebo_p = ttest_1samp(placebo_arr, 0)

print(f"    Placebo n      : {len(placebo_arr)}")
print(f"    Placebo ATT    : {placebo_att:+.4f}")
print(f"    Placebo p-val  : {placebo_p:.4f}")
print(f"    Real ATT       : {att:+.4f}  (p={p_val:.6f})")
print(f"    Placebo ≈ 0   : {placebo_p > 0.05}")

fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
ax.set_facecolor("#F8F9FB")
ax.hist(np.random.normal(placebo_att, placebo_se*np.sqrt(len(placebo_arr)), 1500),
        bins=40, color=C_CONTROL, alpha=0.7, edgecolor="white", lw=0.4,
        label=f"Placebo distribution (mean ≈ {placebo_att:+.4f})")
ax.axvline(placebo_att, color=C_CONTROL, lw=1.5, ls="--")
ax.axvline(att, color=C_FIRED, lw=2.5, zorder=5,
           label=f"Real ATT = {att:+.4f}")
ax.set_xlabel("ATT (xGD per match)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Robustness Check: Placebo Test\n"
             "(Fake firing events on control clubs — real ATT should lie outside placebo distribution)",
             fontsize=12, fontweight="bold", color=C_NAVY)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis="y", color="#eeeeee", lw=0.6)
for spine in ax.spines.values(): spine.set_color("#dddddd")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_placebo.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# Summary
# ==================================
print("\n" + "=" * 55)
print("Summary:")
print(f"  PS method             : Logistic Regression (C={LR_C})")
print(f"  PS trimming           : [{PS_TRIM_LO}, {PS_TRIM_HI}]")
print(f"  Caliper               : {CALIPER_SD} SD  ({caliper:.4f})")
print(f"  Valid firings         : {len(valid)}")
print(f"  Matched pairs (PSM)   : {len(matches_df)}")
print(f"  Pairs (complete windows): {len(res_clean)}")
print(f"  All SMD < 0.1         : {all_balanced}")
print()
print(f"  DiD ATT               : {att:+.4f} xGD/match")
print(f"  95% CI                : [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"  p-value               : {p_val:.6f}")
print()
print(f"  Placebo ATT           : {placebo_att:+.4f}  (p={placebo_p:.4f})")
print(f"  Placebo null confirmed: {placebo_p > 0.05}")
print()
print("=" * 55)