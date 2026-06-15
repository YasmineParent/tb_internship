# CFS comparison design (for the Friday meeting)

Comparing causal-prior FasterRisk against established causal feature selection (CFS), per Nataliya's request. Her bar, quoted: "It might be difficult to show that your method performs (in terms of accuracy) better but if it is not worse, it is already good (and it provides us with an interpretable score)." So the target is parity-or-better accuracy plus the interpretable scorecard, not beating CFS on accuracy.

## The framing that matters

CFS methods (IAMB, HITON-MB, the 2025 GCFS) are feature *selectors*, evaluated in their own papers on Markov-blanket *recovery* against a known ground-truth graph (F1 / precision / recall of selected features). Our method produces an interpretable integer *scorecard*, evaluated on prediction, calibration, stability, and readability on data with no ground truth. Different tasks.

Decision (agreed): lead with the **downstream** comparison. A CFS method gives a feature set; we then fit a scorecard on it and compare to our causal-prior scorecard. This is our home turf and matches Nataliya's framing: a feature set is not deployable, an interpretable integer score is.

## Arms

| arm | what | CI test / mechanism | status |
|---|---|---|---|
| vanilla | FasterRisk, mu=0 | none | ready |
| causal | causal-prior FasterRisk (ours) | ges_cg (conditional-Gaussian) soft prior | ready |
| cfs_iamb | pyCausalFS IAMB -> FasterRisk | Fisher-Z (Gaussian), off-the-shelf | ready |
| cfs_hiton_mb | pyCausalFS HITON-MB -> FasterRisk | Fisher-Z (Gaussian), off-the-shelf | ready |
| cfs_cg | bnlearn IAMB -> FasterRisk | mi-cg (conditional-Gaussian) | ready |
| cfs_gcfs | GCFS (Ling et al. 2025) -> FasterRisk | gradient / AutoEncoder, no CI tests | optional, ~1 day to integrate |

Note on fairness: pyCausalFS only ships Fisher-Z (continuous) and chi-square (discrete), no mixed-data test (confirmed in its manual). So `cfs_iamb` / `cfs_hiton_mb` with Fisher-Z is using the library *as designed*; the "no CG test" is a property of those methods, not a handicap we imposed. `cfs_cg` (bnlearn mi-cg) is a fairer mixed-data baseline we added on top, more generous than the library itself.

## Axis 1: downstream accuracy (ready)

Preliminary FICO smoke (k=10, 3 splits, leakage-free, selections on a held-out set):

| method | test AUC | scorecard size | MB size |
|---|---|---|---|
| vanilla | 0.777 | 10 | – |
| causal (ours) | 0.777 | 10 | – |
| cfs_iamb (Fisher-Z) | 0.774 | 10 | 5 |
| cfs_hiton_mb (Fisher-Z) | 0.774 | 10 | 5 |
| cfs_cg (mi-cg) | 0.615 | 2 | 3 |

Reading: ours ties vanilla and edges the off-the-shelf CFS, the "not worse" bar holds. The CG baseline collapsed because mi-cg returned a 3-indicator blanket, too small to support a scorecard; a real and slightly awkward finding (even the "correct" CI test gives a poor hard pre-selection here), worth presenting honestly.

## Axis 2: selection stability (the key open decision)

Our stability story (the FICO/TB result) is that hard selection is unstable across data resamples and the soft causal prior stabilizes it. But measuring this fairly is subtle:

- Computing the CFS Markov blanket *once* (on a held-out set) makes CFS trivially stable (the selected set cannot change), which *hides* the very instability the comparison is about. The "support stability = 1.00" in the smoke is this artifact, not a CFS win.
- The honest measure: every method **re-selects per resample**, and we report the cross-resample Jaccard of the selected features. CFS's native jumpiness then shows, and that is where our soft-prior advantage should appear.

One nuance to settle: our q is itself a stability selection (B subsamples), while standard CFS is a single run. So a fully fair stability comparison should also offer a **stability-aggregated CFS** variant (run the MB on B subsamples, keep frequently-selected features). Decide whether to include that.

Metric: we report two stability numbers per arm. `stability_jaccard` is the mean pairwise Jaccard of selected supports, the same measure used throughout §6.2/§6.3, so the CFS comparison sits on the same axis as our existing results. But raw Jaccard is not chance-corrected and inflates when selections are small: the CFS arms select a 3-to-4-feature blanket while the causal arm roams over ~7, so a naive Jaccard comparison flatters CFS purely on set size. `stability_nogueira` (Nogueira and Brown, JMLR 2017) corrects for chance and for selection size against a common feature universe, the fair head-to-head number. Read the Nogueira column as the comparison and Jaccard as the continuity-with-our-own-results column.

## Recovery (optional secondary panel)

On the §6.1 synthetic we have ground truth, so we *can* compare selected support against the true MB (F1 / precision / recall), on common ground with the CFS literature and with GCFS. Honest caveat: GCFS is built for exactly this and may beat us on pure recovery; that is *not* our claim. If we include recovery, frame it as "our selection is competitive," not "we recover best," and keep the weight on downstream.

## Open decisions for Friday

1. Downstream only, or add a recovery panel? (Our contribution lives downstream; recovery is GCFS's home turf.)
2. Include GCFS? It is the SOTA Nataliya named, but it is a heavier dependency (their PyTorch repo) and its native task is recovery, so we would use it through the same downstream wrapper.
3. Stability: per-resample selection for all arms (yes), and do we add the stability-aggregated CFS variant for a fully fair test?
4. Datasets: FICO now; add TB (real, our application) and/or folktables and/or a recovery benchmark?

## Ready vs to build

- Ready: downstream accuracy (`experiments/causal_prior/real/fico_cfs.py`), pyCausalFS + CG arms.
- To build: per-resample stability comparison; GCFS integration; optional recovery panel.
