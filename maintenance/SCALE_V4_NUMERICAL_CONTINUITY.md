# Scale-v4 numerical continuity audit

The primary scale-v4 record preserves the earlier root and legacy graph readouts exactly on their shared analysis molecules. Apparent changes in selected report tables reflect ordinary versus balanced accuracy, not a change in the matched results.

## Frozen feature continuity

Align the hard-v2 archived `ids` with the scale-v4 archived `ids`, retaining the original 1,377 analysis molecules in the same order. Compare all nine A/B/random checkpoints at all depths 0 through 8. The new `root` scope corresponds to the archived `local` readout; `legacy_graph` corresponds to `graph`.

| Scope | Arrays | Float32 coordinates | Unequal coordinates | Maximum absolute difference | Median absolute difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| root | 81 | 7,138,368 | 0 | 0 | 0 |
| legacy_graph | 81 | 49,968,576 | 0 | 0 | 0 |

All 162 corresponding arrays are byte-for-byte equal after identity alignment, totaling 57,106,944 float32 coordinates. No numerical drift is observed in this comparison.

## Matched readout continuity

Match the 216 rows of `readout_replay_v3/ridge_diagnostic.csv` to trained, depth-8 root/legacy rows in `scale_v4/adaptation.csv`, using experience, encoder seed, support repeat, scope, new/replay support, and test split. All fit counts and test counts match. Both ordinary accuracy and balanced accuracy are exactly equal in all 216 row pairs; maximum same-metric differences are zero.

The following are repeat-0 replay means across the three encoder seeds. These two metrics weight test examples differently:

| Experience | Scope | Test | Ordinary accuracy (%) | Balanced accuracy (%) |
| --- | --- | --- | ---: | ---: |
| A | root | ood_common | 74.943567 | 73.000248 |
| A | root | iid_a | 56.998557 | 55.620621 |
| A | legacy_graph | ood_common | 93.378480 | 93.180299 |
| A | legacy_graph | iid_a | 94.733045 | 94.693220 |
| B | root | ood_common | 62.753950 | 62.147740 |
| B | root | iid_b | 63.488701 | 64.195555 |
| B | legacy_graph | ood_common | 90.368698 | 90.272928 |
| B | legacy_graph | iid_b | 89.759887 | 89.861708 |

For example, A root replay on common OOD is 74.943567% ordinary accuracy and 73.000248% balanced accuracy in both records. A legacy graph replay is 93.378480% ordinary accuracy and 93.180299% balanced accuracy in both records. Comparisons must use the same metric; the visible 74.9/73.0 and 93.4/93.2 differences are not experiment revisions.

## Numerical controls and limits

Scale-v4 keeps each inference batch at the configured size by repeating valid graphs in the final batch and discarding repeated outputs. Its non-root matched-width means accumulate in float64 before returning float32; root and legacy pooling retain their inherited operations. A pre-primary smoke check found that an unfilled 65-row CUDA tail could create two nominally constant layer-0 vectors, differing by approximately 3e-8 to 6e-8. Fixed-shape batching removed that artifact. That diagnostic motivates the numerical control; it does not establish a difference in the completed primary comparisons above.

This audit does not retrain a model, refit a predictor, infer general equality for every possible CUDA batch shape, or compare raw features for the non-archived v3 support inference. It establishes exact equality of the archived shared analysis features and the matched stored readout scores. Completed archives, weights, predictions, and manifests remain unchanged.

## Reproduction and evidence

Run `python maintenance/audit_scale_continuity.py --root . --output maintenance/ANOTHER_CONTINUITY_AUDIT` to create a separate audit without overwriting existing files. The corresponding JSON contains all 162 feature comparisons, all 216 metric pairs, and hashes of the inputs checked against their archived manifests.
