# Independent review of v6 branch output effects

The shallow branch changes continuous outputs on the common held-out combinations,
but its molecule-specific permutation effects are small. The results support weak
effects under these particular interventions, rather than complete nonuse of the
branch. This review concerns the archived v5 heads and readout vectors; it does
not interpret a new architecture or a new optimization run.

## Scope and numerical checks

The completed [v6 record](../experiments/pooling_v6/REPORT.md) uses source commit
`8a5e6997e7bff4ebc1e0d6438a9847fd8623dee2`. It contains all 24 augmented models:
two regimes, two branches, two training experiences, and three seeds. The main
results below use the 12 shallow-skip models. All three prediction tasks and all
three disjoint test splits remain in the underlying tables.

I independently recomputed effects from the saved dense float32 logits, using
float64 diagnostic arithmetic without importing the diagnostic implementation.
The checks covered 2,181,168 values in the 198,288 paired per-example rows and
32,832 statistics in the 576 summary rows, including means, sample standard
deviations, quantiles, label flips, and correction/error transitions. All checks
passed at absolute tolerance `2e-12`. All six files in the branch subrecord
matched the completed v6 manifest. The separate
[v6 auditor](POOLING_V6_AUDIT.json) also reconstructed every saved head-logit array
and found zero logit or archived-v5 prediction mismatches.

The correct-class margin is `z_truth - max(z_other)`. Every reported change is
`altered - intact`, so a positive signed margin change favors the correct class.
Centered-logit distance removes a common offset across classes. Its relative
version divides by the intact centered-logit norm within the same model and
example. Raw logits from separately trained heads are not calibrated to a common
scale; cross-condition magnitude differences are descriptive, not a measure of
which training regime is better.

## Common-OOD family predictions

Entries are the mean and sample standard deviation of three per-seed means. Each
seed evaluates the same 443 common-OOD molecular identities. Flip and relative
distance columns are percentages; margin columns use each head's logit units.

| Regime | Experience | Intervention | Prediction flips (%) | Signed margin change | Absolute margin change | Relative centered-logit distance (%) |
|---|---|---|---:|---:|---:|---:|
| End to end | A | Zero | 1.129 +/- 0.814 | +0.06089 +/- 0.02885 | 0.18792 +/- 0.02845 | 1.982 +/- 0.842 |
| End to end | A | Permute | 0 | +0.00102 +/- 0.00160 | 0.00552 +/- 0.00188 | 0.0608 +/- 0.0234 |
| End to end | B | Zero | 0.752 +/- 0.690 | +0.09655 +/- 0.01811 | 0.27237 +/- 0.10660 | 3.212 +/- 0.892 |
| End to end | B | Permute | 0 | +0.00041 +/- 0.00029 | 0.00687 +/- 0.00285 | 0.0725 +/- 0.0259 |
| Frozen readout | A | Zero | 0 | +0.07751 +/- 0.03318 | 0.14091 +/- 0.06072 | 2.631 +/- 0.436 |
| Frozen readout | A | Permute | 0 | +0.00016 +/- 0.00126 | 0.00608 +/- 0.00205 | 0.1125 +/- 0.0502 |
| Frozen readout | B | Zero | 0.226 +/- 0.226 | +0.06059 +/- 0.05199 | 0.13961 +/- 0.00259 | 2.697 +/- 0.347 |
| Frozen readout | B | Permute | 0.075 +/- 0.130 | +0.00045 +/- 0.00040 | 0.00484 +/- 0.00055 | 0.1054 +/- 0.0195 |

Across the 12 shallow models, permutation changed family logits for every one of
the 5,316 model-example pairs. It changed one predicted family, from one wrong
class to another. It corrected no family errors and introduced no family errors.
The per-model 95th percentile of absolute family-margin change was
`0.00927-0.02449`, whereas each model's median absolute intact margin was
`7.36-15.93`. These are small continuous changes relative to the existing margins;
unchanged accuracy does not mean unchanged output.

Zeroing produced 28 family-label flips: nine wrong-to-correct, four
correct-to-wrong, and 15 wrong-to-a-different-wrong-class. The corrections/errors
by condition were end-to-end A `2/0`, end-to-end B `6/4`, frozen A `0/0`, and frozen
B `1/0`. Zeroing therefore has a larger effect than permutation, although neither
intervention yields a stable recovery of combination generalization. The
per-model 95th percentile of absolute margin change after zeroing was
`0.18584-0.92017`; all 12 intact mean correct-class margins were negative.

The 5,316 denominator repeats molecular identities across models and training
experiences. These pooled counts are descriptive totals, not 5,316 independent
experimental replicates. The three-seed panels above preserve the experiment's
replication unit.

## Task and control checks

On the same common-OOD examples, permutation caused family/position/distance
label-flip totals of `1/0/3`; zeroing caused `28/0/74`. The three distance flips
under permutation comprised one correction and two wrong-to-wrong changes. Under
zeroing, distance had 27 corrections, 35 newly wrong predictions, and 12
wrong-to-wrong changes. Position logits changed continuously despite no position
label changes. Every common-OOD logit vector changed under both interventions in
all three tasks. The saved row-level outputs retain the full distribution rather
than only the family accuracy result.

The matched duplicate-deep-mean control also responds to branch interventions:
its common-OOD family permutation caused 39 flips, with per-model mean relative
centered-logit distances of `3.44%-5.22%`. For shallow-skip permutation, that range
was `0.034%-0.169%`. The diagnostic is therefore capable of detecting larger
output effects in the same head architecture. This comparison does not identify
whether small shallow effects arise from limited variation or task-relevant
content in the vector, weak head sensitivity to that variation, or both.

The defensible conclusion is that the appended shallow route is computationally
active, while these molecule-specific substitutions usually perturb confident
wrong predictions too little to change their labels. Zeroing is a stronger,
potentially off-distribution intervention and cannot isolate a semantic causal
component. Neither result proves universal branch nonuse or establishes the
training mechanism behind the small v5 advantage.

## Preservation

This review changed no source files, checkpoints, or frozen experiment records.
All 26 executed-source hashes and all 54 branch input hashes still match their
recorded values. The primary independent audit verified the 724-file v6 manifest
and all 536 files across the seven earlier frozen records. This review is stored
separately from those records so their manifests remain unchanged.
