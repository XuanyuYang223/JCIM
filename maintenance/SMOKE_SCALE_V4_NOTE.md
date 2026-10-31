# Scale-v4 pipeline validation and numerical correction

The first local prototype, experiments/smoke_scale_v4, used one encoder seed,
one support draw, and depths 0 and 8. It is excluded from primary scientific
evidence and retained locally without overwriting its record.

The independent prototype audit detected two distinct development issues:

- The report generator was edited after the prototype had captured its source
  hashes. The completed prototype correctly failed an executed-source
  current-file check; it was not relabeled as a clean frozen-source experiment.
- The final 65 graphs of its 1,473-graph feature set used a different CUDA matrix
  shape from the preceding batches of 128. Identical raw attachment-atom
  features then produced two layer-0 vectors, differing by at most about
  3e-8 for A and 6e-8 for B/random. Standardizing near-constant coordinates could
  amplify this numerical fingerprint, and exact ridge refits disagreed on two
  degenerate-layer prediction vectors.

The extraction code now repeats valid graphs within the final batch until the
configured batch size is reached, and discards duplicate outputs afterward.
The original encoder has no operation coupling different graphs. Its per-node
LayerNorm, full graph inputs, query markers, message layers, root operation,
and legacy graph pooling formula remain unchanged. Fixed batch shapes remove
the numerical fingerprint without supplying any extra input information.
Other 64-dimensional means accumulate in float64 before returning float32 to
prevent atom-order summation fingerprints.

The same 1,473-graph prototype selection was checked on all three seed-0
checkpoints after the correction: both root and target layer-0 feature arrays
then have exactly one unique vector and zero coordinate variation. A separate
test compares extraction bit for bit with the original model evaluated on the
same repeated full-graph batch. The primary runner also rejects any nonconstant
root or target layer-0 control and archives these checks for every model.

A fresh final pipeline check uses a new output directory after all executed
source modules are fixed. The primary three-seed, three-support-draw, all-depth
experiment runs only after that pipeline and its independent audit pass.
No result from a completed prior scientific experiment was modified.

The [initial failed audit](SMOKE_SCALE_V4_AUDIT.json) and its
[exact executed auditor source](archive/audit_scale_v4_smoke_initial.py) remain
available. The [fresh final pipeline audit](SMOKE_SCALE_V4_FINAL_AUDIT.json)
passes all 24 file checks, three frozen encoders, 1,473 selected identities,
336 adaptation rows, 1,008 direct-transfer rows, and exact independent
reproduction of all 224 ridge prediction vectors. No numerical tolerance was
introduced for prediction equality.
