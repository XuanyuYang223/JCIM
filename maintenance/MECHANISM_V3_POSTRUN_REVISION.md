# Mechanism v3 post-run source revision

This revision was made only after the readout/replay, local diagnostic, and correlation-intervention runs and their independent audits completed. All three v3 records executed source commit `499194ec5d43c93e004d9ccc9a168df42b2fbd18`. Their recorded environments, source hashes, data, results, checkpoints, generated reports, manifests, and completed audit JSON files remain unchanged.

## Narrow implementation change

`local_records` in [mechanism_diagnostics.py](../jcim/mechanism_diagnostics.py) now materializes the archived `x` and `adj` arrays once before iterating through molecules. Previously, indexing an `NpzFile` member inside each iteration repeatedly decompressed the complete member. The later revision removes that repeated work. Target atom selection, feature values, query markers, graph edges, labels, order, dtype, and training protocol are unchanged.

| Version of `jcim/mechanism_diagnostics.py` | File SHA-256 |
| --- | --- |
| Executed v3 source at `499194e` | `ee4dda1045130517c60c649d14a0eac2c8bf861bdf6bd1968bbb4700808cd93a` |
| Subsequent source with member materialization | `16a62586d91628db6902392642e99793b3b9474387e0a270b190d0ba66848d1d` |

The old SHA-256 was also checked directly against the raw bytes returned by `git show 499194ec5d43c93e004d9ccc9a168df42b2fbd18:jcim/mechanism_diagnostics.py`. Future audits of the executed experiment must validate its recorded historical source checkout or Git blob when the current working file differs. A historical source match establishes the bytes associated with that run; it does not claim that the revised working file was executed in the frozen experiment.

## Verification without rerunning training

The ten focused tests in [test_mechanism_diagnostics.py](../tests/test_mechanism_diagnostics.py) pass after the revision. A counting wrapper around the actual hard-v2 NPZ archive confirmed exactly one `x` member read and one `adj` member read while reconstructing all 6,912 local inputs.

All reconstructed local feature and adjacency arrays were compared against the completed [local_graphs.npz](../experiments/mechanism_diagnostics_v3/local_graphs.npz). Molecular ID order, shapes, dtypes, numerical equality, and raw tensor bytes match. The hashes below refer to C-order array bytes, not to the compressed NPZ file.

| Array | Shape | Dtype | Raw array SHA-256 | Comparison |
| --- | --- | --- | --- | --- |
| `x` | 6912 x 9 x 18 | float32 | `df9958d660660e5271a956ba00b63a41aa718d463df44d9bf3467641e158f2c1` | Bitwise identical |
| `adj` | 6912 x 9 x 9 | float32 | `dd36416ec403789a3b6a4df56fe4bb3346bb6e89bed5eb57e6831550fdafb5fe` | Bitwise identical |

Reconstruction took approximately 0.26 seconds in the local verification process; this is an illustrative observation rather than a controlled runtime benchmark. No model was trained, no archived array was rewritten, and no scientific result was recomputed for this source revision.

All manifest-listed files were checked again after the change, with zero checksum mismatches:

| Frozen record | Verified files | Manifest SHA-256 |
| --- | ---: | --- |
| [hard-v2](../experiments/hard_v2/MANIFEST.json) | 131 | `4b69cd0335cd21ccb5e2a419c1301bfc8042e3bb5e9864322d6ad4ca1f398996` |
| [readout/replay v3](../experiments/readout_replay_v3/MANIFEST.json) | 106 | `e4f6a3a4abfa0a41f4f13a2ae4573ff7008287d6bc84a1306b97f671f5f37d26` |
| [local diagnostics v3](../experiments/mechanism_diagnostics_v3/MANIFEST.json) | 22 | `2bea772ce613fcf2fe79f66326847de5cbb8e01ec62aa727a7fec4740fcad76a` |
| [correlation intervention v3](../experiments/correlation_v3/MANIFEST.json) | 77 | `4d0323dac89048dcc8d5f1c788e78e513412ccd8cb1b5acffd3c5d0c765e1e01` |

This note records the later implementation revision separately so the executed source and frozen scientific record remain traceable.

## Independent auditor revision

The [mechanism maintenance auditor](audit_mechanism_v3.py) now accepts an executed source-file hash only when it matches the current file bytes or the exact Git blob at the full immutable commit recorded at experiment start. It rejects branch names, abbreviated or nonexistent commits, wrong historical bytes, and hash mismatches. Before this revision, all 16 recorded source hashes were independently confirmed against commit `499194ec5d43c93e004d9ccc9a168df42b2fbd18`.

The original [mechanism audit](READOUT_REPLAY_V3_AUDIT.json) remains unchanged. The separate [post-run mechanism audit](READOUT_REPLAY_V3_POSTRUN_AUDIT.json) passes, verifying 15 executed files against current bytes and `jcim/mechanism_diagnostics.py` against its exact recorded-commit blob. It records the provenance used for each file and auditor SHA-256 `2f5bff2d0a17244ba9b659e95194542bbb305d0bcca17334fe27be841ff0cb02`.

The ten focused [auditor tests](../tests/test_mechanism_audit.py), including three revision-validation subtests, pass. Positive and negative controls cover working-tree edits with valid historical recovery, wrong hashes, different historical snapshots, abbreviated revisions, and nonexistent revisions. This later maintenance revision changes provenance verification and does not revise any experimental observation.

Final repository validation after the source and auditor revisions: `python -m pytest -q` passed all 118 tests and 34 subtests.
