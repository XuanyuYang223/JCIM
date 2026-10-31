# Smoke diagnostic provenance

The two-epoch, one-seed run in `experiments/smoke_v1` checked the training and reporting pipeline. It is excluded from Git and is not the primary scientific result.

Some source revisions were made while that diagnostic process was importing dependencies. Its recorded source hashes therefore describe files present when metadata was written, and are not a complete executable-source snapshot for that diagnostic. In particular, the additional local cross-family position probe and atom-map reference metadata were finalized after the diagnostic was launched.

The primary `pilot_v1` experiment starts only after source revisions and verification are complete. Its source commit and source hashes are recorded before training. The diagnostic's numerical records and original manifest are preserved separately without alteration.
