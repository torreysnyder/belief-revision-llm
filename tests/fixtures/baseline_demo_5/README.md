# Baseline demo run

This fixture preserves the first completed API-evaluator experiment output used
to validate the repository refactor. It is a behavioral regression reference,
not scientific ground truth and not a claim that model wording is reproducible.

The attempted demo requested five replicates. Four completed successfully
(replicate indices `0`, `2`, `3`, and `4`); replicate `1` is absent. Every
completed conversation contains 15 paired user/assistant turns and four
evaluator checkpoints.

- `conversations.jsonl` contains one complete serialized conversation per line.
- `results.csv` contains one turn-level row per user/assistant exchange.
- `manifest.json` records immutable hashes and structural expectations.

Regression tests intentionally check schemas, ordering, counts, phase flow,
probe placement, and agreement between the two formats. They do not compare
newly generated natural-language text with this fixture.
