# Snakebot v2 checkpoints

These curated checkpoints belong to task `Mjlab-Locomotion-Flat-Snakebot-v2`
and experiment `snakebot_locomotion_v2`.

| File | Iteration | Recorded training mean reward | SHA-256 |
| --- | ---: | ---: | --- |
| `model_800.pt` | 800 | 328.034119 | `f9974938f20f1aabef7209dfa16895f55a2cb0a1a9c0788fade30a98c32a1457` |
| `model_1200.pt` | 1200 | 312.155579 | `2eb4bcd26c962dde0db1c281481974752ab36f9a2bdb06533e21ca355e945f93` |

The rewards are the `Train/mean_reward` TensorBoard scalars at each checkpoint
iteration; they are not deterministic evaluation scores.

Both files come from source run `2026-03-10_19-34-58`. They are regular Git
blobs of approximately 10.9 MiB each, so Git LFS is not required.

`model_1200.pt` is the checkpoint used for the committed goal-reaching
evaluation and recordings. See
`docs/evaluations/snakebot_v2_model_1200/README.md`.

Load the evaluated checkpoint from the mjlab directory:

```bash
uv run play Mjlab-Locomotion-Flat-Snakebot-v2 --checkpoint-file checkpoints/snakebot_v2/model_1200.pt
```
