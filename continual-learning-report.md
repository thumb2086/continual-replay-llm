# Continual Learning without Catastrophic Forgetting

## Online LLM V3: Backprop Pretraining + EWC + Old-Topic Replay

## Abstract

This report documents a continual-learning experiment using real podcast transcripts. A small Transformer-based online learner first receives backpropagation pretraining on Topic A. It then learns Topic B online with Elastic Weight Consolidation and old-topic replay. The model preserves Topic A while learning Topic B, and further improves Topic A when it returns to it.

Final result: **no catastrophic forgetting in this test**.

## Problem

Standard neural networks forget old tasks when trained on new data. This is called catastrophic forgetting.

Research goal:

```text
Learn Topic B without losing Topic A.
```

## Method

The model combines three mechanisms:

1. Transformer backbone
   - Character-level Transformer encoder.
   - Used for representation and prediction.

2. Elastic Weight Consolidation
   - Fisher information estimates parameter importance after the first training stage.
   - Penalizes changes to parameters important for previous knowledge.

3. Old-topic replay and memory retrieval
   - During online learning, each new-data step also reviews a small replay batch from the old topic.
   - A fixed-capacity memory bank stores important embeddings for retrieval-augmented forward passes.

The online-learning loss is:

```text
total_loss = task_loss + EWC_loss + 0.5 * replay_loss
```

EWC configuration used in this run:

```text
online optimizer: Adam, lr = 1e-4
EWC lambda: 5000
replay batch: 2 old-topic chunks per online step
memory capacity: 50000 slots
```

## Data

Real Chinese podcast transcripts were used:

```text
Total texts: 50
Topic A texts: 25
Topic B texts: 25
Topic definition: first half vs second half of immediate podcast subdirectories
Tokenizer: character-level
Vocabulary: 3239 characters
Training chunks: 3850
Topic A chunks: 1925
Topic B chunks: 1925
Chunk length: 128 characters
```

The Topic A/B labels are directory-group proxies, not manually annotated semantic topics. This is a limitation discussed below.

## Model

```text
Parameters: 1,362,343
Embedding dimension: 128
Hidden dimension: 256
Transformer layers: 4
Attention heads: 4
Device: NVIDIA GeForce RTX 3060 Ti
```

The model is intentionally small. The goal is to test continual-learning behavior, not to challenge large production LLMs.

## Training Protocol

```text
Phase 1: backpropagation pretraining on Topic A
Phase 2: compute Fisher information from Topic A
Phase 3: online learning on Topic B with EWC + old-topic replay
Phase 4: compute Fisher information from Topic B
Phase 5: online learning on Topic A again with EWC + replay
```

Evaluation metric is average cross-entropy loss over 50 held-sample chunks per topic.

Retention criterion:

```text
Topic A is preserved if loss_A_after_B <= loss_A_before_B * 1.15
```

Learning criterion:

```text
Topic B is learned if loss_B_after < loss_B_before * 0.90
```

## Results

### After pretraining on Topic A

```text
Topic A loss: 3.9473
Topic B loss: 5.2451
```

### After online learning on Topic B

```text
Topic A loss: 3.9185
Topic B loss: 4.6102
```

### After returning to Topic A

```text
Topic A loss: 3.7264
Topic B loss: 4.5622
```

### Verdict

```text
Topic A preserved: YES
  3.9473 -> 3.9185

Topic B learned: YES
  5.2451 -> 4.6102

Topic A round-trip: YES
  3.9473 -> 3.7264
```

The strict test conclusion is:

```text
SUCCESS: No catastrophic forgetting
```

Additional state:

```text
Memory entries used: 6500
Total parameters: 1362343
```

## Baseline Comparison: Pure Fine-tuning vs EWC + Replay

To check whether EWC and replay were responsible for retention, two models were cloned from the same checkpoint after Topic A pretraining.

```text
Shared start: A = 3.7817, B = 5.0867

Baseline, pure fine-tuning on Topic B:
  A: 3.7817 -> 4.2390
  B: 5.0867 -> 3.6484

EWC + old-topic replay on Topic B:
  A: 3.7817 -> 3.7494
  B: 5.0867 -> 4.4798
```

Result:

```text
Baseline B improvement: 28.27%
EWC B improvement:      11.93%

Baseline A change: 1.121x worse
EWC A change:      0.991x, preserved
```

Interpretation:

1. Pure fine-tuning learns Topic B faster but forgets Topic A.
2. EWC + replay learns Topic B more slowly but preserves Topic A.
3. This is a stability–plasticity tradeoff.

The baseline experiment makes the V3 result stronger: without EWC and replay, the same starting model loses old-topic performance when learning the new topic.

## Ablation: Pure vs EWC-only vs Replay-only vs EWC+Replay

A second experiment cloned four models from the same Topic A checkpoint and trained all of them on Topic B.

```text
Shared start: A = 3.7817, B = 5.0867

Pure fine-tuning:
  A: 3.7817 -> 4.1688
  B: 5.0867 -> 3.8986

EWC-only:
  A: 3.7817 -> 3.8370
  B: 5.0867 -> 4.4948

Replay-only:
  A: 3.7817 -> 3.7086
  B: 5.0867 -> 3.9695

EWC + replay:
  A: 3.7817 -> 3.7538
  B: 5.0867 -> 4.5219
```

Summary metrics:

```text
Pure:
  A change: 1.102x worse
  B gain: 23.36%

EWC-only:
  A change: 1.015x worse
  B gain: 11.64%

Replay-only:
  A change: 0.981x, preserved
  B gain: 21.96%

EWC + replay:
  A change: 0.993x, preserved
  B gain: 11.10%
```

Protocol for this ablation:

```text
Online epochs: 8
Online chunks per epoch: 300
EWC lambda: 5000
Replay batch: 2 old-topic chunks per new-topic step
Evaluation: 50 chunks per topic
Memory entries: pure 0, EWC-only 0, replay-only 2400, EWC+replay 2400
```

Interpretation:

1. Pure fine-tuning learns Topic B best but loses Topic A.
2. EWC reduces forgetting but also slows new learning.
3. Replay-only gives the best tradeoff in this run:
   - old-topic loss slightly improves
   - new-topic improvement is close to pure fine-tuning
4. Adding EWC on top of replay does not improve this run:
   - retention is already handled by replay
   - EWC damping reduces new-topic learning

So the strongest working component in this test is explicit old-topic replay, not EWC alone.

## EWC Lambda Sweep with Fixed Replay

A lambda sweep was run with old-topic replay fixed at 2 chunks per new-topic step.

```text
Shared start: A = 3.7817, B = 5.0867

Lambda 0:
  A: 3.7817 -> 3.7227
  B: 5.0867 -> 3.9718

Lambda 100:
  A: 3.7817 -> 3.7140
  B: 5.0867 -> 4.0082

Lambda 500:
  A: 3.7817 -> 3.7159
  B: 5.0867 -> 4.1185

Lambda 2000:
  A: 3.7817 -> 3.7314
  B: 5.0867 -> 4.3407

Lambda 5000:
  A: 3.7817 -> 3.7457
  B: 5.0867 -> 4.5205

Lambda 20000:
  A: 3.7817 -> 3.7600
  B: 5.0867 -> 4.7348
```

Summary:

```text
Lambda 0:     A 0.984x, B +21.92%
Lambda 100:   A 0.982x, B +21.20%
Lambda 500:   A 0.983x, B +19.03%
Lambda 2000:  A 0.987x, B +14.67%
Lambda 5000:  A 0.991x, B +11.13%
Lambda 20000: A 0.994x, B +6.92%
```

Protocol:

```text
Online epochs: 8
Online chunks per epoch: 300
Replay batch: 2 old-topic chunks per new-topic step
Evaluation: 50 chunks per topic
Memory entries: 2400 for every lambda
```

Interpretation:

1. Topic A remains preserved across all lambda values because replay is fixed.
2. Larger lambda monotonically reduces Topic B learning speed.
3. Lambda 0–100 keeps almost all replay-only learning speed.
4. Therefore, in this setup, EWC is mainly a damping control on plasticity; once replay protects retention, EWC is not the source of the main benefit.

## Replay-Size Sweep with EWC Off

A separate sweep varied old-topic replay while turning EWC off.

```text
Shared start: A = 3.7817, B = 5.0867

Replay 0:
  A: 3.7817 -> 4.1701
  B: 5.0867 -> 3.8989

Replay 1:
  A: 3.7817 -> 3.7510
  B: 5.0867 -> 3.9988

Replay 2:
  A: 3.7817 -> 3.7227
  B: 5.0867 -> 3.9718

Replay 4:
  A: 3.7817 -> 3.7000
  B: 5.0867 -> 3.9540

Replay 8:
  A: 3.7817 -> 3.6800
  B: 5.0867 -> 3.9376
```

Summary:

```text
Replay 0: A 1.103x worse, B +23.35%
Replay 1: A 0.992x, B +21.39%
Replay 2: A 0.984x, B +21.92%
Replay 4: A 0.978x, B +22.27%
Replay 8: A 0.973x, B +22.59%
```

Protocol:

```text
Online epochs: 8
Online chunks per epoch: 300
EWC: off
Replay batch: 0, 1, 2, 4, or 8 old-topic chunks per new-topic step
Evaluation: 50 chunks per topic
Memory entries: 0 for replay 0, 2400 for replay 1–8
```

Interpretation:

1. Without replay, Topic A is lost.
2. Even replay batch 1 preserves Topic A.
3. Larger replay slightly improves both retention and new-topic learning in this run.
4. The best practical setting here is replay 8:
   - strongest Topic A retention
   - Topic B learning nearly as strong as pure fine-tuning.

## Long-Cycle Held-Out Validation: A -> B -> A -> B

The earlier V3 run evaluated on training-distribution chunks. To remove train/eval leakage, a long-cycle run reserved 200 chunks per topic before training and evaluated only on held-out chunks.

```text
Config:
  pretrain epochs: 5
  online epochs per phase: 8
  online chunks per phase: 300
  replay batch: 8 old-topic chunks per new-topic step
  EWC lambda: 0
  retain limit: old-topic loss <= 1.05x old-topic loss before the phase
  B1 learning gate: >= 10% improvement
  later-cycle gate: no material worsening

After pretrain:
  A = 4.1977
  B = 5.2536

Phase B1:
  B: 5.2536 -> 4.5846
  A: 4.1977 -> 4.1111

Phase A2:
  A: 4.1111 -> 4.1054
  B: 4.5846 -> 4.3827

Phase B2:
  B: 4.3827 -> 4.3102
  A: 4.1054 -> 4.0021
```

Interpretation:

1. Held-out B1 shows real learning, not just training-chunk memorization:
   - Topic B improves by about 12.7%.
2. Held-out A is retained across every phase:
   - after B1: 0.979x
   - after A2, B is 0.956x
   - after B2, A is 0.975x
3. Later cycles do not need large new-topic gains because the topic is already partly learned; the correct gate is retention.
4. Final state:
   - Topic A improves from 4.1977 to 4.0021 across the full cycle.
   - Topic B improves from 5.2536 to 4.3102 across the full cycle.

This long-cycle result is the paper-grade evidence.

## Interpretation

1. Old knowledge was not displaced.
   - Topic A loss improved slightly after learning Topic B.
   - This is consistent with EWC and replay protecting important parameters.

2. New knowledge was actually learned.
   - Topic B loss decreased by about 12%.
   - This exceeded the predefined 10% learning threshold.

3. Returning to Topic A further improved it.
   - This suggests that old-topic replay and later retraining can consolidate prior knowledge.

## Limitations

1. Topics are directory proxies, not manually labeled domains.
2. The model is tiny, with 1.36M parameters.
3. The tokenizer is character-level.
4. The dataset is noisy podcast transcript text.
5. Only 50 files and relatively short training were used.
6. The result does not prove general continual-learning ability.
7. Larger-scale replication is still needed.

## Next Work

1. Use clean, semantically labeled datasets.
2. Test even longer A/B/A/B/C cycles.
3. Measure forward and backward transfer more comprehensively.
4. Compare against standard continual-learning baselines on public benchmarks.
5. Increase model size while staying within 3060 Ti limits.
6. Publish code, configuration, and full logs.

## Files

```text
online-llm/train_v3.py
online-llm/compare_baseline.py
online-llm/compare_ablation.py
online-llm/sweep_lambda.py
online-llm/sweep_replay.py
online-llm/run_research_loop.py
online-llm/paper.md
online-llm/data/v3_results.json
online-llm/data/v3_model.pt
online-llm/data/baseline_comparison.json
online-llm/data/ablation_results.json
online-llm/data/lambda_sweep.json
online-llm/data/replay_sweep.json
online-llm/data/research_state.json
online-llm/data/research_loop_model.pt
```
