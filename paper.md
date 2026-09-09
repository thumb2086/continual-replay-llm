# Replay-First Continual Learning for Small Streaming Language Models

## Draft: Technical Report

### Abstract

We study continual learning in a tiny character-level Transformer trained on real podcast transcripts. Using the same starting checkpoint, we compare pure fine-tuning, Elastic Weight Consolidation, explicit old-topic replay, and EWC plus replay. Pure fine-tuning learns the new topic fastest but forgets the old topic. EWC slows forgetting but also slows new learning. Explicit old-topic replay gives the best tradeoff in our runs: it preserves old-topic loss while retaining most of the new-topic learning speed. An EWC lambda sweep shows increasing lambda mainly damps plasticity without improving retention once replay is fixed. A replay-size sweep shows even a small replay batch preserves the old topic, while larger replay improves both retention and new-topic learning. Finally, a held-out A/B/A/B long-cycle run passes all retention gates and improves both topics.

This is a small-scale technical report, not a claim of general continual-learning superiority.

### 1. Introduction

Neural networks trained sequentially often lose old capabilities when learning new data. This is catastrophic forgetting. The core tension is stability versus plasticity: protect old parameters or adapt quickly to new data.

We focus on a practical setting:

```text
A small language model learns Topic A, then Topic B, then Topic A again.
It must not forget Topic A while learning Topic B.
```

Our main empirical finding is simple:

```text
In this podcast-transcript setup, explicit old-topic replay is the main retention mechanism.
EWC primarily controls how fast the model changes, not whether old knowledge survives.
```

### 2. Related Work

Catastrophic forgetting has been studied since McCloskey and Cohen (1989) and reviewed by French (1999). Elastic Weight Consolidation (Kirkpatrick et al., 2017) protects parameters that are important for previous tasks by using Fisher information. Experience replay stores or revisits old examples; Rolnick et al. (2019) showed replay can substantially reduce forgetting in continual reinforcement learning.

Our contribution is not a new algorithm. It is a controlled small-scale comparison for streaming character-level language modeling:

1. Same-checkpoint online continual-learning protocol.
2. Four-condition ablation: pure, EWC-only, replay-only, EWC+replay.
3. EWC lambda and replay-size sweeps.
4. Held-out A/B/A/B long-cycle validation.

### 3. Method

#### 3.1 Model

We use a small Transformer encoder with a retrieval memory bank.

```text
Parameters: about 1.36M
Vocabulary: character-level, about 3.2k
Embedding dimension: 128
Hidden dimension: 256
Layers: 4
Heads: 4
Device: NVIDIA GeForce RTX 3060 Ti
```

A Hebbian layer and memory bank were implemented as part of the broader CortexFlow architecture, but the decisive retention mechanism in these runs is explicit old-topic replay.

#### 3.2 Data

We use real Chinese podcast transcripts.

```text
Total texts: 50
Topic A texts: 25
Topic B texts: 25
Topic definition: first half vs second half of immediate podcast subdirectories
Chunk length: 128 characters
Training chunks: about 3,850 total
Evaluation: held-out chunks reserved before training
```

Topics are directory proxies rather than manually annotated semantic domains. This is a limitation.

#### 3.3 Training protocol

Phase 0:

```text
Backpropagation pretraining on Topic A.
```

Online phases:

```text
Learn the new topic while replaying old-topic chunks.
```

Loss:

```text
total_loss = task_loss + EWC_loss + 0.5 * replay_loss
```

Retention gate:

```text
old-topic loss after phase <= 1.05 * old-topic loss before phase
```

Learning gate:

```text
new-topic loss improves by a phase-specific threshold
```

The long-cycle run reserves 200 chunks per topic for evaluation and trains only on the remaining chunks.

### 4. Results

#### 4.1 Baseline: pure fine-tuning forgets

From the same Topic A checkpoint:

```text
Shared start:
  A = 3.7817
  B = 5.0867

Pure fine-tuning on Topic B:
  A: 3.7817 -> 4.2390
  B: 5.0867 -> 3.6484
```

Pure fine-tuning improves Topic B by 28.27% but worsens Topic A by a factor of 1.121.

#### 4.2 Four-condition ablation

All four models start from the same checkpoint.

```text
Pure:
  A 1.102x worse, B +23.36%

EWC-only:
  A 1.015x worse, B +11.64%

Replay-only:
  A 0.981x, B +21.96%

EWC+replay:
  A 0.993x, B +11.10%
```

Replay-only gives the best tradeoff. EWC preserves Topic A better than pure fine-tuning but substantially slows Topic B learning.

#### 4.3 EWC lambda sweep

With replay fixed at two old chunks per new step:

```text
Lambda 0:     A 0.984x, B +21.92%
Lambda 100:   A 0.982x, B +21.20%
Lambda 500:   A 0.983x, B +19.03%
Lambda 2000:  A 0.987x, B +14.67%
Lambda 5000:  A 0.991x, B +11.13%
Lambda 20000: A 0.994x, B +6.92%
```

Once replay is present, increasing lambda reduces new-topic learning without materially improving retention.

#### 4.4 Replay-size sweep

With EWC off:

```text
Replay 0: A 1.103x worse, B +23.35%
Replay 1: A 0.992x, B +21.39%
Replay 2: A 0.984x, B +21.92%
Replay 4: A 0.978x, B +22.27%
Replay 8: A 0.973x, B +22.59%
```

Even replay batch 1 prevents forgetting. Replay batch 8 gives the strongest retention with almost no loss of new-topic learning speed.

#### 4.5 Held-out A/B/A/B long cycle

Best practical setting:

```text
replay batch: 8
EWC lambda: 0
online epochs per phase: 8
online chunks per phase: 300
held-out evaluation chunks: 200 per topic
```

Results:

```text
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

Final held-out state:

```text
Topic A: 4.1977 -> 4.0021
Topic B: 5.2536 -> 4.3102
```

All retention gates pass. Topic A never gets materially worse after learning Topic B, and both topics improve across the full cycle.

### 5. Discussion

The cleanest conclusion is:

```text
Replay preserves memory.
EWC mainly throttles change.
For this streaming setup, replay dominates.
```

This does not mean EWC is useless in general. In settings without stored old data, or with privacy constraints, EWC and related regularization methods remain important. But when old-topic replay is available, our runs suggest spending budget on replay is more effective than increasing EWC strength.

### 6. Limitations

1. Topics are directory proxies, not clean semantic tasks.
2. The model has only about 1.36M parameters.
3. Tokenization is character-level.
4. Podcast transcripts are noisy.
5. Only 50 files and short training were used.
6. Evaluation is loss-based, not task-based.
7. No public benchmark comparison yet.
8. Results do not establish general continual-learning superiority.

### 7. Reproducibility

Core files:

```text
online-llm/train_v3.py
online-llm/compare_baseline.py
online-llm/compare_ablation.py
online-llm/sweep_lambda.py
online-llm/sweep_replay.py
online-llm/run_research_loop.py
```

Result files:

```text
online-llm/data/baseline_comparison.json
online-llm/data/ablation_results.json
online-llm/data/lambda_sweep.json
online-llm/data/replay_sweep.json
online-llm/data/research_state.json
```

Shared pretraining uses seed 123. Replay sweep and ablations use the same starting-checkpoint design to isolate the online-learning method.

### 8. Future Work

1. Use clean, semantically labeled public datasets.
2. Add task-level benchmarks rather than only loss.
3. Test longer A/B/A/C cycles.
4. Sweep replay size, Fisher sample count, and EWC schedules jointly.
5. Increase model size within single-GPU limits.
6. Compare against standard continual-learning baselines.
