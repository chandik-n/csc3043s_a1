# Experiment Log — CSC3043S Assignment 1

## System Info
- Python: 3.13.1
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA Tesla T4 (Colab)
- OS: Ubuntu 22.04 (Colab)

---

## 1. Tokenizer (§3)

### Vocabulary Study (§3.4)
| Vocab Size | Bytes/Token | Token Count (Val) | Train Time (s) |
|------------|-------------|-------------------|----------------|
| 1,000 | 3.17 | 7,096,654 | 231.96 |
| 2,000 | 3.68 | 6,113,802 | 233.11 |
| 4,000 | 3.98 | 5,652,600 | 239.91 |
| 8,000 | 4.11 | 5,479,500 | 253.42 |
| 16,000 | 4.13 | 5,453,404 | 280.64 |

**Source**: `logs/vocab_study_results.csv`

**Choice**: V=4000 chosen due to diminishing returns beyond this point (3.98→4.11 is only 3.3% improvement for 2× vocab size) and parameter budget (V=4000 uses 4.1M embedding+head params vs 16.4M for V=16000).

### Optimised vs Tutorial Speedup (Q2)
| Implementation | Time (10 MB) | Speedup |
|----------------|--------------|---------|
| Tutorial (naive recount) | 348.58 s | 1× |
| Optimised (incremental) | 219.99 s | 1.6× |

**Source**: Measured on the same 10 MB slice of training data, same machine.

---

## 2. Model (§4)

### Parameter Count (Q5)
| Component | Parameters |
|-----------|------------|
| Token Embedding | 2,048,000 |
| LM Head | 2,048,000 |
| Embedding + Head | 4,096,000 |
| Non-embedding | 12,456,960 |
| **Total** | **16,552,960** |

### KV Cache Verification (Q6)
- Max logit difference (cached vs uncached): **1.64×10⁻⁶**
- Greedy generations: bitwise identical over 100 tokens

---

## 3. Training (§5)

### Step Time (Q8)
| Precision | Step Time | Source |
|-----------|-----------|--------|
| bf16 (autocast) | ~375 ms | Measured on Colab T4 (8000 steps / 50 min) |
| fp32 | ~675 ms | Estimated (bf16 typically 1.8× faster on T4) |

---

## 4. Experiments (§7)

### LR Sweep (Q10)
| LR | Val Loss |
|----|----------|
| 1e-4 | 3.5751 |
| 3e-4 | 2.9056 |
| 1e-3 | 2.4088 |
| **3e-3** | **2.2308** (BEST) |
| 1e-2 | 2.3595 |

### RMSNorm Ablation (Q11)
- Status: **Diverged** at LR=3e-3
- Gradient norm progression: 12.47 (step 50) → ∞ (step 70)
- Conclusion: RMSNorm is critical for training stability

### NoPE Ablation (Q12)
| Config | Val Loss | Gap |
|--------|----------|-----|
| Baseline (with RoPE) | 2.2308 | — |
| No RoPE | 2.5710 | 0.3402 |

**Conclusion**: RoPE provides meaningful positional information even with causal attention.

### Design Impact (Q14)
| Choice | Gap | Impact (Relative) |
|--------|-----|-------------------|
| LR (3e-3 vs 1e-3) | 0.1780 | — |
| RoPE vs NoPE | 0.3402 | — |
| RMSNorm | Diverged | Critical (not numerically comparable) |

*Numeric gaps are directly comparable between the two completed ablations; RMSNorm's failure mode is not a comparable numeric gap.*

### Vocabulary Comparison (Q15)
| Model | Loss | PPL | BPC | Total Params | Embedding + Head |
|-------|------|-----|-----|--------------|------------------|
| V=1000 | 1.8838 | 6.58 | 0.857 | 13,480,960 | 1,024,000 |
| V=4000 | 2.2308 | 9.31 | 0.809 | 16,552,960 | 4,096,000 |

*Note: Non-embedding backbone is identical at 12,456,960 parameters for both.*

---

## 5. Final Model (Q18)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Model | V=4000, LR=3e-3 |
| Steps | 20,000 (4× standard) |
| Training Time | ~2.1 hours (0.375s/step) |
| Batch Size | 32 |
| Context Length | 256 |

### Results on Proper Splits
| Dataset | Loss | PPL | BPC | Tokens |
|---------|------|-----|-----|--------|
| Validation Holdout | 1.5444 | 4.6851 | 0.5598 | 5,185,140 |
| **Test Set** (Touched Once) | **1.5618** | **4.7675** | **0.5661** | **404,244** |

**Data Split Note**: A dedicated test split of the last 2,000 validation documents was created after the LR sweep and ablation experiments had been run. Those experiments were evaluated on the full validation file (including what became the test documents). The final model results reported here use the properly split test set, touched exactly once.

---

## 6. Failed Runs and Process Evidence

| Run | Failure | Step | Recovery |
|-----|---------|------|----------|
| No RMSNorm | Gradient explosion | ~50 | None (diverged) |
| Final Model | Colab disconnect | ~9945 | Resumed from checkpoint_step_8000.pt |

---

## 7. Disclosures

1. **Late test split**: The test split (last 2,000 docs) was created after LR sweep/ablation runs. Those experiments used the full validation file.
2. **Lost logs**: Per-step training logs (JSONL) were lost when the Colab session disconnected.
3. **Reduced steps**: Sweeps and ablations run at 500 steps (vs 5000 standard) due to compute constraints.
4. **Colab disconnections**: Training interrupted multiple times; resume logic used to recover.
5. **Skipped ablations**: ReLU FFN not run due to GPU budget constraints.

---

## 8. Traceability

All results can be traced to:
- `logs/vocab_study_results.csv` — Vocabulary study
- `logs/*.txt` — Final results summaries
- `logs/*.png` — Figures
- `samples.txt`, `samples_v1000.txt` — Generated samples
- `data/vocab_4k.json`, `data/merges_4k.txt` — Tokenizer files
- `checkpoints/` — Model checkpoints (lost with Colab session)