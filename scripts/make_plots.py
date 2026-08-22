import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# ============================================================
# PLOT 1: VOCAB COMPRESSION CURVE (Q3)
def plot_vocab_compression():
    df = pd.read_csv("logs/vocab_study_results.csv")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["vocab_size"], df["bytes_per_token"], 'o-', linewidth=2, markersize=10, color='#1f77b4')
    ax.axvline(x=4000, color='r', linestyle='--', alpha=0.7, label='Chosen V=4000')
    
    ax.set_xlabel("Vocabulary Size", fontsize=12)
    ax.set_ylabel("Bytes per Token", fontsize=12)
    ax.set_title("BPE Vocabulary Size vs. Compression Efficiency", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xticks([1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(['1k', '2k', '4k', '8k', '16k'])
    
    plt.tight_layout()
    plt.savefig("logs/vocab_compression_curve.png", dpi=300)
    plt.close()
    print("✅ Saved: logs/vocab_compression_curve.png")

# ============================================================
# PLOT 2: LR SWEEP RESULTS (Q10)

def plot_lr_sweep():
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    losses = [3.5751, 2.9056, 2.4088, 2.2308, 2.3595]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(lrs, losses, 'o-', linewidth=2, markersize=10, color='#1f77b4')
    ax.axvline(x=3e-3, color='r', linestyle='--', alpha=0.7, label='Best LR = 3e-3')
    
    ax.text(5e-3, 2.5, 'No divergence observed\nin tested range', 
            fontsize=9, style='italic', ha='center')
    
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Learning Rate Sweep Results", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("logs/lr_sweep_plot.png", dpi=300)
    plt.close()

# ============================================================
# PLOT 3: ABLATION RESULTS (Q11/Q12)
def plot_ablations():
    models = ["Baseline (V=4000)", "No RoPE"]
    losses = [2.2308, 2.5710]
    colors = ['#1f77b4', '#ff7f0e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, losses, color=colors)
    
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.text(0.5, 2.0, 'RMSNorm: Diverged\n(not shown as comparable)', 
            ha='center', fontsize=10, style='italic', color='#d62728',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffeeee', alpha=0.8))
    
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Ablation Study Results", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 3.0)
    
    plt.tight_layout()
    plt.savefig("logs/ablations_plot.png", dpi=300)
    plt.close()

# ============================================================
# RUN ALL PLOTS
if __name__ == "__main__":
    print("Generating plots...")
    print("=" * 50)
    plot_vocab_compression()
    plot_lr_sweep()
    plot_ablations()
    print("=" * 50)
    print("All plots generated in logs/")
    
    png_files = glob.glob("logs/*.png")
    print("\nFiles created:")
    for f in png_files:
        size = os.path.getsize(f) / 1024
        print(f"  {f} ({size:.1f} KB)")