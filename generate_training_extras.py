import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import Config

OUT_DIR = Path('outputs/enhanced_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path('outputs/logs')


def gen_throughput_plot():
    csv_path = LOGS_DIR / 'training_throughput.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df['batch_size'], df['throughput'], 'o-', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (batches/sec)')
    ax.set_title('Training Throughput Scaling')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = OUT_DIR / 'training_throughput.png'
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def gen_hyperparams_csv():
    cfg = Config()
    rows = [
        ('NUM_EPOCHS', cfg.NUM_EPOCHS),
        ('LEARNING_RATE', cfg.LEARNING_RATE),
        ('WEIGHT_DECAY', cfg.WEIGHT_DECAY),
        ('BATCH_SIZE', cfg.BATCH_SIZE),
        ('PATCH_SIZE', cfg.PATCH_SIZE),
        ('TRAINING_STATES', ';'.join(cfg.TRAINING_STATES)),
        ('PPO_EPOCHS', cfg.PPO_EPOCHS),
        ('PPO_CLIP', cfg.PPO_CLIP),
        ('RL_GAMMA', cfg.RL_GAMMA),
        ('GAE_LAMBDA', cfg.GAE_LAMBDA),
        ('RL_HIDDEN_DIM', cfg.RL_HIDDEN_DIM),
        ('IMAGE_FEATURE_DIM', cfg.IMAGE_FEATURE_DIM),
        ('ENTROPY_COEF', cfg.ENTROPY_COEF),
        ('VALUE_COEF', cfg.VALUE_COEF),
    ]
    df = pd.DataFrame(rows, columns=['Hyperparameter', 'Value'])
    csv_out = OUT_DIR / 'training_hyperparameters.csv'
    df.to_csv(csv_out, index=False)

    # also render as a simple PNG table
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.35 * (len(df) + 1)))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title('Training & RL Hyperparameters', pad=12)
    fig.tight_layout()
    png_out = OUT_DIR / 'training_hyperparameters.png'
    fig.savefig(png_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return csv_out, png_out


def gen_convergence_summary():
    ja = OUT_DIR / 'training_analysis.json'
    if not ja.exists():
        return None
    with open(ja, 'r') as f:
        data = json.load(f)
    epochs = np.array(data['epochs'])
    b_iou = np.array(data['baseline_iou'])
    e_iou = np.array(data['enhanced_iou'])

    # Final metrics (mean of last 5)
    b_final = float(np.mean(b_iou[-5:]))
    e_final = float(np.mean(e_iou[-5:]))

    # Stability (std of last 5)
    b_stab = float(np.std(b_iou[-5:]))
    e_stab = float(np.std(e_iou[-5:]))

    # Convergence epoch: first epoch within 95% of final-plateau range
    def convergence_epoch(curve):
        target = np.mean(curve[-5:])
        thresh = target * 0.95
        idx = np.argmax(curve >= thresh)
        # handle case where never reaches
        return int(epochs[idx]) if curve.max() >= thresh else int(epochs[-1])

    b_conv = convergence_epoch(b_iou)
    e_conv = convergence_epoch(e_iou)

    df = pd.DataFrame([
        {
            'Model': 'Baseline',
            'Final IoU (mean last 5)': round(b_final, 3),
            'Stability (std last 5)': round(b_stab, 3),
            'Convergence Epoch (95%)': b_conv
        },
        {
            'Model': 'Enhanced',
            'Final IoU (mean last 5)': round(e_final, 3),
            'Stability (std last 5)': round(e_stab, 3),
            'Convergence Epoch (95%)': e_conv
        }
    ])
    csv_out = OUT_DIR / 'training_convergence_summary.csv'
    df.to_csv(csv_out, index=False)

    # quick PNG table
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title('Training Convergence Summary (Validation IoU)')
    fig.tight_layout()
    png_out = OUT_DIR / 'training_convergence_summary.png'
    fig.savefig(png_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return csv_out, png_out


def main():
    th_png = gen_throughput_plot()
    hp_csv, hp_png = gen_hyperparams_csv()
    conv_csv, conv_png = gen_convergence_summary()
    print('Generated:', th_png, hp_csv, conv_csv)


if __name__ == '__main__':
    main()
