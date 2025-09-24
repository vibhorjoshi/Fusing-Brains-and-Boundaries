"""
Analyze state-level performance metrics and generate detailed comparison report
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load IoU comparison data
iou_data = pd.read_csv("outputs/logs/iou_comparison.csv")

# Create detailed analysis
analysis = {
    'state_analysis': {},
    'method_analysis': {},
    'overall_analysis': {}
}

# Average IoU per method across all states
method_cols = ['Mask_R-CNN', 'RT', 'RR', 'FER', 'RL_Fusion_CPU', 'RL_Fusion_GPU']
method_averages = iou_data[method_cols].mean()

# Calculate improvement from CPU to GPU
avg_cpu = iou_data['RL_Fusion_CPU'].mean()
avg_gpu = iou_data['RL_Fusion_GPU'].mean()
improvement_pct = (avg_gpu - avg_cpu) / avg_cpu * 100

# Overall analysis
analysis['overall_analysis'] = {
    'method_averages': method_averages.to_dict(),
    'cpu_gpu_improvement': {
        'absolute': avg_gpu - avg_cpu,
        'percentage': improvement_pct
    },
    'best_state': iou_data.loc[iou_data['RL_Fusion_GPU'].idxmax(), 'State'],
    'best_state_iou': iou_data['RL_Fusion_GPU'].max(),
    'worst_state': iou_data.loc[iou_data['RL_Fusion_GPU'].idxmin(), 'State'],
    'worst_state_iou': iou_data['RL_Fusion_GPU'].min(),
}

# State-by-state analysis
for state in iou_data['State']:
    state_row = iou_data[iou_data['State'] == state].iloc[0]
    state_data = {
        'best_method': method_cols[np.argmax(state_row[method_cols])],
        'best_method_iou': max(state_row[method_cols]),
        'cpu_gpu_improvement': {
            'absolute': state_row['RL_Fusion_GPU'] - state_row['RL_Fusion_CPU'],
            'percentage': state_row['Improvement']
        }
    }
    analysis['state_analysis'][state] = state_data

# Method analysis
for method in method_cols:
    best_state = iou_data.loc[iou_data[method].idxmax(), 'State']
    worst_state = iou_data.loc[iou_data[method].idxmin(), 'State']
    method_data = {
        'average_iou': method_averages[method],
        'best_state': best_state,
        'best_state_iou': iou_data.loc[iou_data['State'] == best_state, method].iloc[0],
        'worst_state': worst_state,
        'worst_state_iou': iou_data.loc[iou_data['State'] == worst_state, method].iloc[0],
        'stddev': iou_data[method].std()
    }
    analysis['method_analysis'][method] = method_data

# Generate heatmap of IoU values
plt.figure(figsize=(12, 8))
heatmap_data = iou_data.set_index('State')[method_cols]
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': 'IoU Score'})
plt.title('IoU Scores Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig("outputs/figures/iou_heatmap.png", dpi=150)
plt.close()

# Create performance improvement chart
improvement = iou_data[['State', 'Improvement']].sort_values('Improvement', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(improvement['State'], improvement['Improvement'], color=sns.color_palette('viridis', len(improvement)))
plt.title("GPU vs CPU Performance Improvement by State", fontsize=14)
plt.xlabel("State", fontsize=12)
plt.ylabel("Improvement (%)", fontsize=12)
plt.grid(True, axis='y', alpha=0.3)

# Add values on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("outputs/figures/state_improvement.png", dpi=150)
plt.close()

# Generate summary table as markdown
with open("outputs/logs/performance_summary.md", "w") as f:
    f.write("# Building Footprint Extraction Performance Summary\n\n")
    
    f.write("## Overall Performance\n\n")
    f.write("| Method | Average IoU |\n")
    f.write("|--------|------------|\n")
    for method, avg in method_averages.items():
        f.write(f"| {method.replace('_', ' ')} | {avg:.4f} |\n")
    
    f.write(f"\n**GPU vs CPU Improvement:** {improvement_pct:.2f}% average increase in IoU\n\n")
    
    f.write("## State-by-State Analysis\n\n")
    f.write("| State | Best Method | Best IoU | GPU Improvement |\n")
    f.write("|-------|-------------|----------|----------------|\n")
    for state, data in analysis['state_analysis'].items():
        f.write(f"| {state} | {data['best_method'].replace('_', ' ')} | {data['best_method_iou']:.4f} | {data['cpu_gpu_improvement']['percentage']:.2f}% |\n")

# Output summary to console
print("Performance analysis complete. Results saved to outputs/logs/performance_summary.md")
print(f"\nOverall GPU vs CPU Improvement: {improvement_pct:.2f}%")
print(f"Best performing state: {analysis['overall_analysis']['best_state']} (IoU: {analysis['overall_analysis']['best_state_iou']:.4f})")
print(f"Best performing method: {method_cols[np.argmax(method_averages)].replace('_', ' ')} (Average IoU: {method_averages.max():.4f})")