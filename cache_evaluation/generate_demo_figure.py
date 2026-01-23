#!/usr/bin/env python3
"""
Generate a publication-quality figure demonstrating multi-tier caching functionality.

This script creates a comparison plot showing:
- No Cache (baseline)
- Single-Tier Cache (SSD-based file cache)
- Multi-Tier Cache (Memory + SSD hierarchical cache)

Uses generic, system-agnostic terminology suitable for demo papers.

QPS Calculation:
    throughput_qps = avg_bandwidth_gbps × 100

    Where avg_bandwidth is the weighted average based on storage tier distribution:
        avg_bandwidth = (dram_pct × 40 GB/s) + (ssd_pct × 4 GB/s) + (s3_pct × 1 GB/s)

    This assumes 100 queries/second per GB/s of bandwidth, reflecting that:
    - Memory queries are ~40x faster than remote storage
    - SSD queries are ~4x faster than remote storage
    - Remote (S3) queries are the baseline (1 GB/s)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication-quality defaults for single-column figure
plt.rcParams['font.size'] = 14           # Increased from 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2     # Slightly thicker
plt.rcParams['axes.labelsize'] = 15      # Larger axis labels
plt.rcParams['axes.titlesize'] = 16      # Larger subplot titles
plt.rcParams['xtick.labelsize'] = 13     # Larger tick labels
plt.rcParams['ytick.labelsize'] = 13     # Larger tick labels
plt.rcParams['legend.fontsize'] = 12     # Larger legend
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

def load_results(results_file: str) -> dict:
    """Load production analysis results."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_demo_figure(results: dict, output_file: str = "cache_comparison_demo.pdf"):
    """
    Create a multi-panel figure comparing caching strategies.

    Panels:
    1. Overall cache hit rates
    2. Storage tier distribution (Memory/SSD/Remote)
    3. Query latency improvement
    4. Throughput improvement
    """

    # Extract relevant configurations for demo
    configs = {
        'No Cache': results['configurations']['no_cache'],
        'Single-Tier\n(SSD Cache)': results['configurations']['P'],
        'Multi-Tier\n(Mem+SSD)': results['configurations']['V+H+P']
    }

    labels = list(configs.keys())
    n_configs = len(labels)

    # Create figure with 2x2 subplots (larger for single column)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle('Multi-Tier Caching: Performance Comparison',
                 fontsize=18, fontweight='bold', y=0.99)

    # Color scheme (professional, colorblind-friendly)
    colors = {
        'primary': '#2E86AB',      # Blue
        'secondary': '#A23B72',    # Purple
        'tertiary': '#F18F01',     # Orange
        'memory': '#06A77D',       # Green (for DRAM)
        'ssd': '#F18F01',          # Orange (for SSD)
        'remote': '#D32F2F'        # Red (for S3/Remote)
    }

    # =================================================================
    # Panel 1: Cache Hit Rates
    # =================================================================
    ax1 = axes[0, 0]
    hit_rates = [configs[label]['overall_hit_rate'] for label in labels]

    bars1 = ax1.bar(range(n_configs), hit_rates,
                    color=[colors['primary'], colors['secondary'], colors['tertiary']],
                    alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, hit_rates)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=13)

    ax1.set_ylabel('Cache Hit Rate (%)', fontweight='bold')
    ax1.set_title('(a) Overall Cache Hit Rate', loc='left', pad=10)
    ax1.set_xticks(range(n_configs))
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', pad=8)  # Add padding to x-axis labels
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% target')

    # =================================================================
    # Panel 2: Storage Tier Distribution (Stacked Bar)
    # =================================================================
    ax2 = axes[0, 1]

    dram_pcts = [configs[label]['dram_pct'] for label in labels]
    ssd_pcts = [configs[label]['ssd_pct'] for label in labels]
    s3_pcts = [configs[label]['s3_pct'] for label in labels]

    x_pos = np.arange(n_configs)
    width = 0.6

    # Stacked bars
    ax2.bar(x_pos, dram_pcts, width, label='Memory (DRAM)',
            color=colors['memory'], edgecolor='black', linewidth=1)
    ax2.bar(x_pos, ssd_pcts, width, bottom=dram_pcts,
            label='Fast Storage (SSD)',
            color=colors['ssd'], edgecolor='black', linewidth=1)
    ax2.bar(x_pos, s3_pcts, width,
            bottom=[d+s for d, s in zip(dram_pcts, ssd_pcts)],
            label='Remote Storage (S3/Cloud)',
            color=colors['remote'], edgecolor='black', linewidth=1)

    # Add percentage labels inside stacked bars
    for i in range(n_configs):
        # DRAM label
        if dram_pcts[i] > 5:
            ax2.text(i, dram_pcts[i]/2, f'{dram_pcts[i]:.0f}%',
                    ha='center', va='center', fontweight='bold',
                    fontsize=12, color='white')
        # SSD label
        if ssd_pcts[i] > 5:
            ax2.text(i, dram_pcts[i] + ssd_pcts[i]/2,
                    f'{ssd_pcts[i]:.0f}%',
                    ha='center', va='center', fontweight='bold',
                    fontsize=12, color='white')
        # S3 label
        if s3_pcts[i] > 5:
            ax2.text(i, dram_pcts[i] + ssd_pcts[i] + s3_pcts[i]/2,
                    f'{s3_pcts[i]:.0f}%',
                    ha='center', va='center', fontweight='bold',
                    fontsize=12, color='white')

    ax2.set_ylabel('Query Distribution (%)', fontweight='bold')
    ax2.set_title('(b) Storage Tier Distribution', loc='left', pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', pad=8)  # Add padding to x-axis labels
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # =================================================================
    # Panel 3: Latency Reduction
    # =================================================================
    ax3 = axes[1, 0]

    baseline_latency = configs['No Cache']['extended_metrics']['latency_ms']
    latencies = [configs[label]['extended_metrics']['latency_ms'] for label in labels]
    latency_reduction = [(baseline_latency - lat) / baseline_latency * 100 for lat in latencies]

    bars3 = ax3.bar(range(n_configs), latency_reduction,
                    color=[colors['primary'], colors['secondary'], colors['tertiary']],
                    alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, val, lat) in enumerate(zip(bars3, latency_reduction,
                                             latencies)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.0f}%\n({lat:.0f}ms)', ha='center', va='bottom',
                fontweight='bold', fontsize=12)

    ax3.set_ylabel('Latency Reduction (%)', fontweight='bold')
    ax3.set_title('(c) Query Latency Improvement', loc='left', pad=10)
    ax3.set_xticks(range(n_configs))
    ax3.set_xticklabels(labels, fontsize=12)
    ax3.set_ylim(-10, 110)  # Increased to prevent collision with title
    ax3.tick_params(axis='x', pad=8)  # Add padding to x-axis labels
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # =================================================================
    # Panel 4: Throughput Improvement
    # =================================================================
    ax4 = axes[1, 1]

    baseline_throughput = configs['No Cache']['extended_metrics']['throughput_qps']
    throughputs = [configs[label]['extended_metrics']['throughput_qps'] for label in labels]
    throughput_speedup = [tput / baseline_throughput for tput in throughputs]

    bars4 = ax4.bar(range(n_configs), throughput_speedup,
                    color=[colors['primary'], colors['secondary'], colors['tertiary']],
                    alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, val, tput) in enumerate(zip(bars4, throughput_speedup,
                                              throughputs)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.1f}x\n({tput:.0f} QPS)', ha='center', va='bottom',
                fontweight='bold', fontsize=12)

    ax4.set_ylabel('Throughput Speedup (x baseline)', fontweight='bold')
    ax4.set_title('(d) Query Throughput Speedup', loc='left', pad=10)
    ax4.set_xticks(range(n_configs))
    ax4.set_xticklabels(labels, fontsize=12)
    ax4.set_ylim(0, max(throughput_speedup) * 1.2)
    ax4.tick_params(axis='x', pad=8)  # Add padding to x-axis labels
    ax4.axhline(y=1, color='gray', linestyle='--',
                linewidth=1, alpha=0.5, label='Baseline')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.legend(loc='upper left')

    # =================================================================
    # Add configuration details as footer
    # =================================================================
    metadata = results['metadata']
    footer_text = (
        f"Workload: {metadata['queries_simulated']} queries, "
        f"{int(metadata['read_update_ratio']*100)}% reads / "
        f"{int((1-metadata['read_update_ratio'])*100)}% updates"
    )
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=11,
             style='italic', color='gray')

    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figure
    output_path = Path(output_file)
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path.absolute()}")

    # Also save PNG for preview
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Preview saved to: {png_path.absolute()}")

    plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for label in labels:
        config = configs[label]
        # Remove newlines from label for printing
        print_label = label.replace('\n', ' ')
        print(f"\n{print_label}:")
        print(f"  Cache Hit Rate:     {config['overall_hit_rate']:.1f}%")
        print(f"  Memory Tier:        {config['dram_pct']:.1f}%")
        print(f"  SSD Tier:           {config['ssd_pct']:.1f}%")
        print(f"  Remote (S3) Tier:   {config['s3_pct']:.1f}%")
        print(f"  Query Latency:      {config['extended_metrics']['latency_ms']:.1f} ms")
        print(f"  Throughput:         {config['extended_metrics']['throughput_qps']:.1f} QPS")
    print("="*60)


def main():
    """Main entry point."""
    # Locate results file
    results_file = Path(__file__).parent / "results" / "production_analysis_results.json"

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please run the cache evaluation first to generate results.")
        return

    # Load results
    print(f"Loading results from: {results_file}")
    results = load_results(str(results_file))

    # Generate figure
    output_file = Path(__file__).parent / "cache_comparison_demo.pdf"
    print(f"\nGenerating demo figure...")
    create_demo_figure(results, str(output_file))

    print("\n✓ Demo figure generation complete!")
    print(f"\nThe figure is ready for inclusion in your demo paper.")
    print(f"It compares three caching strategies:")
    print(f"  1. No Cache (baseline)")
    print(f"  2. Single-Tier Cache (SSD-based file cache)")
    print(f"  3. Multi-Tier Cache (Memory + SSD hierarchical cache)")


if __name__ == "__main__":
    main()
