#!/usr/bin/env python3
"""
ENHANCED Multi-Tier Cache Analysis - Complete Implementation

This script provides ALL requested features:
1. ALL valid cache configurations (5 total)
2. Extended metrics (latency, throughput, bandwidth, efficiency)
3. Performance heatmap
4. Efficiency vs complexity plots
5. Degradation analysis
6. Comprehensive rankings

Run this to get complete analysis!
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.getcwd()))

from cloudglide.config import MultiTierCacheConfig
from cloudglide.cache_dynamics import CacheDynamicsModel, CacheState
from cloudglide.cache_integration import create_cache_tier_assigner
from cloudglide.cache_metrics import CacheMetricsTracker
from cache_evaluation.multi_tier_cache_model import MultiTierCacheModel, MultiTierCacheConfig as EvalConfig

# Create results directory
os.makedirs('results', exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

print("=" * 80)
print("ENHANCED MULTI-TIER CACHE ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. DEFINE ALL CONFIGURATIONS
# ============================================================================

READ_UPDATE_RATIO = 0.7
WARMUP_RATE = 100.0
CONVERGENCE_QUERIES = 1000

configurations = {
    'no_cache': MultiTierCacheConfig(
        enabled_tiers=[],
        warmup_rate=0.0,
        read_update_ratio=READ_UPDATE_RATIO
    ),
    'parquet_only': MultiTierCacheConfig(
        enabled_tiers=["PARQUET_CACHE"],
        warmup_rate=WARMUP_RATE,
        read_update_ratio=READ_UPDATE_RATIO,
        parquet_cache_volatility=0.1,
        convergence_queries=CONVERGENCE_QUERIES
    ),
    'hex_parquet': MultiTierCacheConfig(
        enabled_tiers=["HEX_CACHE", "PARQUET_CACHE"],
        warmup_rate=WARMUP_RATE,
        read_update_ratio=READ_UPDATE_RATIO,
        hex_cache_volatility=0.3,
        parquet_cache_volatility=0.1,
        convergence_queries=CONVERGENCE_QUERIES
    ),
    'view_hex_parquet': MultiTierCacheConfig(
        enabled_tiers=["VIEW_CACHE", "HEX_CACHE", "PARQUET_CACHE"],
        warmup_rate=WARMUP_RATE,
        read_update_ratio=READ_UPDATE_RATIO,
        view_cache_volatility=0.7,
        hex_cache_volatility=0.3,
        parquet_cache_volatility=0.1,
        convergence_queries=CONVERGENCE_QUERIES
    ),
    'all_four_tiers': MultiTierCacheConfig(
        enabled_tiers=["MDS_CUBE_CACHE", "VIEW_CACHE", "HEX_CACHE", "PARQUET_CACHE"],
        warmup_rate=WARMUP_RATE,
        read_update_ratio=READ_UPDATE_RATIO,
        cube_cache_volatility=1.0,
        view_cache_volatility=0.7,
        hex_cache_volatility=0.3,
        parquet_cache_volatility=0.1,
        convergence_queries=CONVERGENCE_QUERIES
    )
}

print(f"\n✅ Configured {len(configurations)} valid cache configurations")

# ============================================================================
# 2. EXTENDED METRICS
# ============================================================================

BANDWIDTHS = {'DRAM': 40.0, 'SSD': 4.0, 'S3': 1.0}  # GB/s

def calculate_extended_metrics(result):
    """Calculate latency, throughput, bandwidth, efficiency"""
    avg_bandwidth = (
        (result['dram_pct'] / 100.0) * BANDWIDTHS['DRAM'] +
        (result['ssd_pct'] / 100.0) * BANDWIDTHS['SSD'] +
        (result['s3_pct'] / 100.0) * BANDWIDTHS['S3']
    )

    latency_ms = (1.0 / avg_bandwidth) * 1000 if avg_bandwidth > 0 else float('inf')
    throughput_qps = avg_bandwidth * 100
    bandwidth_savings_pct = ((avg_bandwidth - BANDWIDTHS['S3']) / BANDWIDTHS['S3']) * 100
    efficiency_score = throughput_qps / max(result.get('tier_count', 1), 1)

    return {
        'avg_bandwidth_gbps': avg_bandwidth,
        'latency_ms': latency_ms,
        'throughput_qps': throughput_qps,
        'bandwidth_savings_pct': bandwidth_savings_pct,
        'efficiency_score': efficiency_score
    }

# ============================================================================
# 3. SIMULATION
# ============================================================================

def simulate_cache_configuration(config, config_name, n_queries=2000, seed=42):
    """Simulate with production model"""
    if not config.enabled_tiers:
        return {
            'config_name': config_name,
            'tier_count': 0,
            'overall_hit_rate': 0.0,
            'dram_pct': 0.0,
            'ssd_pct': 0.0,
            's3_pct': 100.0,
            'n_updates': 0,
            'steady_state_hit_rate': 0.0
        }

    state = CacheState()
    dynamics_model = CacheDynamicsModel(config, state)
    dynamics_model.set_seed(seed)

    assigner = create_cache_tier_assigner(
        cache_config=config,
        cache_state=state,
        use_dynamic_model=True
    )

    tracker = CacheMetricsTracker()
    hit_rates_over_time = []

    for i in range(n_queries):
        tier = assigner(architecture=0, n=i, warmup_rate=None)
        tracker.record_assignment(tier)

        if tracker.total_assignments > 0:
            cache_hits = tracker.tier_counter['DRAM'] + tracker.tier_counter['SSD']
            hit_rate = cache_hits / tracker.total_assignments
        else:
            hit_rate = 0.0
        hit_rates_over_time.append(hit_rate)

    summary = tracker.get_summary()
    steady_state = np.mean(hit_rates_over_time[-500:]) if len(hit_rates_over_time) >= 500 else summary['overall_cache_hit_rate'] / 100.0

    return {
        'config_name': config_name,
        'tier_count': len(config.enabled_tiers),
        'enabled_tiers': config.enabled_tiers,
        'overall_hit_rate': summary['overall_cache_hit_rate'],
        'dram_pct': summary['dram_hit_rate'],
        'ssd_pct': summary['ssd_hit_rate'],
        's3_pct': summary['s3_miss_rate'],
        'n_updates': len(state.update_events),
        'steady_state_hit_rate': steady_state * 100,
        'hit_rates_over_time': hit_rates_over_time
    }

print("\n🚀 Running simulations...")
results = {}
for config_name, config in configurations.items():
    result = simulate_cache_configuration(config, config_name)
    # Add extended metrics
    extended = calculate_extended_metrics(result)
    result.update(extended)
    results[config_name] = result
    print(f"  ✓ {config_name:20s} - Hit: {result['overall_hit_rate']:.1f}%, "
          f"Throughput: {result['throughput_qps']:.0f} QPS, "
          f"Latency: {result['latency_ms']:.1f}ms")

print("✅ All simulations complete!\n")

# ============================================================================
# 4. PERFORMANCE HEATMAP
# ============================================================================

print("📊 Creating performance heatmap...")

# Prepare data for heatmap
heatmap_data = []
for name, result in results.items():
    heatmap_data.append({
        'Config': name,
        'Hit Rate (%)': result['overall_hit_rate'],
        'Throughput (QPS)': result['throughput_qps'],
        'Latency (ms)': result['latency_ms'],
        'Efficiency': result['efficiency_score'],
        'Bandwidth (GB/s)': result['avg_bandwidth_gbps']
    })

df_heat = pd.DataFrame(heatmap_data).set_index('Config')

# Normalize for heatmap (0-1 scale, invert latency)
df_norm = df_heat.copy()
for col in df_norm.columns:
    if col == 'Latency (ms)':
        # Invert: lower is better
        df_norm[col] = 1 - (df_norm[col] / df_norm[col].max())
    else:
        # Higher is better
        df_norm[col] = df_norm[col] / df_norm[col].max()

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    df_norm,
    annot=df_heat.values,  # Show actual values
    fmt='.1f',
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    ax=ax,
    cbar_kws={'label': 'Normalized Score (0=worst, 1=best)'},
    linewidths=0.5
)
ax.set_title('Performance Heatmap Across All Metrics', fontweight='bold', fontsize=14, pad=20)
ax.set_ylabel('Configuration', fontweight='bold')
plt.tight_layout()
plt.savefig('results/performance_heatmap.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: results/performance_heatmap.png")

# ============================================================================
# 5. EFFICIENCY VS COMPLEXITY PLOT
# ============================================================================

print("📊 Creating efficiency vs complexity plot...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Tiers vs Performance Metrics
ax1 = axes[0]
tier_counts = [results[name]['tier_count'] for name in results.keys()]
throughputs = [results[name]['throughput_qps'] for name in results.keys()]
hit_rates = [results[name]['overall_hit_rate'] for name in results.keys()]

ax1.scatter(tier_counts, throughputs, s=200, c=hit_rates, cmap='RdYlGn',
           alpha=0.7, edgecolor='black', linewidth=2, vmin=0, vmax=100)
for name in results.keys():
    ax1.annotate(name, (results[name]['tier_count'], results[name]['throughput_qps']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax1.set_xlabel('Number of Tiers (Complexity)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Throughput (QPS)', fontweight='bold', fontsize=12)
ax1.set_title('Complexity vs Performance', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
cbar1.set_label('Hit Rate (%)', fontweight='bold')

# Plot 2: Efficiency Score
ax2 = axes[1]
efficiencies = [results[name]['efficiency_score'] for name in results.keys()]
colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

bars = ax2.bar(range(len(results)), efficiencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(results)))
ax2.set_xticklabels(list(results.keys()), rotation=45, ha='right')
ax2.set_ylabel('Efficiency Score (Throughput/Tier)', fontweight='bold', fontsize=12)
ax2.set_title('Efficiency Score by Configuration', fontweight='bold', fontsize=13)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/efficiency_vs_complexity.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: results/efficiency_vs_complexity.png")

# ============================================================================
# 6. COMPREHENSIVE RANKINGS
# ============================================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE PERFORMANCE RANKINGS")
print("=" * 80)

# Create comprehensive summary
summary_data = []
for name, result in results.items():
    summary_data.append({
        'Configuration': name,
        'Tiers': result['tier_count'],
        'Hit Rate (%)': result['overall_hit_rate'],
        'Throughput (QPS)': result['throughput_qps'],
        'Latency (ms)': result['latency_ms'],
        'Bandwidth (GB/s)': result['avg_bandwidth_gbps'],
        'Efficiency Score': result['efficiency_score'],
        'DRAM %': result['dram_pct'],
        'SSD %': result['ssd_pct'],
        'S3 %': result['s3_pct']
    })

df_summary = pd.DataFrame(summary_data)

print("\n📊 RANKED BY THROUGHPUT:")
print(df_summary.sort_values('Throughput (QPS)', ascending=False)[
    ['Configuration', 'Tiers', 'Throughput (QPS)', 'Hit Rate (%)', 'Latency (ms)']
].to_string(index=False))

print("\n📊 RANKED BY EFFICIENCY:")
print(df_summary.sort_values('Efficiency Score', ascending=False)[
    ['Configuration', 'Tiers', 'Efficiency Score', 'Throughput (QPS)', 'Hit Rate (%)']
].to_string(index=False))

print("\n📊 RANKED BY HIT RATE:")
print(df_summary.sort_values('Hit Rate (%)', ascending=False)[
    ['Configuration', 'Tiers', 'Hit Rate (%)', 'DRAM %', 'SSD %', 'S3 %']
].to_string(index=False))

# Save to CSV
df_summary.to_csv('results/comprehensive_rankings.csv', index=False)
print("\n💾 Saved: results/comprehensive_rankings.csv")

# ============================================================================
# 7. HIT RATE DEGRADATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("HIT RATE DEGRADATION ANALYSIS (vs Baseline)")
print("=" * 80)

# Compare to no_cache baseline
baseline_throughput = results['no_cache']['throughput_qps']

degradation_data = []
for name, result in results.items():
    if name == 'no_cache':
        continue

    throughput_improvement = ((result['throughput_qps'] - baseline_throughput) / baseline_throughput) * 100

    degradation_data.append({
        'Configuration': name,
        'Tiers': result['tier_count'],
        'Hit Rate (%)': result['overall_hit_rate'],
        'Throughput Improvement (%)': throughput_improvement,
        'Latency vs Baseline': result['latency_ms'] / results['no_cache']['latency_ms']
    })

df_degrad = pd.DataFrame(degradation_data)
print(df_degrad.to_string(index=False))

# Visualize degradation
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_degrad))
width = 0.35

bars1 = ax.bar(x - width/2, df_degrad['Hit Rate (%)'], width,
              label='Hit Rate (%)', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, df_degrad['Throughput Improvement (%)'], width,
              label='Throughput Improvement (%)', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Configuration', fontweight='bold')
ax.set_ylabel('Percentage', fontweight='bold')
ax.set_title('Hit Rate & Throughput Improvement vs Baseline', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df_degrad['Configuration'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/degradation_analysis.png', dpi=150, bbox_inches='tight')
print("\n💾 Saved: results/degradation_analysis.png")

# ============================================================================
# 8. EXPORT ALL RESULTS
# ============================================================================

export_data = {
    'metadata': {
        'model': 'Enhanced Production CloudGlide Dynamic Cache Model',
        'read_update_ratio': READ_UPDATE_RATIO,
        'warmup_rate': WARMUP_RATE,
        'queries_simulated': 2000,
        'metrics': ['hit_rate', 'throughput', 'latency', 'bandwidth', 'efficiency']
    },
    'configurations': {}
}

for name, result in results.items():
    export_data['configurations'][name] = {
        'tier_count': result['tier_count'],
        'overall_hit_rate': result['overall_hit_rate'],
        'throughput_qps': result['throughput_qps'],
        'latency_ms': result['latency_ms'],
        'efficiency_score': result['efficiency_score'],
        'bandwidth_gbps': result['avg_bandwidth_gbps']
    }

with open('results/enhanced_analysis_results.json', 'w') as f:
    json.dump(export_data, f, indent=2)

print("\n💾 Saved: results/enhanced_analysis_results.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ ENHANCED ANALYSIS COMPLETE!")
print("=" * 80)

best_throughput = df_summary.iloc[df_summary['Throughput (QPS)'].idxmax()]
best_efficiency = df_summary.iloc[df_summary['Efficiency Score'].idxmax()]
best_hit_rate = df_summary.iloc[df_summary['Hit Rate (%)'].idxmax()]

print(f"\n🏆 BEST THROUGHPUT: {best_throughput['Configuration']}")
print(f"   Throughput: {best_throughput['Throughput (QPS)']:.0f} QPS")
print(f"   Hit Rate: {best_throughput['Hit Rate (%)']:.1f}%")

print(f"\n🏆 BEST EFFICIENCY: {best_efficiency['Configuration']}")
print(f"   Efficiency: {best_efficiency['Efficiency Score']:.0f} QPS/tier")
print(f"   Throughput: {best_efficiency['Throughput (QPS)']:.0f} QPS")

print(f"\n🏆 BEST HIT RATE: {best_hit_rate['Configuration']}")
print(f"   Hit Rate: {best_hit_rate['Hit Rate (%)']:.1f}%")
print(f"   Throughput: {best_hit_rate['Throughput (QPS)']:.0f} QPS")

print("\n📁 Generated Files:")
print("   • results/performance_heatmap.png")
print("   • results/efficiency_vs_complexity.png")
print("   • results/degradation_analysis.png")
print("   • results/comprehensive_rankings.csv")
print("   • results/enhanced_analysis_results.json")

print("\n" + "=" * 80)
plt.show()
