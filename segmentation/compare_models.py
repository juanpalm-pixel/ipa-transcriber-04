"""
Compare segmentation results across different models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_FILE = SCRIPT_DIR / "segmentation_results.csv"
REPORT_DIR = PROJECT_ROOT / "verification_1" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_results():
    """Load segmentation results"""
    try:
        df = pd.read_csv(RESULTS_FILE)
        print(f"Loaded {len(df)} segments from {RESULTS_FILE}")
        return df
    except FileNotFoundError:
        print(f"ERROR: {RESULTS_FILE} not found. Run run_segmentation.py first.")
        sys.exit(1)

def analyze_segments(df):
    """Analyze segmentation patterns"""
    
    print("\n" + "=" * 80)
    print("SEGMENTATION ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total segments: {len(df)}")
    print(f"  Total models tested: {df['model_name'].nunique()}")
    print(f"  Input files processed: {df['input_file'].nunique()}")
    
    # Per-model statistics
    print("\nPer-Model Statistics:")
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        print(f"\n  {model}:")
        print(f"    Segments: {len(model_df)}")
        print(f"    Avg duration: {model_df['duration_ms'].mean():.0f} ms")
        print(f"    Std duration: {model_df['duration_ms'].std():.0f} ms")
        print(f"    Min duration: {model_df['duration_ms'].min():.0f} ms")
        print(f"    Max duration: {model_df['duration_ms'].max():.0f} ms")
        print(f"    Processing time: {model_df['processing_time_s'].iloc[0]:.2f} s")
    
    # Duration distribution
    print("\nDuration Distribution by Model:")
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        durations = model_df['duration_ms']
        
        print(f"\n  {model}:")
        print(f"    Quartiles: Q1={durations.quantile(0.25):.0f}ms, "
              f"Median={durations.median():.0f}ms, Q3={durations.quantile(0.75):.0f}ms")
        
        # Duration bins
        bins = [0, 200, 500, 1000, 2000, float('inf')]
        labels = ['<200ms', '200-500ms', '500-1000ms', '1-2s', '>2s']
        counts = pd.cut(durations, bins=bins, labels=labels).value_counts().sort_index()
        
        print("    Duration bins:")
        for label, count in counts.items():
            pct = (count / len(durations)) * 100
            print(f"      {label}: {count} ({pct:.1f}%)")

def create_visualizations(df):
    """Create comparison visualizations"""
    
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Duration distribution by model
    ax = axes[0, 0]
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        ax.hist(model_df['duration_ms'], bins=50, alpha=0.6, label=model)
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Segment Duration Distribution by Model')
    ax.legend()
    ax.set_xlim(0, min(5000, df['duration_ms'].max()))
    
    # 2. Box plot of durations
    ax = axes[0, 1]
    df.boxplot(column='duration_ms', by='model_name', ax=ax)
    ax.set_xlabel('Model')
    ax.set_ylabel('Duration (ms)')
    ax.set_title('Duration Distribution Comparison')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    # 3. Segment count comparison
    ax = axes[1, 0]
    segment_counts = df['model_name'].value_counts()
    segment_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('Model')
    ax.set_ylabel('Number of Segments')
    ax.set_title('Total Segments per Model')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    # 4. Processing time comparison
    ax = axes[1, 1]
    proc_times = df.groupby('model_name')['processing_time_s'].first()
    proc_times.plot(kind='bar', ax=ax, color='coral')
    ax.set_xlabel('Model')
    ax.set_ylabel('Processing Time (seconds)')
    ax.set_title('Processing Time Comparison')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_file = REPORT_DIR / "segmentation_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {output_file}")
    plt.close()
    
    # Timeline visualization for each model
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model].sort_values('start_time_ms')
        
        fig, ax = plt.subplots(figsize=(15, 4))
        
        for idx, row in model_df.iterrows():
            start_s = row['start_time_ms'] / 1000
            duration_s = row['duration_ms'] / 1000
            ax.barh(0, duration_s, left=start_s, height=0.8, alpha=0.7, color='steelblue')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_yticks([])
        ax.set_title(f'Segmentation Timeline - {model}')
        ax.grid(axis='x', alpha=0.3)
        
        output_file = REPORT_DIR / f"timeline_{model}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved timeline: {output_file}")
        plt.close()

def find_overlaps(df):
    """Find temporal overlaps between models"""
    
    print("\nAnalyzing temporal agreement between models...")
    
    models = df['model_name'].unique()
    if len(models) < 2:
        print("  Need at least 2 models to compare overlaps")
        return
    
    # Convert to intervals for each model
    overlaps = {}
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            df1 = df[df['model_name'] == model1].copy()
            df2 = df[df['model_name'] == model2].copy()
            
            overlap_count = 0
            
            # Check for overlaps
            for _, seg1 in df1.iterrows():
                for _, seg2 in df2.iterrows():
                    # Check if segments overlap
                    start_max = max(seg1['start_time_ms'], seg2['start_time_ms'])
                    end_min = min(seg1['end_time_ms'], seg2['end_time_ms'])
                    
                    if start_max < end_min:
                        overlap_count += 1
                        break  # Count each seg1 only once
            
            overlap_pct = (overlap_count / len(df1)) * 100 if len(df1) > 0 else 0
            overlaps[f"{model1} vs {model2}"] = {
                'count': overlap_count,
                'percentage': overlap_pct,
                'total_segments_model1': len(df1)
            }
            
            print(f"\n  {model1} vs {model2}:")
            print(f"    {overlap_count}/{len(df1)} segments overlap ({overlap_pct:.1f}%)")
    
    return overlaps

def generate_report(df):
    """Generate HTML report"""
    
    print("\nGenerating HTML report...")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Segmentation Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
            .section {{ margin: 30px 0; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Audio Segmentation Comparison Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Summary</h2>
            <p>Total segments: <span class="metric">{len(df)}</span></p>
            <p>Models tested: <span class="metric">{df['model_name'].nunique()}</span></p>
        </div>
        
        <div class="section">
            <h2>Model Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Segments</th>
                    <th>Avg Duration (ms)</th>
                    <th>Std Duration (ms)</th>
                    <th>Processing Time (s)</th>
                </tr>
    """
    
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{len(model_df)}</td>
                    <td>{model_df['duration_ms'].mean():.0f}</td>
                    <td>{model_df['duration_ms'].std():.0f}</td>
                    <td>{model_df['processing_time_s'].iloc[0]:.2f}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <img src="segmentation_comparison.png" alt="Segmentation Comparison">
        </div>
    """
    
    # Add timeline images
    for model in df['model_name'].unique():
        html += f"""
        <div class="section">
            <h3>Timeline: {model}</h3>
            <img src="timeline_{model}.png" alt="Timeline for {model}">
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    output_file = REPORT_DIR / "segmentation_report.html"
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"  ✓ Saved report: {output_file}")

def main():
    """Main comparison function"""
    
    print("=" * 80)
    print("SEGMENTATION MODEL COMPARISON")
    print("=" * 80)
    
    # Load results
    df = load_results()
    if df is None:
        return
    
    # Analyze
    analyze_segments(df)
    
    # Find overlaps
    find_overlaps(df)
    
    # Visualize
    create_visualizations(df)
    
    # Generate report
    generate_report(df)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print(f"Reports saved to: {REPORT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
