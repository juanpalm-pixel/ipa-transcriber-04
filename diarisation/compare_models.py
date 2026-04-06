"""
Compare diarisation results across different models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from collections import defaultdict

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_FILE = SCRIPT_DIR / "diarisation_results.csv"
REPORT_DIR = PROJECT_ROOT / "verification_2" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_results():
    """Load diarisation results"""
    try:
        df = pd.read_csv(RESULTS_FILE)
        print(f"Loaded {len(df)} diarisation results")
        return df
    except FileNotFoundError:
        print(f"ERROR: {RESULTS_FILE} not found. Run run_diarisation.py first.")
        sys.exit(1)

def analyze_agreement(df):
    """Analyze agreement between models"""
    
    print("\n" + "=" * 80)
    print("INTER-MODEL AGREEMENT ANALYSIS")
    print("=" * 80)
    
    models = df['diarisation_model'].unique()
    
    if len(models) < 2:
        print("Need at least 2 models for agreement analysis")
        return
    
    # For each segment, compare speaker assignments
    segments = df['segment_filename'].unique()
    
    agreement_matrix = defaultdict(lambda: defaultdict(int))
    
    for segment in segments:
        segment_df = df[df['segment_filename'] == segment]
        
        # Get speaker assignments from each model
        assignments = {}
        for _, row in segment_df.iterrows():
            assignments[row['diarisation_model']] = row['speaker_id']
        
        # Compare all pairs
        model_list = list(assignments.keys())
        for i in range(len(model_list)):
            for j in range(i+1, len(model_list)):
                m1, m2 = model_list[i], model_list[j]
                if assignments[m1] == assignments[m2]:
                    agreement_matrix[m1][m2] += 1
                    agreement_matrix[m2][m1] += 1
    
    # Print agreement
    print("\nPairwise Agreement (segments with same speaker label):")
    total_segments = len(segments)
    
    for m1 in models:
        for m2 in models:
            if m1 < m2:  # Only print each pair once
                agree_count = agreement_matrix[m1][m2]
                agree_pct = (agree_count / total_segments) * 100 if total_segments > 0 else 0
                print(f"  {m1} vs {m2}: {agree_count}/{total_segments} ({agree_pct:.1f}%)")

def create_visualizations(df):
    """Create comparison visualizations"""
    
    print("\nGenerating visualizations...")
    
    sns.set_style("whitegrid")
    
    # 1. Speaker distribution comparison
    fig, axes = plt.subplots(1, len(df['diarisation_model'].unique()), figsize=(15, 5))
    
    if len(df['diarisation_model'].unique()) == 1:
        axes = [axes]
    
    for idx, model in enumerate(df['diarisation_model'].unique()):
        model_df = df[df['diarisation_model'] == model]
        speaker_counts = model_df['speaker_id'].value_counts()
        
        ax = axes[idx]
        speaker_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'{model}\nSpeaker Distribution')
        ax.set_xlabel('Speaker')
        ax.set_ylabel('Number of Segments')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = REPORT_DIR / "speaker_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # 2. Confidence distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in df['diarisation_model'].unique():
        model_df = df[df['diarisation_model'] == model]
        ax.hist(model_df['confidence'], bins=20, alpha=0.6, label=model)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Score Distribution by Model')
    ax.legend()
    
    output_file = REPORT_DIR / "confidence_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # 3. Speaker timeline for each model
    for model in df['diarisation_model'].unique():
        model_df = df[df['diarisation_model'] == model].sort_values('start_time_ms')
        
        fig, ax = plt.subplots(figsize=(15, 4))
        
        # Color map for speakers
        speakers = model_df['speaker_id'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
        speaker_colors = dict(zip(speakers, colors))
        
        for idx, row in model_df.iterrows():
            start_s = row['start_time_ms'] / 1000
            duration_s = row['duration_ms'] / 1000
            speaker = row['speaker_id']
            
            ax.barh(
                0, duration_s, left=start_s,
                height=0.8,
                color=speaker_colors[speaker],
                alpha=0.8,
                label=speaker if speaker not in ax.get_legend_handles_labels()[1] else ""
            )
        
        ax.set_xlabel('Time (seconds)')
        ax.set_yticks([])
        ax.set_title(f'Speaker Timeline - {model}')
        ax.legend(loc='upper right')
        ax.grid(axis='x', alpha=0.3)
        
        output_file = REPORT_DIR / f"timeline_{model}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()

def generate_report(df):
    """Generate HTML report"""
    
    print("\nGenerating HTML report...")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diarisation Comparison Report</title>
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
        <h1>Speaker Diarisation Comparison Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Summary</h2>
            <p>Total segments processed: <span class="metric">{len(df)}</span></p>
            <p>Models tested: <span class="metric">{df['diarisation_model'].nunique()}</span></p>
        </div>
        
        <div class="section">
            <h2>Model Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Segments</th>
                    <th>Unique Speakers</th>
                    <th>Avg Confidence</th>
                    <th>Processing Time (s)</th>
                </tr>
    """
    
    for model in df['diarisation_model'].unique():
        model_df = df[df['diarisation_model'] == model]
        html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{len(model_df)}</td>
                    <td>{model_df['speaker_id'].nunique()}</td>
                    <td>{model_df['confidence'].mean():.3f}</td>
                    <td>{model_df['processing_time_s'].iloc[0]:.2f}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Speaker Distribution</h2>
            <img src="speaker_distribution.png" alt="Speaker Distribution">
        </div>
        
        <div class="section">
            <h2>Confidence Scores</h2>
            <img src="confidence_distribution.png" alt="Confidence Distribution">
        </div>
    """
    
    # Add timelines
    for model in df['diarisation_model'].unique():
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
    
    output_file = REPORT_DIR / "diarisation_report.html"
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"  ✓ Saved report: {output_file}")

def main():
    """Main comparison function"""
    
    print("=" * 80)
    print("DIARISATION MODEL COMPARISON")
    print("=" * 80)
    
    # Load results
    df = load_results()
    if df is None:
        return
    
    # Print statistics
    print("\nModel Statistics:")
    for model in df['diarisation_model'].unique():
        model_df = df[df['diarisation_model'] == model]
        print(f"\n{model}:")
        print(f"  Segments: {len(model_df)}")
        print(f"  Unique speakers: {model_df['speaker_id'].nunique()}")
        print(f"  Speaker distribution:")
        for speaker, count in model_df['speaker_id'].value_counts().items():
            pct = (count / len(model_df)) * 100
            print(f"    {speaker}: {count} ({pct:.1f}%)")
        print(f"  Average confidence: {model_df['confidence'].mean():.3f}")
        print(f"  Processing time: {model_df['processing_time_s'].iloc[0]:.2f}s")
    
    # Analyze agreement
    analyze_agreement(df)
    
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
