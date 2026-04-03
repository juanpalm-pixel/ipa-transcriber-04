"""
Compare IPA transcription results across models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import difflib

# Configuration
RESULTS_FILE = "ipa_transcriptions.csv"
REPORT_DIR = Path("../verification_3/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_results():
    """Load IPA transcription results"""
    try:
        df = pd.read_csv(RESULTS_FILE)
        print(f"Loaded {len(df)} IPA transcriptions")
        return df
    except FileNotFoundError:
        print(f"ERROR: {RESULTS_FILE} not found. Run run_ipa.py first.")
        return None

def calculate_agreement(df):
    """Calculate inter-model agreement"""
    
    print("\n" + "=" * 80)
    print("INTER-MODEL AGREEMENT ANALYSIS")
    print("=" * 80)
    
    models = df['ipa_model'].unique()
    
    if len(models) < 2:
        print("Need at least 2 models for agreement analysis")
        return
    
    # For each segment, compare transcriptions
    segments = df['segment_filename'].unique()
    
    agreement_scores = defaultdict(lambda: defaultdict(list))
    
    for segment in segments:
        segment_df = df[df['segment_filename'] == segment]
        
        # Get transcriptions from each model
        transcriptions = {}
        for _, row in segment_df.iterrows():
            transcriptions[row['ipa_model']] = row['ipa_transcription']
        
        # Compare all pairs
        model_list = list(transcriptions.keys())
        for i in range(len(model_list)):
            for j in range(i+1, len(model_list)):
                m1, m2 = model_list[i], model_list[j]
                t1, t2 = transcriptions[m1], transcriptions[m2]
                
                # Calculate similarity using SequenceMatcher
                similarity = difflib.SequenceMatcher(None, t1, t2).ratio()
                agreement_scores[m1][m2].append(similarity)
                agreement_scores[m2][m1].append(similarity)
    
    # Print agreement
    print("\nPairwise Agreement (average similarity):")
    
    for m1 in models:
        for m2 in models:
            if m1 < m2 and m1 in agreement_scores and m2 in agreement_scores[m1]:
                scores = agreement_scores[m1][m2]
                avg_similarity = np.mean(scores) * 100
                print(f"  {m1} vs {m2}: {avg_similarity:.1f}% similarity")

def analyze_by_speaker(df):
    """Analyze transcription quality by speaker"""
    
    print("\n" + "=" * 80)
    print("ANALYSIS BY SPEAKER")
    print("=" * 80)
    
    for speaker in df['speaker_id'].unique():
        speaker_df = df[df['speaker_id'] == speaker]
        
        print(f"\n{speaker}:")
        for model in speaker_df['ipa_model'].unique():
            model_df = speaker_df[speaker_df['ipa_model'] == model]
            non_empty = len(model_df[model_df['ipa_transcription'].str.len() > 0])
            avg_length = model_df['ipa_transcription'].str.len().mean()
            
            print(f"  {model}:")
            print(f"    Segments: {len(model_df)}")
            print(f"    Non-empty: {non_empty} ({(non_empty/len(model_df)*100):.1f}%)")
            print(f"    Avg length: {avg_length:.1f} characters")

def generate_report(df):
    """Generate HTML report"""
    
    print("\nGenerating HTML report...")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>IPA Transcription Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 12px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
            .section {{ margin: 30px 0; }}
            .ipa {{ font-family: 'Doulos SIL', 'Charis SIL', 'Gentium', monospace; }}
        </style>
    </head>
    <body>
        <h1>IPA Transcription Comparison Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Summary</h2>
            <p>Total segments: <span class="metric">{df['segment_filename'].nunique()}</span></p>
            <p>Models tested: <span class="metric">{df['ipa_model'].nunique()}</span></p>
            <p>Total transcriptions: <span class="metric">{len(df)}</span></p>
        </div>
        
        <div class="section">
            <h2>Model Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Segments</th>
                    <th>Non-Empty</th>
                    <th>Success Rate</th>
                    <th>Avg Length</th>
                    <th>Processing Time (s)</th>
                </tr>
    """
    
    for model in df['ipa_model'].unique():
        model_df = df[df['ipa_model'] == model]
        non_empty = len(model_df[model_df['ipa_transcription'].str.len() > 0])
        success_rate = (non_empty / len(model_df)) * 100
        avg_length = model_df['ipa_transcription'].str.len().mean()
        proc_time = model_df['processing_time_s'].iloc[0]
        
        html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{len(model_df)}</td>
                    <td>{non_empty}</td>
                    <td>{success_rate:.1f}%</td>
                    <td>{avg_length:.1f} chars</td>
                    <td>{proc_time:.2f}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Sample Transcriptions</h2>
            <table>
                <tr>
                    <th>Segment</th>
                    <th>Speaker</th>
                    <th>Model</th>
                    <th>IPA Transcription</th>
                </tr>
    """
    
    # Show first 20 transcriptions
    for _, row in df.head(20).iterrows():
        html += f"""
                <tr>
                    <td>{row['segment_filename']}</td>
                    <td>{row['speaker_id']}</td>
                    <td>{row['ipa_model']}</td>
                    <td class="ipa">{row['ipa_transcription']}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    output_file = REPORT_DIR / "ipa_comparison.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  ✓ Saved report: {output_file}")

def main():
    """Main comparison function"""
    
    print("=" * 80)
    print("IPA TRANSCRIPTION MODEL COMPARISON")
    print("=" * 80)
    
    # Load results
    df = load_results()
    if df is None:
        return
    
    # Print statistics
    print("\nModel Statistics:")
    for model in df['ipa_model'].unique():
        model_df = df[df['ipa_model'] == model]
        non_empty = len(model_df[model_df['ipa_transcription'].str.len() > 0])
        print(f"\n{model}:")
        print(f"  Segments: {len(model_df)}")
        print(f"  Non-empty transcriptions: {non_empty} ({(non_empty/len(model_df)*100):.1f}%)")
        print(f"  Average transcription length: {model_df['ipa_transcription'].str.len().mean():.1f} chars")
        print(f"  Processing time: {model_df['processing_time_s'].iloc[0]:.2f}s")
    
    # Analyze agreement
    calculate_agreement(df)
    
    # Analyze by speaker
    analyze_by_speaker(df)
    
    # Generate report
    generate_report(df)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print(f"Reports saved to: {REPORT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
