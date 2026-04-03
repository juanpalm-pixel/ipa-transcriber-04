# Pipeline Architecture

## Overview
The IPA Transcriber pipeline processes multi-speaker audio recordings to produce phonetic (IPA) transcriptions with tone annotations. The pipeline is designed for zero-shot transcription of low-resource languages.

## Pipeline Stages

```
┌──────────────┐
│  Input Audio │
│   (5-40min)  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  1. SEGMENTATION │  Split into word-level clips
│  (VAD-based)     │  Output: {start_ms}_{end_ms}.wav
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  VERIFICATION 1  │  Manual review of segments
│  (Interactive)   │  Adjust boundaries, filter noise
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  2. DIARISATION  │  Assign speaker IDs
│  (Pitch/ML)      │  Output: speaker labels + confidence
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  VERIFICATION 2  │  Review speaker assignments
│  (Interactive)   │  Reassign speakers, group by voice
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  3. IPA TRANS.   │  Generate IPA transcriptions
│  (Multiple ASR)  │  Output: phonetic text
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  VERIFICATION 3  │  Compare model outputs
│  (Interactive)   │  Manual corrections, select best
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  4. TONE DETECT. │  Analyze pitch contours
│  (F0 Analysis)   │  Output: tone categories
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  VERIFICATION 4  │  Review tone assignments
│  (Interactive)   │  Visualize pitch, classify tones
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Final Output    │  IPA + Tones + Metadata
│  (CSV/JSON)      │
└──────────────────┘
```

## Data Flow

### Stage 1: Segmentation
**Input**: `input/1.wav` (original audio)
**Process**:
- Voice Activity Detection
- Silence removal
- Word boundary detection
**Output**: 
- `segmentation/output/{model}/*.wav` - Audio clips
- `segmentation/segmentation_results.csv` - Metadata

**CSV Schema**:
```
filename, start_time_ms, end_time_ms, duration_ms, model_name, processing_time_s, full_path
```

### Stage 2: Diarisation
**Input**: Segmented audio clips from Stage 1
**Process**:
- Pitch extraction (F0)
- Spectral analysis
- Speaker classification
**Output**:
- `diarisation/diarisation_results.csv`

**CSV Schema**:
```
segment_filename, start_time_ms, end_time_ms, duration_ms,
segmentation_model, diarisation_model, speaker_id, confidence, audio_path, processing_time_s
```

### Stage 3: IPA Transcription
**Input**: Diarised segments from Stage 2
**Process**:
- Load audio segment
- Run through multiple IPA models
- Extract phonetic transcription
**Output**:
- `ipa/ipa_transcriptions.csv`

**CSV Schema**:
```
segment_filename, start_time_ms, end_time_ms, duration_ms, speaker_id,
diarisation_model, ipa_model, ipa_transcription, confidence, audio_path, processing_time_s
```

### Stage 4: Tone Detection
**Input**: IPA-transcribed segments from Stage 3
**Process**:
- F0 extraction
- Pitch contour analysis
- Tone classification
**Output**:
- `tone-correction/tone_analysis.csv`
- `tone-correction/output/tone_inventory.json`

**CSV Schema**:
```
segment_filename, speaker_id, start_time_ms, end_time_ms, duration_ms,
ipa_transcription, tone_model, detected_tones, tone_category,
mean_f0, f0_std, f0_range, audio_path
```

## Verification Tools

Each verification tool provides:
- **Audio playback**: Listen to segments
- **Visual display**: Waveforms, pitch contours, comparisons
- **Manual correction**: Override automatic classifications
- **Export functionality**: Save corrections for next stage
- **Reports**: Generate HTML/text summaries

### Usage Pattern:
```python
# 1. Run automatic stage
cd segmentation
python run_segmentation.py

# 2. Review and correct
cd ../verification_1
python review_tool.py

# 3. Proceed to next stage
cd ../diarisation
python run_diarisation.py
```

## Configuration

### Model Selection
Edit `run_*.py` files to enable/disable models:
```python
models = [
    FastModel(),    # Uncomment for quick tests
    # SlowModel(), # Comment out for faster processing
]
```

### CPU Optimization
All models configured for CPU-only:
```python
device = "cpu"
model.to(device)
```

### Parameters
Key parameters can be adjusted:

**Segmentation**:
- `min_speech_duration_ms`: Minimum segment length
- `min_silence_duration_ms`: Silence threshold

**Diarisation**:
- `num_speakers`: Expected speaker count
- `pitch_threshold`: F0 cutoff for gender classification

**IPA**:
- `model_size`: "tiny", "base", "small", "large"
- `language`: Language hint (or None for auto-detect)

**Tone**:
- `fmin`, `fmax`: F0 extraction range (75-400 Hz)
- `tone_categories`: Custom tone labels

## Error Handling

### Missing Files
- Pipeline checks for input at each stage
- Clear error messages indicate which stage to run first

### Failed Models
- Models that fail to load are skipped
- At least one model must succeed per stage
- Errors logged but don't stop pipeline

### Empty Results
- Segments with no transcription marked with empty string
- Confidence scores indicate quality
- Manual review highlights issues

## Performance Optimization

### For Faster Processing:
1. Use fewer models (1-2 per stage)
2. Reduce audio length (test on 1-minute clips)
3. Use lightweight models (Simple VAD, Allosaurus)
4. Skip optional verification steps

### For Better Accuracy:
1. Test all available models
2. Use verification tools for manual review
3. Run multiple iterations with corrections
4. Compare model agreement for validation

## Scaling to 40-Minute Audio

### Approach:
1. **Test on 5-minute** first
2. **Validate pipeline** works correctly
3. **Select best models** based on results
4. **Process in chunks** if memory constrained
5. **Use parallel processing** where possible

### Expected Time:
- 5-minute audio: ~6-90 minutes (depending on config)
- 40-minute audio: ~48-720 minutes (8x longer)

### Memory Management:
- Process segments individually
- Clear model cache between stages
- Save intermediate results frequently

## Future Extensions

### Batch Processing:
```python
for audio_file in input_files:
    run_pipeline(audio_file)
```

### Parallel Model Execution:
```python
from multiprocessing import Pool
with Pool(processes=2) as pool:
    results = pool.map(transcribe, segments)
```

### Web Interface:
- Upload audio files
- Monitor pipeline progress
- Download results

### API Endpoint:
```python
POST /transcribe
{
  "audio": "base64_encoded_audio",
  "config": {...}
}
```

---

## Directory Structure

```
ipa-transcriber(4.0)/
├── input/                   # Input audio files
├── segmentation/
│   ├── run_segmentation.py
│   ├── compare_models.py
│   ├── output/             # Segmented clips
│   └── segmentation_results.csv
├── verification_1/
│   ├── review_tool.py
│   └── reports/
├── diarisation/
│   ├── run_diarisation.py
│   ├── compare_models.py
│   └── diarisation_results.csv
├── verification_2/
│   ├── review_tool.py
│   └── reports/
├── ipa/
│   ├── run_ipa.py
│   ├── compare_models.py
│   └── ipa_transcriptions.csv
├── verification_3/
│   ├── review_tool.py
│   └── reports/
├── tone-correction/
│   ├── run_tone.py
│   ├── tone_analysis.csv
│   └── output/
│       └── tone_inventory.json
├── verification_4/
│   ├── review_tool.py
│   └── reports/
└── output/                  # Final results
```

---

*This pipeline is designed for CPU-only execution and optimized for low-resource languages with zero-shot capabilities.*
