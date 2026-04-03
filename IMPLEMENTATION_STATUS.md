# IPA Transcriber 4.0 - Implementation Status

## Current Status: IN PROGRESS

**Last Updated**: 2026-04-03

## Completed Stages ✅

### 1. Environment Setup ✅
- Created conda `environment.yml` with all dependencies
- Created `requirements.txt` for pip users
- Set up `.gitignore` for audio/model files
- Created directory structure with `.gitkeep` files
- **GitHub Repository**: https://github.com/juanpalm-pixel/ipa-transcriber-04

### 2. Segmentation Stage ✅
**Files Created**:
- `segmentation/run_segmentation.py` - Main segmentation script
- `segmentation/compare_models.py` - Model comparison analysis

**Models Implemented**:
1. **Silero VAD** - Voice Activity Detection based segmentation
2. **Whisper-base** - Word-level timestamps from transcription
3. **Simple VAD** - Energy-based segmentation

**Output Format**:
- Audio clips: `{start_ms}_{end_ms}.wav`
- CSV: `segmentation_results.csv` with columns:
  - filename, start_time_ms, end_time_ms, duration_ms
  - model_name, processing_time_s, full_path

**Features**:
- Parallel testing of all models
- Statistical comparison (duration distribution, processing time)
- Visualization with matplotlib
- HTML reports

### 3. Verification 1 Tool ✅
**File Created**: `verification_1/review_tool.py`

**Features**:
- Interactive GUI with tkinter
- Model selection dropdown
- Waveform visualization
- Audio playback with pygame
- Manual segment labeling (good/too_short/too_long/bad_quality)
- JSON export of corrections
- Text report generation

### 4. Diarisation Stage ✅
**Files Created**:
- `diarisation/run_diarisation.py` - Main diarisation script
- `diarisation/compare_models.py` - Model comparison analysis

**Models Implemented**:
1. **PyAnnote Speaker-Diarization-3.1** - State-of-the-art diarisation
2. **Simple Pitch** - F0-based speaker classification (female vs male)
3. **Energy-Spectral** - Energy and spectral centroid based classification

**Output Format**:
- CSV: `diarisation_results.csv` with columns:
  - segment_filename, start_time_ms, end_time_ms, duration_ms
  - segmentation_model, diarisation_model
  - speaker_id, confidence, audio_path
  - processing_time_s, extra features

**Features**:
- Processes segmented audio clips
- Speaker distribution analysis
- Inter-model agreement metrics
- Confidence score tracking
- Timeline visualizations

---

## Remaining Stages 🔧

### 5. Verification 2 Tool [NEXT]
**To Create**: `verification_2/review_tool.py`

**Required Features**:
- Load diarisation results CSV
- Group segments by speaker for review
- Play audio clips by speaker
- Speaker distribution visualization
- Manual speaker reassignment
- Inter-model agreement display
- Export corrected labels

**Template**: Similar to verification_1 but focused on speaker review

---

### 6. IPA Transcription Stage [CRITICAL]
**To Create**: `ipa/run_ipa.py`

**Models to Test** (from HF collection):
1. neurlang/ipa-whisper-small
2. neurlang/ipa-whisper-base
3. anyspeech/ipa-align-base-phone
4. fdemelo/g2p-multilingual-byt5-tiny-8l-ipa-childes
5. vinai/xphonebert-base
6. openai/whisper-large-v3
7. Allosaurus (already in `ipa/allosaurus/`)
8. MMS models (already in `ipa/mms/`)
9. mistralai/Voxtral-Mini-4B-Realtime-2602 (already in `ipa/voxtral-mini-4b-realtime-2602/`)

**Output Format**:
- Master CSV: `ipa_transcriptions.csv` with columns:
  - segment_filename, speaker_id
  - model_name, ipa_transcription, confidence_score
  - start_time_ms, end_time_ms, duration_ms
  - processing_time_s

**Key Challenges**:
- Zero-shot transcription (no Rengmitca training data)
- Model-specific output formatting
- Handling long audio vs short clips
- Confidence estimation

**Approach**:
- Load diarised segments
- Run each segment through ALL IPA models
- Store all results for comparison
- Calculate inter-model agreement

---

### 7. Verification 3 Tool
**To Create**: `verification_3/review_tool.py`

**Required Features**:
- Side-by-side IPA transcription comparison (all models)
- Audio playback with synchronized IPA display
- Model agreement highlighting (green=agree, red=disagree)
- Manual IPA correction interface
- Export corrected transcriptions
- Model consistency metrics

---

### 8. Tone Detection Stage
**To Create**: `tone-correction/run_tone.py`

**Models/Methods**:
1. Librosa pitch tracking (F0 contours)
2. Parselmouth/PRAAT pitch analysis
3. yiyanghkust/finbert-tone-chinese (adapted for pitch)
4. Custom tone clustering

**Output Format**:
- CSV: `tone_analysis.csv` with columns:
  - segment_filename, speaker_id, ipa_transcription
  - detected_tones (tone categories)
  - pitch_contour_values (F0 array)
  - tone_category, model_name
- Aggregate tone inventory: `tone_inventory.json`

**Analysis Goals**:
- Identify if Rengmitca has lexical tones
- Propose tone categories (e.g., H, L, HL, LH, M)
- Calculate tone distribution
- Compare with known tonal languages

---

### 9. Verification 4 Tool
**To Create**: `verification_4/review_tool.py`

**Required Features**:
- Pitch contour visualization (matplotlib/plotly)
- Audio playback with overlaid F0 tracking
- Tone clustering display (scatter plot)
- Manual tone classification
- Tone inventory proposal
- Statistical analysis

---

### 10. Model Comparison & Documentation
**To Create**:
- `MODELS.md` - Detailed model comparison
- `PIPELINE.md` - Architecture documentation
- `RESULTS.md` - Testing results and findings

**Metrics to Include**:
- WER (where applicable with Bengali/Mru references)
- Processing time per model
- Memory usage
- Confidence scores
- Inter-model agreement
- Qualitative assessment

---

### 11. Full Pipeline Testing
**Steps**:
1. Run complete pipeline on `input/1.wav` (5-minute test)
2. Validate outputs at each stage
3. Check CSV formats
4. Verify verification tools work
5. Generate final consolidated report

---

### 12. GitHub Integration
**Ongoing**: Commit after each stage completion with descriptive messages

---

### 13. Final Validation
**Checklist**:
- [ ] All models tested
- [ ] All verification tools functional
- [ ] Documentation complete
- [ ] WER calculated (where possible)
- [ ] Tone inventory proposed
- [ ] Best models identified
- [ ] Pipeline ready for 40-minute blocks

---

## Quick Start Guide

### Testing Current Implementation

1. **Run Segmentation**:
   ```bash
   cd segmentation
   python run_segmentation.py
   python compare_models.py
   ```

2. **Review Segments**:
   ```bash
   cd ../verification_1
   python review_tool.py
   ```

3. **Run Diarisation**:
   ```bash
   cd ../diarisation
   python run_diarisation.py
   python compare_models.py
   ```

4. **Review Diarisation** (pending):
   ```bash
   cd ../verification_2
   python review_tool.py  # To be created
   ```

### Next Steps for Implementation

1. **Create Verification 2 Tool** - Copy structure from verification_1
2. **Implement IPA Stage** - Most critical, test all 9 models
3. **Create Verification 3 Tool** - IPA comparison interface
4. **Implement Tone Stage** - Pitch analysis
5. **Create Verification 4 Tool** - Tone visualization
6. **Generate Documentation** - MODELS.md, PIPELINE.md, RESULTS.md

---

## Technical Notes

### HuggingFace Models Access
- Token stored in environment: `os.getenv('HF_TOKEN')`
- Models automatically cached in `.cache/`
- Some models require authentication

### Audio Format
- Sample rate: 16kHz (standardized)
- Format: WAV, mono
- Segmented clips named by timestamps

### Data Flow
```
input/1.wav
  → segmentation/output/{model}/*.wav
  → diarisation_results.csv (speaker labels)
  → ipa_transcriptions.csv (IPA text)
  → tone_analysis.csv (tone labels)
  → output/final_results/
```

### Verification Workflow
Each stage has a verification tool that:
1. Loads previous stage results
2. Allows manual review
3. Exports corrections
4. Generates reports

---

## Known Issues

1. **Pyannote models**: May require HF authentication
2. **Male speaker distinction**: Models struggle to distinguish between two male speakers (expected)
3. **Zero-shot IPA**: No Rengmitca training data, relying on model generalization
4. **Tone detection**: Challenging without labeled tone examples

---

## Contact & Repository

- **GitHub**: https://github.com/juanpalm-pixel/ipa-transcriber-04
- **Local Path**: `C:\Users\pablo\OneDrive\Desktop\Functions\ipa-transcribers\ipa-transcriber(4.0)`
- **HF Collection**: https://huggingface.co/collections/juanpalm/ipa-transcriber

---

## Progress Summary

**Completed**: 5/15 todos (33%)

**Core Pipeline**:
- ✅ Segmentation
- ✅ Diarisation
- ⏳ IPA Transcription (NEXT PRIORITY)
- ⏳ Tone Detection

**Verification Tools**:
- ✅ Tool 1 (Segmentation)
- ⏳ Tool 2 (Diarisation)
- ⏳ Tool 3 (IPA)
- ⏳ Tool 4 (Tone)

**Documentation**: Pending comprehensive model comparison and results

---

*Last commit: c829f2f - "Add diarisation stage implementation"*
