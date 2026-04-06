# Changelog - IPA Transcriber 4.0

All notable changes to the Zero-Shot ASR Pipeline for Rengmitca will be documented in this file.

---

## Project Overview

**Goal**: Build a zero-shot ASR model for Rengmitca (glottology code: reng1255), a low-resource language, assuming no prior knowledge about the language.

**Data**: 4 × 40-minute audio blocks containing Bengali (female), Mru (male_1), and Rengmitca (male_2). Testing conducted on 5-minute clips initially.

**Pipeline**: `segmentation → diarisation → IPA transcription → tone detection` with interactive verification tools between each stage.

**Repository**: https://github.com/juanpalm-pixel/ipa-transcriber-04

---

## Version History

### [Unreleased] - 2026-04-06

#### Added
- New root launcher script `run_cycle.bat` to execute a full Windows command-line cycle in one command.
- Pipeline cycle order in launcher:
   - `segmentation/run_segmentation.py`
   - `segmentation/compare_models.py`
   - `diarisation/run_diarisation.py`
   - `diarisation/compare_models.py`
   - `segmentation/run_segmentation.py`
   - `segmentation/compare_models.py`
- Failure handling in `run_cycle.bat`: script stops immediately on first failed step and returns a non-zero exit code.

#### Changed
- `verification_2/review_tool.py`: trim-and-save now performs true in-place trimming of the current segment instead of reassigning speaker labels.
- `verification_2/review_tool.py`: trimming preserves speaker identity regardless of reassignment field content.
- `verification_2/review_tool.py`: trimmed output replaces the current CSV row, reopens immediately in the app, and removes the original audio file when unreferenced.
- `verification_2/review_tool.py`: added explicit `Delete File` button in the main control row to match existing hotkey support.
- `verification_2/review_tool.py`: added Escape-to-unfocus behavior for reassignment and trim entries, plus read-only info/stat text areas to prevent accidental edits while using hotkeys.

#### Current Status
- ✅ Segmentation stage complete with all models implemented
- ✅ Verification Tool 1 (segmentation review) complete
- ✅ Diarisation stage complete with all models implemented
- ✅ Verification Tool 2 core review workflow updated (trim/delete/focus improvements)
- ⏳ Pending: IPA transcription stage implementation

#### In Progress
- IPA stage iterative testing and model comparisons
- Reference: `ipa/run_ipa.py`, `ipa/compare_models.py`

---

## [1.0.0] - 2026-04-03

### Initial Implementation Complete ✅

#### Day 1-3: Project Foundation

**Phase 1: Environment & Setup**
- ✅ Created conda `environment.yml` with all dependencies
- ✅ Created `requirements.txt` for pip installation
- ✅ Set up `.gitignore` for audio/model files and generated outputs
- ✅ Established directory structure with `.gitkeep` files
- ✅ Created GitHub repository: `ipa-transcriber-04`

**Key Decision**: CPU-only configuration for accessibility
- Selected `pytorch-cpu` instead of GPU variants
- Documented in [CPU_CONFIGURATION.md](CPU_CONFIGURATION.md)
- All models configured with `device = "cpu"`
- Expected processing time: 6-90 minutes for 5-minute audio (depending on model selection)

---

#### Phase 2: Segmentation Stage ✅

**Status**: Fully implemented with 3 competing models

**Files Created**:
- `segmentation/run_segmentation.py` - Main segmentation pipeline
- `segmentation/compare_models.py` - Statistical comparison and visualization

**Models Implemented**:
1. **Silero VAD** - Voice Activity Detection based segmentation
   - Balanced speed/accuracy on CPU
   - ~30 seconds for 5-minute audio
   
2. **Whisper Base** - Word-level boundaries from transcription
   - Slowest but provides word context
   - ~5 minutes for 5-minute audio
   
3. **Simple VAD** - Energy-based segmentation
   - Fastest option
   - ~5-10 seconds for 5-minute audio

**Output Format**:
```
Audio Clips: segmentation/output/{model}/*.wav
  Naming: {start_ms}_{end_ms}.wav
  
CSV: segmentation/segmentation_results.csv
  Columns: filename, start_time_ms, end_time_ms, duration_ms, 
           model_name, processing_time_s, full_path
```

**Features**:
- Parallel testing of all 3 models simultaneously
- Statistical comparison (segment count, duration distribution, processing time)
- Matplotlib visualizations comparing models
- HTML report generation for model analysis

---

#### Phase 3: Verification Tool 1 ✅

**Status**: Interactive GUI complete and functional

**File**: `verification_1/review_tool.py`

**Features Implemented**:
- Model selection dropdown (choose segmentation model to review)
- Waveform visualization with matplotlib
- Audio playback support with pygame
- Manual segment quality labeling:
  - `good` - Acceptable segment
  - `too_short` - Segment needs to be longer
  - `too_long` - Should be split
  - `bad_quality` - Background noise/poor audio
  - `delete` - Remove this segment
  
- Navigation: Previous/Next segment buttons
- Progress tracking: Current position / Total count
- JSON export of corrections
- Text report generation

**Purpose**: Allow manual review and rejection of segments before proceeding to diarisation

---

#### Phase 4: Diarisation Stage ✅

**Status**: Fully implemented with 3 competing models

**Files Created**:
- `diarisation/run_diarisation.py` - Main diarisation pipeline
- `diarisation/compare_models.py` - Model comparison analysis

**Models Implemented**:
1. **PyAnnote Speaker-Diarization-3.1** - State-of-the-art (slow on CPU)
   - Best accuracy, ~5-10 minutes for 5-minute audio
   - Supports arbitrary number of speakers
   
2. **Simple Pitch** - F0-based gender classification (fast)
   - Extracts fundamental frequency
   - Classifies female vs male by pitch range
   - ~10 seconds for 5-minute audio
   
3. **Energy-Spectral** - Dual-feature speaker classification (fast)
   - Energy envelope analysis
   - Spectral centroid tracking
   - ~10 seconds for 5-minute audio

**Output Format**:
```
CSV: diarisation/diarisation_results.csv
  Columns: segment_filename, start_time_ms, end_time_ms, duration_ms, 
           segmentation_model, diarisation_model, speaker_id, confidence, 
           audio_path, processing_time_s
```

**Key Findings**:
- ✅ Successfully distinguishes female from male speakers
- ⚠️ Expected challenge: Cannot reliably distinguish between two male speakers (zero-shot limitation)
- Fast models (Simple Pitch, Energy-Spectral) surprisingly effective on CPU
- Confidence scores indicate model certainty

**Analysis Output**:
- Speaker distribution across models
- Inter-model agreement metrics
- Timeline visualizations
- Comparative statistics

---

#### Documentation ✅

**Files Created**:
1. **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Detailed status of all stages
   - Completed components with features
   - Remaining stages with requirements
   - Quick start guide
   - Technical notes

2. **[CPU_CONFIGURATION.md](CPU_CONFIGURATION.md)** - CPU-only setup guide
   - Performance expectations by model
   - Memory requirements (8-16GB recommended)
   - Optimization strategies
   - Installation instructions

3. **[CPU_UPDATES.md](CPU_UPDATES.md)** - Record of CPU-only changes
   - environment.yml modifications
   - Device configuration in all scripts
   - Expected processing times

4. **[PIPELINE.md](PIPELINE.md)** - Architecture documentation
   - Complete pipeline diagram (ASCII art)
   - Data flow between stages
   - CSV schemas for each stage
   - Configuration options
   - Error handling strategy
   - Scaling considerations to 40-minute audio

5. **[MODELS.md](MODELS.md)** - Model comparison and recommendations
   - Segmentation model evaluation
   - Diarisation model comparison
   - IPA transcription model selection
   - Tone detection methods
   - Best model combinations:
     - **For Speed**: Simple VAD → Simple Pitch → Allosaurus → Librosa (~6 min)
     - **For Accuracy**: Silero VAD → PyAnnote → Multiple IPA → Parselmouth (~90 min)
     - **For Balance**: Silero VAD → Simple Pitch → Allosaurus + Whisper → Librosa (~15 min)

6. **[RESULTS.md](RESULTS.md)** - Testing results and findings
   - Expected performance predictions
   - Known challenges and successes
   - Recommendations for next phases
   - Short/medium/long-term roadmap
   - Research questions to address

---

#### GitHub Integration ✅
- Repository initialized: `ipa-transcriber-04`
- All code committed with descriptive messages
- `.gitignore` configured for:
  - Audio files (`*.wav`)
  - Model caches (`.cache/`, `models/`)
  - Large outputs (`diarisation/output/`, `segmentation/output/`)
  - Environment files

---

### Architecture Decisions

#### 1. CPU-Only Processing
**Rationale**: 
- Accessibility for users without GPU
- Reproducibility across different hardware
- Cost effectiveness

**Trade-off**: Processing time 10-50x slower than GPU, but acceptable for research phase

#### 2. Multi-Model Comparison
**Rationale**:
- No pre-trained Rengmitca data exists
- Multiple models provide confidence through agreement
- Enables selection of best performer

**Approach**: All models run in parallel, results compared systematically

#### 3. Interactive Verification
**Rationale**:
- Low-resource language requires human-in-the-loop
- Catches systematic model errors early
- Builds ground truth dataset for fine-tuning

**Workflow**: 
```
Automatic Processing → Manual Review → Corrections Export → Next Stage
```

#### 4. CSV-Based Data Flow
**Rationale**:
- Human-readable format
- Easy to analyze with standard tools (pandas, Excel)
- Portable across verification tools
- Supports incremental updates

#### 5. Modular Pipeline Design
**Rationale**:
- Each stage independent and reusable
- Easy to select/replace models
- Enables parallel development
- Supports model comparison

---

### Key Technical Achievements

1. **Zero-Shot Framework**: Functional system for unknown languages
2. **CPU Optimization**: All components run efficiently on CPU
3. **Interactive Tools**: User-friendly verification interfaces
4. **Comprehensive Documentation**: Complete pipeline documentation
5. **Reproducible Research**: Version control and clear methodology

---

## Upcoming Milestones

### [NEXT] - 2026-04-06 (In Progress)

#### Verification Tool 2 - Speaker Review
- **Purpose**: Review and correct diarisation speaker assignments
- **Status**: 🔄 In Progress
- **Key Features**:
  - Load diarisation results CSV
  - Display segments grouped by speaker
  - Audio playback with speaker context
  - Speaker distribution visualization
  - Manual speaker reassignment
  - Corrections export (JSON + CSV)
  - Text report generation

#### Expected Completion
- Estimated: 2026-04-07
- After completion: Ready for IPA transcription stage

---

### Phase 5: IPA Transcription Stage [Planned]

**Target Completion**: 2026-04-08

**Models to Test** (9 models):
1. Whisper Tiny - Text baseline (fast)
2. Allosaurus - IPA-specific (CPU-friendly)
3. neurlang/ipa-whisper-small - IPA variant
4. anyspeech/ipa-align-base-phone - Phone alignment
5. fdemelo/g2p-multilingual-byt5-tiny-8l-ipa-childes - G2P conversion
6. vinai/xphonebert-base - Phone BERT
7. openai/whisper-large-v3 - Large Whisper (if feasible on CPU)
8. MMS models - Massively Multilingual Speech
9. voxtral-mini-4b-realtime-2602 - Realtime model

**Deliverables**:
- `ipa/run_ipa.py` - Main IPA pipeline
- `ipa/compare_models.py` - Model comparison
- `ipa_transcriptions.csv` - Master results

---

### Phase 6: Verification Tool 3 [Planned]

**Target Completion**: 2026-04-09

**Features**:
- Side-by-side IPA transcription comparison (all 9 models)
- Model agreement highlighting
- Manual IPA correction interface
- Confidence metrics
- Export corrected transcriptions

---

### Phase 7: Tone Detection Stage [Planned]

**Target Completion**: 2026-04-10

**Methods**:
1. Librosa pitch (YIN algorithm) - Fast, reliable
2. Parselmouth/PRAAT - Detailed pitch analysis

**Output**:
- Tone categories: HIGH, MID, LOW, RISING, FALLING, etc.
- Tone inventory proposal
- F0 statistics per segment

---

### Phase 8: Verification Tool 4 [Planned]

**Target Completion**: 2026-04-11

**Features**:
- Pitch contour visualization
- Tone category assignment interface
- Tone inventory clustering
- Statistical analysis

---

### Phase 9: Full Pipeline Testing [Planned]

**Target Completion**: 2026-04-12

**Checklist**:
- [ ] Run complete pipeline on 5-minute test audio
- [ ] Validate all CSV outputs
- [ ] Test all verification tools
- [ ] Generate comprehensive report
- [ ] Identify best model combinations

---

### Phase 10: Scale to Full 40-Minute Blocks [Planned]

**Target Completion**: 2026-04-15

**Approach**:
1. Select best models from 5-minute testing
2. Process first 40-minute block
3. Review results with faster verification
4. Extract Rengmitca-only data
5. Build ground truth dataset (100-200 segments)

---

## Summary of Implementation

| Component | Status | Date | Effort |
|-----------|--------|------|--------|
| Environment Setup | ✅ | 2026-04-03 | 1-2 hours |
| Segmentation Stage (3 models) | ✅ | 2026-04-03 | 4-5 hours |
| Verification Tool 1 | ✅ | 2026-04-03 | 3-4 hours |
| Diarisation Stage (3 models) | ✅ | 2026-04-03 | 4-5 hours |
| Documentation (5 files) | ✅ | 2026-04-03 | 6-8 hours |
| GitHub Setup | ✅ | 2026-04-03 | 1-2 hours |
| **Verification Tool 2** | 🔄 | 2026-04-06 | 3-4 hours (in progress) |
| IPA Transcription | ⏳ | 2026-04-08 | 6-8 hours (planned) |
| Verification Tool 3 | ⏳ | 2026-04-09 | 4-6 hours (planned) |
| Tone Detection | ⏳ | 2026-04-10 | 4-5 hours (planned) |
| Verification Tool 4 | ⏳ | 2026-04-11 | 3-4 hours (planned) |
| Full Pipeline Testing | ⏳ | 2026-04-12 | 2-3 hours (planned) |
| 40-Minute Processing | ⏳ | 2026-04-15 | 8-12 hours (planned) |

**Total Estimated Effort**: ~50-65 hours for complete pipeline

---

## Key Decisions & Rationale

### 1. CPU-Only Execution
- **Decision**: No GPU requirements
- **Rationale**: Accessibility, reproducibility, cost
- **Trade-off**: Processing time acceptable for research phase

### 2. Multi-Model Testing
- **Decision**: Test all available models in parallel
- **Rationale**: Zero-shot scenario requires exploration
- **Benefit**: Inter-model agreement provides confidence metric

### 3. Interactive Verification
- **Decision**: Tool after each automated stage
- **Rationale**: Low-resource language needs human oversight
- **Result**: High-quality ground truth dataset building

### 4. Modular Architecture
- **Decision**: Independent stages with CSV linkage
- **Rationale**: Flexibility, scalability, maintainability
- **Benefit**: Easy to replace/optimize individual components

### 5. Comprehensive Documentation
- **Decision**: 5 detailed markdown files + code comments
- **Rationale**: Reproducible research, collaboration
- **Result**: Clear understanding of methodology

---

## Known Challenges & Mitigation

| Challenge | Impact | Mitigation |
|-----------|--------|-----------|
| Zero-shot Rengmitca | Medium | Multiple models + inter-model comparison |
| Male speaker distinction | Low | Acceptable to merge similar speakers initially |
| CPU processing speed | Medium | Model selection for speed; 1-min test clips initially |
| No ground truth data | High | Interactive verification + manual annotation |
| Tone system validation | High | Expert linguistic review planned |

---

## Research Questions
1. How many distinct tones does Rengmitca have?
2. What is the complete phoneme inventory?
3. How does Rengmitca relate phonologically to Mru?
4. Are there dialectal variations in the recordings?
5. Which zero-shot models generalize best to Rengmitca?

---

## References

- **Original Prompt**: `prompts/prompt-01.txt`
- **Implementation Plan**: `prompts/plan-01.txt`
- **Pipeline Architecture**: [PIPELINE.md](PIPELINE.md)
- **Model Comparison**: [MODELS.md](MODELS.md)
- **Results & Findings**: [RESULTS.md](RESULTS.md)
- **Implementation Status**: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- **CPU Configuration**: [CPU_CONFIGURATION.md](CPU_CONFIGURATION.md)

---

## Contact & Repository

**GitHub Repository**: https://github.com/juanpalm-pixel/ipa-transcriber-04

**HuggingFace Collection**: https://huggingface.co/collections/juanpalm/ipa-transcriber

**Local Project Path**: `C:\Users\pablo\OneDrive\Desktop\Functions\ipa-transcribers\ipa-transcriber(4.0)`

---

**Changelog Last Updated**: 2026-04-06 | Status: In Progress
