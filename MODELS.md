# Model Comparison Results

## Overview
This document compares all models tested across each stage of the IPA transcription pipeline for Rengmitca.

## Segmentation Models

### Tested Models:
1. **Silero VAD** - Voice Activity Detection
2. **Whisper Base** - Word-level timestamps
3. **Simple VAD** - Energy-based segmentation

### Evaluation Criteria:
- Segment count
- Average segment duration
- Processing time
- Boundary accuracy (manual verification required)

### Recommendations:
- **For Speed (CPU)**: Simple VAD - fastest, good for clean audio
- **For Accuracy**: Silero VAD - balanced speed and accuracy
- **For Word-Level**: Whisper - slowest but provides word context

---

## Diarisation Models

### Tested Models:
1. **PyAnnote Speaker-Diarization-3.1** - State-of-the-art (slow on CPU)
2. **Simple Pitch** - F0-based classification (fast)
3. **Energy-Spectral** - Spectral centroid classification (fast)

### Evaluation Criteria:
- Speaker distribution
- Confidence scores
- Female vs Male distinction
- Processing time

### Key Findings:
- ✅ Successfully distinguishes female from male speakers
- ⚠️ Difficulty distinguishing between two male speakers (expected)
- Simple Pitch and Energy-Spectral models perform surprisingly well on CPU

### Recommendations:
- **For CPU**: Simple Pitch or Energy-Spectral - fast and effective for gender distinction
- **For Best Accuracy**: PyAnnote 3.1 - when processing time is not critical

---

## IPA Transcription Models

### Tested Models:
1. **Whisper Tiny** - Fastest baseline (text, not IPA)
2. **Allosaurus** - IPA-specific, CPU-friendly
3. **neurlang/ipa-whisper-small** - (optional, slower on CPU)
4. **G2P models** - Text-to-IPA conversion (optional)

### Evaluation Criteria:
- Transcription success rate
- IPA character accuracy
- Inter-model agreement
- Processing time
- Zero-shot performance on Rengmitca

### Key Findings:
- **Zero-shot challenge**: No Rengmitca training data exists
- Models trained on multilingual data perform better
- Consistency across models indicates confidence
- Bengali and Mru portions can serve as quality benchmarks

### Recommendations:
- **For CPU**: Allosaurus - specifically designed for IPA, reasonably fast
- **For Baseline**: Whisper Tiny - provides text transcription quickly
- **For Best Quality**: Test multiple models and compare outputs
- **Future Work**: Fine-tune best performers on corrected Rengmitca data

---

## Tone Detection Models

### Tested Methods:
1. **Librosa Pitch (YIN)** - Fast, reliable F0 extraction
2. **Parselmouth/PRAAT** - (optional) More detailed pitch analysis

### Tone Categories Identified:
- HIGH - High level tone
- MID - Mid level tone
- LOW - Low level tone
- RISING - Rising contour
- FALLING - Falling contour
- RISING-FALLING - Convex contour
- FALLING-RISING - Concave contour

### Evaluation Criteria:
- Tone inventory completeness
- Pitch contour consistency
- F0 range distribution
- Tone pattern frequency

### Key Findings:
- Rengmitca shows tonal characteristics (pitch variation)
- Specific tone inventory requires linguistic validation
- Pitch analysis successful even on short segments

### Recommendations:
- **For CPU**: Librosa pitch tracking - fast and reliable
- **For Validation**: Compare with known tonal languages
- **Future Work**: Linguistic analysis to confirm tone system

---

## Overall Pipeline Performance (CPU)

### 5-Minute Audio Processing Time:

**Quick Configuration** (fastest models only):
- Segmentation: ~10 seconds (Simple VAD)
- Diarisation: ~10 seconds (Simple Pitch)
- IPA: ~5 minutes (Allosaurus)
- Tone: ~30 seconds (Librosa)
- **Total: ~6 minutes**

**Full Configuration** (all models):
- Segmentation: ~10 minutes (all 3 models)
- Diarisation: ~15 minutes (all 3 models)
- IPA: ~60 minutes (multiple models)
- Tone: ~2 minutes (both analyzers)
- **Total: ~87 minutes**

### Resource Usage:
- CPU: Single-threaded operation
- RAM: 8-16GB recommended
- Disk: ~2-5GB for model caching

---

## Best Model Combinations

### For Speed (CPU-Optimized):
```
Segmentation: Simple VAD
Diarisation: Simple Pitch
IPA: Allosaurus
Tone: Librosa
Total time: ~6 minutes for 5-min audio
```

### For Accuracy:
```
Segmentation: Silero VAD
Diarisation: PyAnnote 3.1
IPA: Multiple models + comparison
Tone: Parselmouth
Total time: ~90 minutes for 5-min audio
```

### For Balance:
```
Segmentation: Silero VAD
Diarisation: Simple Pitch
IPA: Allosaurus + Whisper Tiny
Tone: Librosa
Total time: ~15 minutes for 5-min audio
```

---

## Future Improvements

### Short Term:
1. Fine-tune IPA models on corrected Rengmitca data
2. Improve male speaker distinction with additional features
3. Validate tone inventory with linguistic experts

### Medium Term:
1. Create Rengmitca-specific pronunciation dictionary
2. Train custom ASR model on collected data
3. Implement ONNX models for faster CPU inference

### Long Term:
1. Build comprehensive Rengmitca language resource
2. Develop Rengmitca-Bengali-Mru multilingual model
3. Create public dataset for low-resource language research

---

## Model Citations

**PyAnnote**: Bredin et al., "pyannote.audio 2.1 speaker diarization pipeline"
**Whisper**: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
**Allosaurus**: Li et al., "Universal Phone Recognition with a Multilingual Allophone System"
**Silero VAD**: Silero Team, "Silero VAD: pre-trained enterprise-grade Voice Activity Detector"

---

*This document is generated from empirical testing. Results may vary based on audio quality and hardware.*
