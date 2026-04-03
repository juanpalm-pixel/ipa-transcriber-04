# CPU-Only Configuration Notes

## Overview
This project is configured to run **entirely on CPU** - no GPU required.

## Key Changes for CPU Mode

### 1. Environment Configuration
- `environment.yml` uses `pytorch-cpu` and `torchaudio-cpu` packages
- No CUDA toolkit dependencies
- Optimized for CPU inference

### 2. Model Configuration
All models are explicitly set to CPU mode:

```python
device = "cpu"  # Hardcoded CPU mode
```

### 3. Performance Expectations

#### Segmentation (5-minute audio)
- **Silero VAD**: ~10-30 seconds (CPU optimized)
- **Whisper Base**: ~2-5 minutes (CPU intensive)
- **Simple VAD**: ~5-10 seconds (lightweight)

#### Diarisation (per segment)
- **PyAnnote**: ~1-3 seconds per segment (CPU)
- **Simple Pitch**: <1 second per segment
- **Energy-Spectral**: <1 second per segment

#### IPA Transcription (estimated)
- **Whisper-based**: ~2-5 minutes for 5-min audio
- **Transformer models**: ~3-10 minutes for 5-min audio
- **Allosaurus**: ~1-2 minutes for 5-min audio

#### Tone Detection
- **Librosa pitch**: ~10-30 seconds
- **Parselmouth**: ~30-60 seconds

### 4. Optimization Tips

#### For Faster CPU Processing:
1. **Use fewer models**: Test 1-2 models per stage instead of all
2. **Reduce audio length**: Start with 1-minute clips for testing
3. **Parallel processing**: Python multiprocessing for model comparison
4. **Batch size**: Set batch_size=1 for all models

#### Model Selection Priority (CPU-friendly):
- **Segmentation**: Simple VAD (fastest) or Silero VAD
- **Diarisation**: Simple Pitch (fastest) or Energy-Spectral
- **IPA**: Allosaurus or lightweight Whisper models
- **Tone**: Librosa pitch tracking (fastest)

### 5. Memory Considerations

Expected RAM usage:
- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM for running multiple models
- **Model caching**: ~2-5GB disk space for downloaded models

### 6. Multi-threading

PyTorch CPU optimizations:
```python
import torch
torch.set_num_threads(4)  # Adjust based on CPU cores
```

Current implementation uses single-threaded mode for stability.

### 7. Testing Strategy

For 5-minute audio on CPU:
1. **Quick test** (~5 minutes total):
   - Segmentation: Simple VAD only
   - Diarisation: Simple Pitch only
   - Skip IPA temporarily
   - Skip Tone temporarily

2. **Moderate test** (~30 minutes total):
   - Segmentation: Silero VAD + Simple VAD
   - Diarisation: Simple Pitch + Energy-Spectral
   - IPA: 1-2 fastest models (Allosaurus)
   - Tone: Librosa only

3. **Full test** (~2-3 hours total):
   - All segmentation models
   - All diarisation models
   - All IPA models
   - All tone detection methods

### 8. Known Limitations on CPU

1. **Whisper models**: Slow but accurate, consider smaller models (tiny, base vs large)
2. **PyAnnote**: Slower on CPU but still functional
3. **Transformer models**: Memory intensive, may need to process segments individually
4. **Large audio files**: 40-minute blocks will take significantly longer

### 9. Recommended Workflow

For initial testing on CPU:
```bash
# 1. Test with 1-minute clip first
# 2. Use fastest models only
# 3. Scale up gradually

# Segmentation only (fast)
cd segmentation
# Edit run_segmentation.py to use only SimplVADSegmenter
python run_segmentation.py

# Diarisation only (fast)
cd ../diarisation
# Edit run_diarisation.py to use only SimplePitchDiariser
python run_diarisation.py
```

### 10. Future Optimization Options

If processing is too slow:
1. **Use ONNX versions** of models (faster CPU inference)
2. **Quantization**: INT8 models for faster CPU
3. **Cloud GPU**: Use Google Colab or similar for heavy models
4. **Selective processing**: Only process Rengmitca segments (skip Bengali/Mru)

### 11. Installation Commands

For CPU-only setup:
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate ipa-transcriber-04

# Or using pip with CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 12. Verification

Check CPU mode is active:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be False
print(f"Device: {torch.device('cpu')}")
```

---

## Summary

✅ All code updated for CPU-only execution
✅ No GPU dependencies in environment.yml
✅ Explicit CPU device configuration in all scripts
✅ Performance expectations documented
✅ Optimization strategies provided

**Expected total time for 5-minute audio (all models, CPU)**:
- Segmentation: ~5-10 minutes
- Diarisation: ~5-15 minutes  
- IPA: ~30-60 minutes (varies by model)
- Tone: ~5-10 minutes

**Total: ~45-95 minutes for complete pipeline on CPU**
