# CPU-Only Updates Summary

## Changes Made

All code has been updated to run **exclusively on CPU** with no GPU dependencies.

### 1. Environment Configuration
**File**: `environment.yml`
- Changed: `pytorch>=2.0.0` → `pytorch-cpu>=2.0.0`
- Changed: `torchaudio>=2.0.0` → `torchaudio-cpu>=2.0.0`
- Removed: `cudatoolkit=11.8` (GPU dependency)

### 2. Segmentation Script
**File**: `segmentation/run_segmentation.py`
- Device configuration: `device = "cpu"` (hardcoded)
- Silero VAD: `.cpu()` explicitly called
- Whisper: `device="cpu"` parameter
- All torch tensors created without `.to(device)`

### 3. Diarisation Script
**File**: `diarisation/run_diarisation.py`
- Device configuration: `device = "cpu"` (hardcoded)
- PyAnnote pipeline: `.to(torch.device("cpu"))` explicitly called
- No conditional GPU checks

### 4. Documentation
**New Files**:
- `CPU_CONFIGURATION.md` - Comprehensive CPU performance guide
- Updated `README.md` - CPU-only installation instructions
- Updated plan.md - CPU constraint noted

## Installation Commands (CPU-Only)

### Option 1: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate ipa-transcriber-04
```

### Option 2: Pip
```bash
# Install CPU-only PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install other requirements
pip install -r requirements.txt
```

## Expected Performance (5-minute audio on CPU)

### Quick Estimate by Stage:
- **Segmentation**: 5-10 minutes
  - Simple VAD: ~5 sec
  - Silero VAD: ~30 sec
  - Whisper: ~5 min
  
- **Diarisation**: 5-15 minutes
  - Simple Pitch: ~10 sec
  - Energy-Spectral: ~10 sec
  - PyAnnote: ~5-10 min

- **IPA Transcription**: 30-60 minutes (varies greatly)
  - Allosaurus: ~5-10 min
  - Whisper models: ~10-30 min
  - Transformer models: ~20-60 min

- **Tone Detection**: 5-10 minutes
  - Librosa: ~30 sec
  - Parselmouth: ~5 min

### Total Pipeline Time:
**Estimated: 45-95 minutes** for complete pipeline with all models

### Optimization Tips:
1. **Test fewer models**: Choose 1-2 fastest per stage
2. **Use 1-minute clips**: For initial testing
3. **Sequential processing**: Run one stage at a time
4. **Lightweight models**: Prefer Allosaurus, Simple VAD, etc.

## Fastest CPU Configuration

For quickest results, use only these models:
```python
# Segmentation: Simple VAD only (~5 seconds)
# Diarisation: Simple Pitch only (~10 seconds)
# IPA: Allosaurus only (~5 minutes)
# Tone: Librosa only (~30 seconds)
# Total: ~6 minutes for 5-minute audio
```

## Memory Requirements
- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM (for running all models)
- **Disk space**: ~2-5GB for model caching

## Verification

Check CPU mode:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")  # Should print: False
print(f"Current device: cpu")
```

## Current Status

✅ All existing code updated for CPU
✅ Environment files configured for CPU-only
✅ Documentation updated
✅ Performance expectations documented
✅ Committed and pushed to GitHub

## Next Steps

When implementing remaining stages (IPA, Tone):
1. Always use `device = "cpu"`
2. Avoid `.cuda()` calls
3. Set `device="cpu"` in model.load_model() calls
4. Use `.cpu()` for any tensors
5. Consider CPU-optimized model variants when available

## Model Notes

### CPU-Friendly Models:
- ✅ Silero VAD (optimized for CPU)
- ✅ Allosaurus (relatively fast)
- ✅ Simple pitch/energy methods (no ML inference)
- ✅ Librosa-based tools (pure NumPy)

### CPU-Intensive Models:
- ⚠️ Whisper large (very slow on CPU)
- ⚠️ Large transformer models (slow)
- ⚠️ PyAnnote (functional but slow)

### Recommendations:
- Use Whisper "tiny" or "base" instead of "large"
- Test with 1-minute clips before full 5-minute
- Consider ONNX versions for faster CPU inference (future optimization)

---

**All code is now CPU-only compatible. No GPU will be used.**
