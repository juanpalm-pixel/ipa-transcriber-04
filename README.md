# IPA Transcriber 4.0 - Zero-Shot ASR for Rengmitca

A comprehensive pipeline for phonetic transcription and analysis of Rengmitca (glottology code: reng1255), a low-resource language, using zero-shot ASR models.

## Project Overview

This project processes multi-speaker audio recordings containing Bengali (female speaker), Mru (male speaker_1), and Rengmitca (male speaker_2) to produce IPA (International Phonetic Alphabet) transcriptions and tone analysis.

### Pipeline Stages

```
Input Audio → Segmentation → Diarisation → IPA Transcription → Tone Detection → Output
                    ↓              ↓                ↓                   ↓
              Verification_1  Verification_2  Verification_3    Verification_4
```

1. **Segmentation**: Split audio into individual word-level segments
2. **Diarisation**: Assign speaker labels to each segment
3. **IPA Transcription**: Generate phonetic transcriptions using multiple models
4. **Tone Detection**: Analyze pitch contours and identify tone patterns

Each stage includes a verification step with interactive review tools.

## Quick Start

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- **CPU-only** (No GPU required)
- 8GB+ RAM recommended
- HuggingFace account with access token
- GitHub account with personal access token

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ipa-transcriber-04.git
   cd ipa-transcriber-04
   ```

2. **Set up environment variables**
   Create a `.env` file:
   ```bash
   HF_TOKEN=your_huggingface_token_here
   GH_TOKEN=your_github_token_here
   ```

3. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate ipa-transcriber-04
   ```

   Or using pip (CPU-only PyTorch):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

   **Note**: This project is configured for **CPU-only** execution. See `CPU_CONFIGURATION.md` for performance notes.

### Basic Usage

1. **Place your audio file**
   ```bash
   cp your_audio.wav input/1.wav
   ```

2. **Run the segmentation stage**
   ```bash
   cd segmentation
   python run_segmentation.py
   ```

3. **Review segmentation results**
   ```bash
   cd ../verification_1
   python review_tool.py
   ```

4. **Continue through the pipeline**
   ```bash
   cd ../diarisation
   python run_diarisation.py
   # ... and so on
   ```

## Project Structure

```
ipa-transcriber(4.0)/
├── environment.yml           # Conda environment specification
├── requirements.txt          # Pip requirements
├── .gitignore               # Git ignore patterns
├── README.md                # This file
├── MODELS.md                # Model comparison results
├── PIPELINE.md              # Detailed pipeline documentation
├── RESULTS.md               # Testing results and findings
│
├── input/                   # Input audio files
│   └── 1.wav               # Test audio (5 minutes)
│
├── segmentation/            # Stage 1: Audio segmentation
│   ├── run_segmentation.py
│   ├── compare_models.py
│   ├── segmentation_results.csv
│   ├── models/             # Downloaded models
│   └── output/             # Segmented audio clips
│
├── verification_1/          # Segmentation review tools
│   ├── review_tool.py
│   └── reports/
│
├── diarisation/             # Stage 2: Speaker diarisation
│   ├── run_diarisation.py
│   ├── compare_models.py
│   ├── diarisation_results.csv
│   ├── models/
│   └── output/
│
├── verification_2/          # Diarisation review tools
│   ├── review_tool.py
│   └── reports/
│
├── ipa/                     # Stage 3: IPA transcription
│   ├── run_ipa.py
│   ├── compare_models.py
│   ├── ipa_transcriptions.csv
│   ├── models/
│   └── output/
│
├── verification_3/          # IPA review tools
│   ├── review_tool.py
│   └── reports/
│
├── tone-correction/         # Stage 4: Tone detection
│   ├── run_tone.py
│   ├── compare_models.py
│   ├── tone_analysis.csv
│   ├── models/
│   └── output/
│
├── verification_4/          # Tone review tools
│   ├── review_tool.py
│   └── reports/
│
└── output/                  # Final consolidated outputs
    └── final_results/
```

## Models Tested

### Segmentation
- Microsoft VibeVoice-ASR
- Silero VAD
- Whisper-based segmentation

### Diarisation
- pyannote/speaker-diarization-3.1
- pyannote/speaker-diarization-community-1
- Microsoft VibeVoice-ASR

### IPA Transcription
- neurlang/ipa-whisper-small
- neurlang/ipa-whisper-base
- anyspeech/ipa-align-base-phone
- fdemelo/g2p-multilingual-byt5-tiny-8l-ipa-childes
- vinai/xphonebert-base
- openai/whisper-large-v3
- Allosaurus
- MMS models
- mistralai/Voxtral-Mini-4B-Realtime-2602

### Tone Detection
- Custom pitch extraction (librosa)
- Parselmouth/PRAAT analysis
- yiyanghkust/finbert-tone-chinese (adapted)

## Data Format

- **Input**: WAV audio files (16kHz recommended)
- **Segmented clips**: `{start_ms}_{end_ms}.wav`
- **Results**: CSV files with timestamps, labels, and confidence scores
- **Reports**: HTML/text files with visualizations and statistics

## Evaluation Metrics

- **Word Error Rate (WER)**: For transcription accuracy
- **Processing time**: Per model and per stage
- **Confidence scores**: Model-reported certainty
- **Inter-model agreement**: Consistency across models

## Current Status

- [x] Environment setup
- [ ] Segmentation implementation
- [ ] Verification tools (stage 1)
- [ ] Diarisation implementation
- [ ] Verification tools (stage 2)
- [ ] IPA transcription implementation
- [ ] Verification tools (stage 3)
- [ ] Tone detection implementation
- [ ] Verification tools (stage 4)
- [ ] Model comparison and documentation
- [ ] Full pipeline testing

## Verification Tools

Each verification stage provides:
- **Interactive audio playback**: Listen to segments with context
- **Visual reports**: Statistics, distributions, and comparisons
- **Manual correction**: Edit labels, timestamps, and transcriptions
- **Export functionality**: Save corrected data for next stage

## Future Work

- Fine-tune best-performing models on corrected data
- Expand to full 40-minute audio blocks
- Develop custom models for Rengmitca
- Create Rengmitca-specific phoneme inventory
- Build pronunciation dictionaries

## Contributing

This is a research project. Documentation is continuously updated as models are tested and results are analyzed.

## License

[Specify license here]

## Citation

If you use this work, please cite:
```
[Citation information to be added]
```

## Contact

[Your contact information]

## Acknowledgments

- HuggingFace for model hosting
- pyannote.audio for diarisation tools
- OpenAI for Whisper models
- All model creators listed in MODELS.md
