@echo off
if not exist "input" mkdir "input"
if not exist "output" mkdir "output"
if not exist "prompts" mkdir "prompts"
if not exist "website" mkdir "website"

if not exist "verification_1" mkdir "verification_1"
if not exist "verification_2" mkdir "verification_2"
if not exist "verification_3" mkdir "verification_3"
if not exist "verification_4" mkdir "verification_4"

if not exist "diarisation\input" mkdir "diarisation\input"
if not exist "diarisation\speaker-diarization-3.1" mkdir "diarisation\speaker-diarization-3.1"
if not exist "diarisation\speaker-diarization-community-1" mkdir "diarisation\speaker-diarization-community-1"

if not exist "segmentation\input" mkdir "segmentation\input"
if not exist "segmentation\segmentation-3.0" mkdir "segmentation\segmentation-3.0"

if not exist "ipa\input" mkdir "ipa\input"
if not exist "ipa\allosaurus" mkdir "ipa\allosaurus"
if not exist "ipa\g2p-multilingual-byt5-tiny-8l-ipa-childes" mkdir "ipa\g2p-multilingual-byt5-tiny-8l-ipa-childes"
if not exist "ipa\ipa-align-base-phone" mkdir "ipa\ipa-align-base-phone"
if not exist "ipa\ipa-whisper-small" mkdir "ipa\ipa-whisper-small"
if not exist "ipa\mms" mkdir "ipa\mms"
if not exist "ipa\voxtral-mini-4b-realtime-2602" mkdir "ipa\voxtral-mini-4b-realtime-2602"
if not exist "ipa\xphonebert-base" mkdir "ipa\xphonebert-base"

if not exist "tone-correction\input" mkdir "tone-correction\input"
if not exist "tone-correction\finbert-tone-chinese" mkdir "tone-correction\finbert-tone-chinese"
pause