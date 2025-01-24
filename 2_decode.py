import pandas as pd
import jiwer
import whisper
import sys

# Check arguments
if len(sys.argv) != 5:
    raise ValueError("Please provide the model size as an argument. Example: python 1_decode_common_voice.py tiny")

# Get model size as argument
model_size = sys.argv[1] # tiny, small, medium, large-v3, turbo

# Load the TSV file
ts_file_path = sys.argv[2] # '/DB/cv-corpus-15.0-2023-09-08/ko/test.tsv'
audio_file_path = sys.argv[3] # '/DB/cv-corpus-15.0-2023-09-08/ko/clips'
language = sys.argv[4] # 'ko' 'es' 'ja'
data = pd.read_csv(ts_file_path, sep='\t')

# Check if the necessary columns exist
if 'path' not in data.columns or 'sentence' not in data.columns:
    raise ValueError("The TSV file must contain 'path' and 'sentence' columns.")

# Load Whisper model
# model = whisper.load_model("turbo")
# model = whisper.load_model("large-v3")
model = whisper.load_model(model_size)
model.eval()

# Check model size (number of parameters)
num_params = 0
for name, param in model.named_parameters():
    num_params += param.numel()
print('Model name:', model_size)
print(f"Model size: {num_params} parameters")

# Function to transcribe audio and compute WER
def calculate_wer(audio_paths, references):
    predictions = []
    for audio_path in audio_paths:
        audio_path = audio_file_path + '/' + audio_path
        result = model.transcribe(audio_path, language=language, beam_size=10, temperature=0.0)
        predictions.append(result['text'])

    # Compute WER and CER
    wer_with_special_tokens = jiwer.wer(references, predictions)
    cer_with_special_tokens = jiwer.cer(references, predictions)

    # Remove special characters
    predictions = [p.replace(".", "").replace("?", "").replace("!", "").replace(",", "").strip() for p in predictions]
    references = [r.replace(".", "").replace("?", "").replace("!", "").replace(",", "").strip() for r in references]

    # Compute WER
    wer = jiwer.wer(references, predictions)
    for idx in range(len(references)):
        print("REF:", references[idx])
        print("HYP:", predictions[idx])
    
    # Compute CER
    cer = jiwer.cer(references, predictions)

    # Compute WER/CER with maximum bound 1.0
    utterance_wer = []
    utterance_cer = []
    for ref, pred in zip(references, predictions):
        wer_utt = jiwer.wer([ref], [pred])
        ce_uttr = jiwer.cer([ref], [pred])
        clamped_wer = min(max(wer_utt, 0), 1)  # Clamp WER between 0 and 1
        clamped_cer = min(max(ce_uttr, 0), 1)  # Clamp CER between 0 and 1
        utterance_wer.append(clamped_wer)
        utterance_cer.append(clamped_cer)

    # Compute overall WER/CER as the average of clamped WERs/CERs
    clamped_wer_avg = sum(utterance_wer) / len(utterance_wer)
    clamped_cer_avg = sum(utterance_cer) / len(utterance_cer)

    return cer, wer, cer_with_special_tokens, wer_with_special_tokens, clamped_wer_avg, clamped_cer_avg, predictions

# Prepare audio paths and reference transcriptions
audio_paths = data['path'].tolist()
references = data['sentence'].tolist()

# Calculate WER
cer, wer, cer_with_special_tokens, wer_with_special_tokens, clamped_wer_avg, clamped_cer_avg, predictions = calculate_wer(audio_paths, references)

# Output results
print(f"WER: {wer}")
print(f"CER: {cer}")

print(f"WER_clamped: {clamped_wer_avg}")
print(f"CER_clamped: {clamped_cer_avg}")

print(f"WER (with special tokens): {wer_with_special_tokens}")
print(f"CER (with special tokens): {cer_with_special_tokens}")

# Optionally, save predictions and references for review
output_df = pd.DataFrame({
    'Audio Path': audio_paths,
    'Reference': references,
    'Prediction': predictions
})
output_path = 'wer_results.csv'
output_df.to_csv(output_path, index=False)
print(f"Detailed results saved to {output_path}")