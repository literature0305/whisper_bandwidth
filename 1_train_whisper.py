import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from torch.optim.lr_scheduler import StepLR

from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)

LANGUAGE_TRAIN='en'
LANGUAGE_VALID='en'
BATCH_SIZE=1

################################
# 1) Dataset loading utilities
################################

class CommonVoiceDataset(Dataset):
    """
    A small dataset wrapper for reading from a .tsv file (train or valid).
    Expects columns (at least): 'path' and 'sentence' from the example you gave.
    """
    def __init__(self, tsv_path: str, audio_dir: str, transform=None):
        super().__init__()
        self.df = pd.read_csv(tsv_path, sep='\t')
        self.audio_dir = audio_dir
        self.transform = transform

        # Filter out rows that have missing path/sentence if needed
        self.df = self.df.dropna(subset=["path", "sentence"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["path"])
        sentence = row["sentence"]
        sample = {
            "audio_path": audio_path,
            "sentence": sentence,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    """
    A simple collate function. In practice, you might:
     - Load waveforms or log-mels inside the transform or here
     - Tokenize text
    """
    audio_paths = [sample["audio_path"] for sample in batch]
    texts = [sample["sentence"] for sample in batch]
    return audio_paths, texts


#######################################
# 2) Load pretrained Whisper (small)
#    and build a wrapper with trainable T
#######################################
import whisper  # if installed via pip; or local import from the code you pasted

class WhisperTemperatureWrapper(nn.Module):
    """
    Wrap a pretrained Whisper model but add a trainable temperature T.
    We override the forward pass so that the final logits can be divided by T
    before computing a loss.
    """
    def __init__(self, whisper_model: whisper.Whisper):
        super().__init__()
        self.whisper_model = whisper_model
        # Freeze all parameters in the base Whisper model
        # for param in self.whisper_model.parameters():
        #     param.requires_grad = False
        for name, param in self.whisper_model.named_parameters():
            # if "temperature" in name or "bandwidth" in name:
            if "temperature" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


        # Create a trainable scalar T, initialized to 1.0
        # We'll ensure T>0 by exponentiating if desired, but here we rely on the optimizer initialization
        num_encoder_layer = self.whisper_model.encoder.num_layers
        num_decoder_layer = self.whisper_model.decoder.num_layers
        self.bandwidth_enc = nn.Parameter(torch.ones(num_encoder_layer, dtype=torch.float32))
        self.bandwidth_dec = nn.Parameter(torch.ones(num_decoder_layer, dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, mel_input: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 
          1) Encode the audio into audio_features
          2) Pass (text_tokens, audio_features) to the decoder
          3) Scale (divide) the final logits by T to control "smoothness"
        """
        audio_features = self.whisper_model.encoder(mel_input, bandwidth=self.bandwidth_enc)
        logits = self.whisper_model.decoder(text_tokens, audio_features, bandwidth=self.bandwidth_dec)
        # Scale logits by 1 / T
        # (You could also multiply by (1 / T) or do: logits / self.temperature)
        scaled_logits = logits / (abs(self.temperature) + 1e-6)
        # print('self.temperature:', self.temperature)
        # print('self.bandwidth_enc:', self.bandwidth_enc)
        return scaled_logits


#################################
# 3) Training Step & CER
#################################

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Very naive character error rate function for demonstration.
    For production, you'd likely want a proper edit-distance-based CER.
    """
    ref = reference.replace(" ", "").lower()
    hyp = hypothesis.replace(" ", "").lower()
    # count mismatches
    min_len = min(len(ref), len(hyp))
    mismatches = sum(r != h for r, h in zip(ref[:min_len], hyp[:min_len]))
    # account for differences in length
    mismatches += abs(len(ref) - len(hyp))
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return mismatches / len(ref)


def train_one_epoch(model, dataloader, optimizer, tokenizer, device="cuda"):
    """
    For each sample:
      1) Load audio -> compute mel
      2) Tokenize text
      3) Forward + cross-entropy loss (freezing model's weights, only T is trained)
    """
    model.train()  # though the underlying whisper is frozen, T is trainable
    model.whisper_model.train()  # freeze the underlying whisper model
    total_loss = 0.0
    num_samples = 0

    # Freezing the model's weights, only temperature and bandwidths are trained
    for name, param in model.named_parameters():
        # if "temperature" in name or "bandwidth" in name:
        if "temperature" in name:
            param.requires_grad = True
            print('name:', name, 'value:', param)
        else:
            param.requires_grad = False
    
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


    ce_loss_fn = nn.CrossEntropyLoss()

    for audio_paths, texts in dataloader:
        # Example: process one by one or in a batch
        # For brevity, let's show single-sample logic:
        optimizer.zero_grad()
        batch_loss = 0.0

        for audio_path, txt in zip(audio_paths, texts):
            # 1) Convert audio -> mel with whisper's helper
            mel = whisper.log_mel_spectrogram(audio_path, model.whisper_model.dims.n_mels, padding=N_SAMPLES).unsqueeze(0).to(device)
            # print('mel size:', mel.size()) # [1, 80, 3838]
            # mel = whisper.log_mel_spectrogram(audio_path, model.dims.n_mels)
            # mel = whisper.log_mel_spectrogram(waveform.squeeze(), self.whisper_model.dims.n_mels, padding=N_SAMPLES - waveform.shape[1])

            # 2) Tokenize text with the model's tokenizer
            #    For real usage, handle longer/shorter texts carefully
            text_tokens = tokenizer.encode(txt)
            # <|startoftranscript|><|en|><|transcribe|><|notimestamps|> + text_tokens
            text_tokens = [tokenizer.sot] + [tokenizer.to_language_token(LANGUAGE_TRAIN)] + [tokenizer.transcribe] + [tokenizer.no_timestamps] + text_tokens
            text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device).unsqueeze(0)

            # 3) Forward pass. We typically want teacher-forcing, i.e. predict next token...
            #    We'll just illustrate a single-step approach, or you can shift tokens.
            logits = model(mel, text_tokens[:, :-1])  # predict the next token
            # logits shape: (batch=1, seq_len, vocab_size)
            # we want CE with text_tokens[:, 1:] as the label
            target = text_tokens[:, 1:].contiguous().view(-1)
            pred = logits.contiguous().view(-1, logits.size(-1))
            if torch.randperm(100)[0] == 0:
                pred_argmax = pred.argmax(dim=-1)
                print('pred:', pred)
                print('target:', target)
                print('pred_argmax:', pred_argmax)
                print('decoded pred:', tokenizer.decode(pred_argmax.tolist()))

            loss = ce_loss_fn(pred, target)
            batch_loss += loss

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_loss += loss.item()
            num_samples += 1

        batch_loss = batch_loss / len(audio_paths)
        if torch.randperm(100)[0] == 0:
            print('batch size:', len(audio_paths))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        batch_loss.backward()
        optimizer.step()


    return total_loss / max(1, num_samples)


def validate(model, dataloader, tokenizer, device="cuda"):
    """
    Evaluate the model by transcribing or partially decoding the audio
    and comparing with references (CER).
    """

    bandwidth_dec = model.bandwidth_dec
    bandwidth_enc = model.bandwidth_enc
    post_hoc_temperature = model.temperature

    model.eval()
    cer_scores = []
    with torch.no_grad():

        for audio_paths, texts in dataloader:
            for audio_path, ref_text in zip(audio_paths, texts):
                # mel = whisper.log_mel_spectrogram(audio_path).unsqueeze(0).to(device)

                # We'll do a simple, greedy decode with the *adjusted* temperature,
                # using the official whisper `transcribe` function. 
                # But we have to patch the temperature. 
                # For brevity, let's do a single segment decode to illustrate:
                
                # Overwrite the original model's decode to inject our scaled logits:
                # The easiest hack in practice is to set model.temperature=...
                # But since we've introduced it as a parameter, let's temporarily override
                # the built-in decode (this can get tricky in real code).
                
                # Quick approach: Just run the standard transcribe while temporarily
                # setting the model's temperature parameter. 
                # Or do a manual decode.  For demonstration, we'll forcibly do:
                predicted_text = model.whisper_model.transcribe(
                    audio_path,
                    verbose=False,
                    post_hoc_temperature=post_hoc_temperature,
                    post_hoc_bandwidth_enc=bandwidth_enc,
                    post_hoc_bandwidth_dec=bandwidth_dec,
                    temperature=0.0,
                )["text"]

                cer = compute_cer(ref_text, predicted_text)
                cer_scores.append(cer)

    if len(cer_scores) == 0:
        return 0.0
    return sum(cer_scores) / len(cer_scores)


############################
# 4) Main Training Loop
############################

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # a) Load CSV/TSV data
    train_tsv = "/DB/cv-corpus-19.0-delta-2024-09-13/en/validated.tsv"  # path to your train.tsv
    valid_tsv = "/DB/cv-corpus-20.0-delta-2024-12-06/en/validated.tsv"  # path to your valid.tsv
    audio_dir_train = "/DB/cv-corpus-19.0-delta-2024-09-13/en/clips"          # folder containing your .mp3 files
    audio_dir_valid = "/DB/cv-corpus-20.0-delta-2024-12-06/en/clips"          # folder containing your .mp3 files

    train_dataset = CommonVoiceDataset(train_tsv, audio_dir_train)
    valid_dataset = CommonVoiceDataset(valid_tsv, audio_dir_valid)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # b) Load pretrained Whisper "small"
    # base_model = whisper.load_model("small", device=device)  
    base_model = whisper.load_model("large-v3", device=device)  
    # or load from local path / cloned repo if needed:
    # base_model = torch.load("YOUR_CHECKPOINT_PATH", map_location=device)

    # c) Wrap the model with trainable temperature
    model = WhisperTemperatureWrapper(base_model).to(device)

    # The whisper tokenizer
    # tokenizer = base_model.tokenizer # not work
    # tokenizer = get_tokenizer(multilingual=False, language=LANGUAGE, task='asr') # not work
    tokenizer = get_tokenizer(multilingual=base_model.is_multilingual, num_languages=base_model.num_languages, language=LANGUAGE_TRAIN, task='asr') # not work
    tokenizer_valid = get_tokenizer(multilingual=base_model.is_multilingual, num_languages=base_model.num_languages, language=LANGUAGE_VALID, task='asr') # not work

    # d) Optimizer to only train the single T parameter
    # optimizer = optim.Adam([model.temperature], lr=1e-3)
    optimizer = optim.Adam([model.temperature, model.bandwidth_dec, model.bandwidth_enc], lr=1e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    
    # d-1) 
    # optimizer = torch.optim.AdamW([
    #     {'params': [p for n, p in model.named_parameters() if "bandwidth" not in n]},
    #     {'params': [p for n, p in model.named_parameters() if "bandwidth" in n], 'lr': 1e-5}
    # ], lr=1e-4)

    best_cer = float("inf")
    best_epoch = -1
    best_temp = None


    # e) Initial validation without temperature and bandwidth training
    val_cer = validate(model, valid_loader, tokenizer_valid, device)
    train_cer = validate(model, train_loader, tokenizer, device)
    print("(before training) Val CER:", val_cer)
    print("(before training) Train CER:", train_cer)

    # f) Train loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, tokenizer, device)
        val_cer = validate(model, valid_loader, tokenizer_valid, device)
        train_cer = validate(model, train_loader, tokenizer, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val CER: {val_cer:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")

        # f) Track the best model (i.e., best temperature T)
        if val_cer < best_cer:
            best_cer = val_cer
            best_epoch = epoch
            best_temp = model.temperature.item()

            # Save the best "model" – in practice, we only need T 
            save_path = "best_temp_checkpoint.pt"
            torch.save({"temperature": best_temp}, save_path)
            print(f"  [*] Best model so far. Saved temperature to {save_path}")

        scheduler.step()

    print(f"Training done. Best CER = {best_cer:.4f} at epoch {best_epoch+1}")
    print(f"Best T = {best_temp:.4f}")


if __name__ == "__main__":
    main()
