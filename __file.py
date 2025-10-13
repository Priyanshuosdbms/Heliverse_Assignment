import whisper
import optuna
import re
import time
import os

# Folder containing your MP3 files
AUDIO_FILES = ["audio1.mp3", "audio2.mp3"]  # add all your files here
OUTPUT_FILE = "best_transcript.txt"

def quality_score(text):
    """
    Heuristic scoring: balance between transcript length and lexical diversity.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    unique_ratio = len(set(words)) / len(words)
    length_score = min(len(words) / 200, 1.0)  # ideal ~200+ words
    return 0.5 * unique_ratio + 0.5 * length_score

def objective(trial):
    # Suggest hyperparameters
    model_name = trial.suggest_categorical("model_name", ["tiny", "base", "small"])
    temperature = trial.suggest_float("temperature", 0.0, 1.0)
    beam_size = trial.suggest_int("beam_size", 1, 5)

    # Load model
    model = whisper.load_model(model_name)
    total_score = 0

    for audio_file in AUDIO_FILES:
        result = model.transcribe(audio_file, temperature=temperature, beam_size=beam_size, verbose=False)
        text = result["text"]
        score = quality_score(text)
        total_score += score

    avg_score = total_score / len(AUDIO_FILES)
    print(f"Trial {trial.number}: model={model_name}, temp={temperature:.2f}, beam={beam_size}, avg_score={avg_score:.3f}")
    return avg_score

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("\n✅ Best parameters found:")
print(study.best_params)

# Transcribe all files with the best parameters
best_model = whisper.load_model(study.best_params["model_name"])
all_transcripts = []

for audio_file in AUDIO_FILES:
    result = best_model.transcribe(
        audio_file,
        temperature=study.best_params["temperature"],
        beam_size=study.best_params["beam_size"]
    )
    transcript = f"--- {audio_file} ---\n{result['text']}\n\n"
    all_transcripts.append(transcript)

# Save combined transcript to a .txt file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(all_transcripts)

print(f"\n✅ Transcripts saved to {OUTPUT_FILE}")