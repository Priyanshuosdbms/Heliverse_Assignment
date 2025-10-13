import whisper
import optuna
import re
import time

AUDIO_FILE = "meeting.mp4"

def quality_score(text):
    """Heuristic scoring: balance between length and lexical diversity."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    unique_ratio = len(set(words)) / len(words)
    length_score = min(len(words) / 200, 1.0)  # 1.0 if ~200+ words
    return 0.5 * unique_ratio + 0.5 * length_score  # weighted average

def objective(trial):
    # Tune hyperparameters
    model_name = trial.suggest_categorical("model_name", ["tiny", "base", "small"])
    temperature = trial.suggest_float("temperature", 0.0, 1.0)
    beam_size = trial.suggest_int("beam_size", 1, 5)

    # Load model
    model = whisper.load_model(model_name)
    start = time.time()
    result = model.transcribe(AUDIO_FILE, temperature=temperature, beam_size=beam_size, verbose=False)
    elapsed = time.time() - start

    text = result["text"]
    score = quality_score(text)

    print(f"Trial {trial.number}: {model_name}, temp={temperature:.2f}, beam={beam_size}, "
          f"score={score:.3f}, words={len(text.split())}, time={elapsed:.1f}s")

    # Return heuristic quality score
    return score

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=8)

print("\n✅ Best parameters:")
print(study.best_params)

------------------------

files = ["vid1.mp4", "vid2.mp4", "vid3.mp4"]
avg_score = 0
for f in files:
    result = model.transcribe(f, temperature=temperature, beam_size=beam_size, verbose=False)
    avg_score += quality_score(result["text"])
return avg_score / len(files)