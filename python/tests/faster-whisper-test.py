from faster_whisper import WhisperModel

model = WhisperModel("testou")

segments, info = model.transcribe("harvard.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))