import pvporcupine
import sounddevice as sd
import numpy as np

ACCESS_KEY = "nW6xiihdtPEdzVznAuja8Cg6WiAYATZdOpQAAoJSQUTwOr3MACfq/w=="  

def wait_for_wake_word():
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keywords=["jarvis"]  
    )

    print("[WAKE] Listening for wake word...")

    with sd.InputStream(
        samplerate=porcupine.sample_rate,
        channels=1,
        dtype="int16",
        blocksize=porcupine.frame_length
    ) as stream:
        while True:
            audio_chunk, _ = stream.read(porcupine.frame_length)
            pcm = audio_chunk.flatten().tolist()
            result = porcupine.process(pcm)
            if result >= 0:
                print("[WAKE] Wake word detected!")
                porcupine.delete()
                return