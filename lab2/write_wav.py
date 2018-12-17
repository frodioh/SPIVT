import sounddevice as sd
import numpy as np
import keyboard
import scipy.io.wavfile as wav

fs=44100
duration = 1
print('Для записи нажмите r...')
is_recording = False
while True:
    if keyboard.is_pressed('r') and not(is_recording):
        myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='int16')
        is_recording = True
    if not(keyboard.is_pressed('r')) and is_recording:
        sd.stop()
        is_recording = False
        break
wav.write('speech_commands/out.wav', fs, myrecording)
print("Play Audio Complete")
