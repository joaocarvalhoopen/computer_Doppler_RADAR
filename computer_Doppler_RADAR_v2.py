'''
Program: computer_Doppler_RADAR_v2.py
Author: Joao Nuno Carvalho
email:  joaonunocarv ( __AT__ ) gmail.com
Description: This program is a Doppler RADAR for a PC. In reality it’s a sonar that can determine the velocity
             of a moving object in relation to the PC speaker/mic when you put the speaker and mic near each
             other like in a laptop.
             This program is written in Python 3.5 and works in real-time using the PyAudio interface and NumPy.
             For this reason, in windows I have used the Anaconda 3.5 Distribution.

             This program is an implementation of the paper by Microsoft Research:

             SoundWave: Using the Doppler Effect to Sense Gestures
             Sidhant Gupta, Dan Morris, Shwetak Patel, Desney Tan
             Proceedings of ACM CHI 2012, May 2012
             http://research.microsoft.com/en-us/um/redmond/groups/cue/SoundWave/

             It has been inspired by a similar project that also implements this paper but for Javascript/Browser.
             https://github.com/DanielRapp/doppler

             To test this program, you should disable in the microphone settings, the echo cancelation and the
             noise reduction. In a small number of laptops, you must use an external microphone because the
             frequency response of those microphones has no gain at high frequencies, but this is rare.

             How the program works?
             This program makes two things at the same time, first it generates a sine wave of 18KHz that is
             outputted by the speakers and at the same time, it is processing the input of the microphone.
             All this action occurs in a callback function of the PyAudio frameworks that interfaces with the
             sound board of the computer. Each time the callback in called it receives a buffer with the
             microphone data of size 1024 and then it accumulates in a 2048 buffer, it then makes the FFT of
             that buffer and look for the amplitude of the signal at the frequencies around the 18KHz primary
             tone that we generate. If we move hour hand in the direction of the microphone we compress the
             sound wave, this means that there will be some amplitude in the frequencies upper to 18Khz,
             if we move hour hand away we will see some amplitude in the frequencies bellow the 18KHz.
             This is the Doppler effect and is why you hear a difference sound when a car passes through you
             pressing the horn. This is also the what police RADAR uses but that is with electromagnetic waves
             and this is also why we know the starts of distant galaxies are going away from us.
             That the universe is expanding. In the last case, the light of supernovas is becoming more reddish
             that it should be if they weren’t moving away. Light is also a electromagnetic wave.

             Going back to the program, then in the callback we put the result is a global variable doppler_shift
             and we copy the previous generated sin tone of 18KHz into the output buffer of the callback.

             In an external thread we periodically (0.5 seconds), see be the value of this variable and print it
             to the screen as a text graph, at the end we plot the last FFT graph so we can see the signal.



The MIT License (MIT)

Copyright (c) 2016 Joao Nuno Carvalho

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import pyaudio
import time
import numpy as np
import wave
import math


CHANNELS = 1
SAMPLE_RATE = 44100

p = pyaudio.PyAudio()

frame_count_global = 0
frames_to_file = []

# Size of the buffer that the callback receives and gives to the PyAudio interface.
chunk_length = 1024
result = np.zeros((chunk_length,1), dtype=np.float32)

######################################################
######################################################
#
# Generate the output tone inaudible tone of 18KHz.
#

# Frequency that we will be generating in the speakers.
gen_freq = 18000 # 18KHz

# The buffer as the size corresponding to 1 second.
sine_tone_buffer_size = 44100
sin_tone_buffer = np.zeros( (sine_tone_buffer_size,1), dtype=np.float32)

# Sin equation to generate a wave.
# y(t) = a*sin(2*pi*f*t + teta)
#
# in wich:
# f = 18000  # 18KHz tone frequency
# t = i / SAMPLE_RATE
# teta is the initial phase

def fill_sin_tone_buffer(buffer, freq):
    amplitude = 1.0
    for i in range(0, len(buffer)):
        t = i / SAMPLE_RATE
        buffer[i] = amplitude * math.sin( 2.0 * math.pi * freq * t ) # + teta initial phase ...... Wave equation with sine.
    buffer[0] = 0.0
    return buffer

sin_tone_buffer = fill_sin_tone_buffer(sin_tone_buffer, gen_freq)

ptr_begin = 0

# Copy sin buffer to speakers.
def copy_sine_tone_buffer_to_output_buffer():
    global result, ptr_begin, sine_tone_buffer_size, sine_tone_buffer

    ptr_end = ptr_begin + chunk_length
    if (ptr_end < sine_tone_buffer_size):
        # copy the buffer in one time. ptr_begin and ptr_end
        np.copyto(result[0 : chunk_length ], sin_tone_buffer[ptr_begin : ptr_end])
        ptr_begin = ptr_begin + chunk_length
        if ptr_begin == sine_tone_buffer_size:
            ptr_begin = 0
    else:
        lacks =  ptr_end - sine_tone_buffer_size
        dest_end = chunk_length - lacks
        # copy the first part of the buffer. ptr_begin and sine_tone_buffer_size
        np.copyto(result[0 : dest_end], sin_tone_buffer[ptr_begin : sine_tone_buffer_size])
        dest_start = dest_end
        ptr_begin = 0
        # copy the second part of the buffer from the begining. ptr_begin and lacks
        np.copyto(result[dest_start : chunk_length ], sin_tone_buffer[ptr_begin : lacks])
        # Prepares for the next chunk filling
        ptr_begin = ptr_begin + lacks
    return result

######################################################
######################################################
#
# Process the microphone input sound.
#

MIC_BUFFER_SIZE = chunk_length*2
mic_buffer = np.zeros( MIC_BUFFER_SIZE, dtype=np.float32)
mic_buffer_cur_index = 0

doppler_shift = (0,0)
# Relevant index window on the FFT, as described in the paper.
relevant_freq_window = 33

def freqToFFTIndex(freq):
    nyquist = SAMPLE_RATE / 2.0;
    return round( (freq / nyquist) * (MIC_BUFFER_SIZE / 2.0) );

fft_primary_tone_index = freqToFFTIndex(gen_freq)

def copy_mic_chunk_to_mic_buffer(decoded_mic_data):
    '''
    Receives a decoded buffer in NumPy format and copies it to the mic_buffer that has two 2048 positions.
    It receives two buffers in two times and at the end, returns that it can calculate the FFT for
    the doppler shift.
    '''
    global mic_buffer, mic_buffer_cur_index
    if mic_buffer_cur_index == 0:
        np.copyto(mic_buffer[0 : chunk_length ], decoded_mic_data[0 : chunk_length])
        mic_buffer_cur_index = 1
        return False
    else:
        np.copyto(mic_buffer[chunk_length : chunk_length * 2 ], decoded_mic_data[0 : chunk_length])
        mic_buffer_cur_index = 0
        return True

def calc_doppler_direction():
    global mic_buffer, fft_primary_tone_index, doppler_shift, relevant_freq_window
    fft_data = np.abs( np.fft.fft(mic_buffer) )

    # Obtain the bandwith of the shifted signal that looks like a step function below the amplitude
    # of the primary tone and upper from the ammplitude of the noise.

    primary_tone_volume = fft_data[fft_primary_tone_index]
    # This is an empirical ratio, find by experimentation but is equal to the one described in the paper 10%.
    max_volume_ratio = 0.1;     # This ratio works
    #max_volume_ratio = 0.05;   # x20 lower than the signal value.

    left_bandwidth = 0
    last_left_bandwidth = 1

    while True:
        left_bandwidth += 1
        volume = fft_data[fft_primary_tone_index - left_bandwidth]
        normalized_volume = volume / primary_tone_volume
        if (normalized_volume > max_volume_ratio):
            last_left_bandwidth = left_bandwidth
        if not ((left_bandwidth < relevant_freq_window)):
            break
    left_bandwidth = last_left_bandwidth

    right_bandwidth = 0
    last_right_bandwidth = 1
    while True:
        right_bandwidth += 1
        volume = fft_data[fft_primary_tone_index + right_bandwidth]
        normalized_volume = volume / primary_tone_volume
        if (normalized_volume > max_volume_ratio):
            last_right_bandwidth = right_bandwidth
        if not ( (right_bandwidth < relevant_freq_window)):
            break
    right_bandwidth = last_right_bandwidth

    # There are two threads working in this program, the comunication between those
    # two theads is made by a global variable "doppler_shift".
    doppler_shift = (int(left_bandwidth), int(right_bandwidth))
    #return (left_bandwidth, right_bandwidth)

######################################################
######################################################

# This program only uses one channel (Mono).
def decode(in_data):
    # Convert the PyAudio mono byte stream [R0, R1, R2, ...] to a NumPy array.
    # Shape(chunk_length) instruction.
    result = np.fromstring(in_data, dtype=np.float32)
    return result

def encode(signal):
    # Convert a 1D numpy array into a mono byte stream for PyAudio.
    # Signal has chunk_length rows and one columns. It's a vector.
    out_data = signal.astype(np.float32).tostring()
    return out_data

def encode_int16(signal):
    # Convert a 1D numpy array into a mono byte stream for PyAudio.
    # Signal has chunk_length rows and one columns.
    # The output is scalled to a int16 value not a -1 to 1 value.
    signal_tmp = signal * ((2**15) - 1)
    out_data = signal_tmp.astype(np.int16).tostring()
    return out_data


# The callback receives the buffer of the input microphone,
# and returns the buffer to output in the speakers.
def callback(in_data, frame_count, time_info, flag):
    global flag_plunk, result, frame_count_global, frames_to_file
    frame_count_global = frame_count

    # Processing the input data from microphone.
    decoded_mic_data = decode(in_data)
    flag_calc_Doppler_shift = copy_mic_chunk_to_mic_buffer(decoded_mic_data)
    # Detect the doppler effect, the shift in frequency around the central tone.
    if flag_calc_Doppler_shift == True:
        calc_doppler_direction()

    # Generating the output data to the speakers (continuos tone).
    copy_sine_tone_buffer_to_output_buffer()
    out_data = encode(result)

    # Collect the output generated and at the end writes a WAV file with them.
    frames_to_file.append(encode_int16(result))
    return (out_data, pyaudio.paContinue)


stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,               # The callback will generate data to output to the speakers,
                input= True,               # will also receive data from the microphone/input.
                stream_callback=callback)  # The data is passed around in buffers of size chunk_length.

stream.start_stream()

time_between_prints = 0.5 # 0.5 seconds
counter = 0
print('Press Ctrl+C to quit: ')
while stream.is_active():
    time.sleep(time_between_prints)
    #ch = input("Press (p to quit): ")

    left_bandwith  = int(doppler_shift[0])
    right_bandwith = int(doppler_shift[1])
    #print(doppler_shift)
    str_left_space = ' ' * ( relevant_freq_window - left_bandwith)
    str_left  = '<' * left_bandwith
    str_right = '>' * right_bandwith
    str_right_space = ' ' * ( relevant_freq_window - right_bandwith)
    print(str_left_space,str_left, "||", str_right, str_right_space )

    if counter == 250:
        # Times out in 250 cycles.
        # Exit and write to file.
        stream.stop_stream()
    counter +=1


stream.stop_stream()
stream.close()
p.terminate()


# Save the output to WAV file.
WAVE_OUTPUT_FILENAME = "WAV_Doppler_output.wav"
#FORMAT = pyaudio.paFloat32
FORMAT = pyaudio.paInt16

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(SAMPLE_RATE)
wf.writeframes(b''.join(frames_to_file))
wf.close()

print('Session saved to WAV file ... WAV_Doppler_output.wav')
print("frame_count_global:", frame_count_global)


###################################
###################################
#
# Plot the final 2048 FFT values.
#

fft_data = np.abs( np.fft.fft(mic_buffer))
print('fft_primary_tone_index: ', fft_primary_tone_index )
print('fft_data[fft_primary_tone_index]: ', fft_data[fft_primary_tone_index] )

for i in range(820, 850):
    print('fft_data[', i, ']: ', fft_data[i] )

import matplotlib.pyplot as plt
#plt.plot( np.abs( np.fft.fft(mic_buffer) ) )
plt.plot( np.abs( np.fft.fft(mic_buffer) )[820:850] )
plt.show()

###################################
###################################
