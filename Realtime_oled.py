import pandas as pd
import math
import numpy as np
from scipy.signal import correlate2d, fftconvolve, butter, filtfilt
from scipy.stats import kurtosis, skew, entropy
import pywt
from pathlib import Path
import joblib
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import spidev
import time
import RPi.GPIO as GPIO
from scipy.signal import welch

# Import the trained models    

ad = joblib.load('/home/Aspect661/Documents/Projects/Real_time_models/xgbod_model.pkl')
clf = joblib.load('/home/Aspect661/Documents/Projects/Real_time_models/svm.pkl')
nm = joblib.load('/home/Aspect661/Documents/Projects/Real_time_models/Normalize.pkl')

# Baseline Removal

def remove_baseline_wander(dataset, sampling_freq=125):
    num_rows, num_cols = dataset.shape
    baseline_corrected_dataset = np.zeros((num_rows, num_cols))
    # interpolated_baselines = np.zeros((num_rows, num_cols))

    for row in range(num_rows):
        signal = dataset[row, :]
        troughs, _ = find_peaks(-signal)
        baseline = np.interp(np.arange(len(signal)), troughs, signal[troughs])

        # Interpolate the baseline using the same time array as the original signal
        time = np.arange(len(signal))
        interpolated_baseline = np.interp(time, np.arange(len(baseline)), baseline)

        baseline_corrected_signal = signal - interpolated_baseline
        baseline_corrected_dataset[row, :] = baseline_corrected_signal

    return baseline_corrected_dataset

def butterworth(X_demo):
    rows, columns = X_demo.shape
    result = np.zeros((rows, columns))
    for i in range(rows):
        sig = X_demo[i,:]
        fs = 125
        fc = 12
        fc1 = 0.01
        w = fc / (fs / 2)
        w1 = fc1 / (fs / 2)
        b, a = butter(6, w1, 'highpass')
        f_sig = filtfilt(b, a, sig)
        b, a = butter(6, w, btype = 'lowpass')
        f_sig = filtfilt(b, a, sig)
        result[i,:] = f_sig
    return result

def normalize(X_demo):
    mm = MinMaxScaler()
    X_temp = (mm.fit_transform(X_demo.T)).T
    return X_temp

def derivatives(X_demo):

    # Initialize lists
    first_derivatives = []
    second_derivatives = []
    third_derivatives = []
    rows, columns = X_demo.shape
    for i in range(rows):  # Transpose to iterate over columns
        signal = X_demo[i,:]
        dt = 125
        first_derivative = np.gradient(signal, dt)
        second_derivative = np.gradient(first_derivative, dt)
        third_derivative = np.gradient(second_derivative, dt)
        first_derivatives.append(first_derivative)
        second_derivatives.append(second_derivative)
        third_derivatives.append(third_derivative)

    first_derivatives = np.array(first_derivatives)
    second_derivatives = np.array(second_derivatives)
    third_derivatives = np.array(third_derivatives)

    first_derivatives *= 1000
    second_derivatives *= 1000000
    third_derivatives *= 100000000

    return first_derivatives, second_derivatives, third_derivatives

## Statistical

def calculate_statistical_features(signal):
    # Calculate statistical features for a given signal
    mean = np.mean(signal)
    first_quartile = np.percentile(signal, 25)
    second_quartile = np.median(signal)
    third_quartile = np.percentile(signal, 75)
    std_dev = np.std(signal)
    kurt = kurtosis(signal)
    skewness = skew(signal)
    mean_abs_dev = np.mean(np.abs(signal - mean))

    return [mean, first_quartile, second_quartile,third_quartile, std_dev, kurt, skewness, mean_abs_dev]

def statistical(X_filtered):
    X_biof_stat = []
    rows, columns = X_filtered.shape

    for i in range(rows):
        row_features = []
        # Extract features from the original signal
        signal = X_filtered[i,:]
        row_features.extend(calculate_statistical_features(signal))
        X_biof_stat.append(row_features)

    # Convert the list to a DataFrame
    columns = ['Mean', 'FirstQuartile', 'SecondQuartile','ThirdQuartile', 'StandardDeviation', 'Kurtosis', 'Skewness', 'MeanAbsoluteDeviation']
    X_biof_stat_df = pd.DataFrame(X_biof_stat, columns=columns)

    return X_biof_stat_df

## Entropy

def ApEn(U, m, r):
    N = len(U)
    phi = np.zeros((N - m + 1, 1))
    r *= np.std(U)
    for i in range(N - m + 1):
        idx = np.arange(i, i + m)
        Phi_i = np.sum(np.abs(U[idx][:, None] - U[idx]), axis=1)
        phi[i] = np.sum(Phi_i <= r) - 1
    return np.mean(phi)
def shannon_entropy(signal):
    """Compute the Shannon entropy of a signal."""
    _, counts = np.unique(signal, return_counts=True)
    probabilities = counts / len(signal)
    return -np.sum(probabilities * np.log2(probabilities))
def sample_entropy(U, m):
    """Compute the sample entropy of a signal U."""
    N = len(U)
    r = 0.15 * np.std(U)
    m += 1
    mm = m - 1
    phi = np.zeros(N - mm)
    X = np.array([U[i:i+m] for i in range(N - mm)])
    D = np.abs(X[:, None, :] - X)
    D = np.max(D, axis=2)
    count = np.sum(D <= r, axis=1)
    count -= 1
    phi = count / (N - mm)
    return np.mean(phi)
def en_fn(X_demo):
    r,c=X_demo.shape
    en = np.zeros((r, 3))
    for i in range(r):
        approximate_entropy = ApEn(X_demo[i,:], 2, 0.2)
        sample_en = sample_entropy(X_demo[i,:], 3)
        shannon_en = shannon_entropy(X_demo[i,:])
        en[i, :] = [approximate_entropy, sample_en, shannon_en]
    return pd.DataFrame(en)
    
def PSD_function(Sig):
  freqs, psd = welch(Sig)
  psd_mean = np.mean(psd)
  psd_max = np.max(psd)
  psd_min = np.min(psd)
  psd_var = np.var(psd)
  centroid_freq = np.sum(freqs * psd) / np.sum(psd)
  freq_variance = np.var(freqs)
  mean_square_freq = np.sum(freqs**2 * psd) / np.sum(psd)
  return psd_mean, psd_max, psd_min, psd_var, centroid_freq, mean_square_freq
  
def psd_fn(X_demo):
    r,c=X_demo.shape
    psd = np.zeros((r, 6))
    for i in range(r):
        psd_mean, psd_max, psd_min, psd_var, centroid_freq, mean_sq_freq = PSD_function(X_demo[i,:])
        psd[i, :] = [centroid_freq, mean_sq_freq, psd_mean, psd_max, psd_min, psd_var]

    return pd.DataFrame(psd)
    
## WPT

from pywt import wavedec
from scipy.stats import kurtosis, skew

def wpt_features(X_demo):
    rows, columns = X_demo.shape
    result_array = np.zeros((rows, 27))
    for i in range(rows):
        sig_demo = X_demo[i, :]
        wavelet = 'haar'
        level = 2
        wp = pywt.WaveletPacket(data=sig_demo, wavelet=wavelet, mode='symmetric', maxlevel=level)

        # Extract coefficients from the wavelet packet decomposition
        coeffs = [node.data for node in wp.get_level(level, 'freq')]
        coeffs_concatenated = np.concatenate(coeffs)

        psd_mean, psd_max, psd_min, psd_var, centroid_freq, mean_sq_freq = PSD_function(coeffs_concatenated)
        approximate_entropy = ApEn(coeffs_concatenated, 2, 0.2)
        sample_en = sample_entropy(coeffs_concatenated, 3)
        shannon_en = shannon_entropy(coeffs_concatenated)

        # Non-fiducial features
        Mean = np.mean(coeffs_concatenated)
        St_Dev = np.std(coeffs_concatenated)
        Median = np.median(coeffs_concatenated)
        Variance = np.var(coeffs_concatenated)
        Kurtosis = kurtosis(coeffs_concatenated)
        Skewness = skew(coeffs_concatenated)
        q1 = np.mean(np.percentile(coeffs_concatenated, 25))
        q3 = np.mean(np.percentile(coeffs_concatenated, 75))
        rms_value = np.sqrt(np.mean(coeffs_concatenated ** 2))
        maximum = np.max(coeffs_concatenated)
        minimum = np.min(coeffs_concatenated)
        signal_range = np.max(coeffs_concatenated) - np.min(coeffs_concatenated)
        iqr = q3 - q1
        cv = St_Dev / Mean
        mad = np.mean(np.abs(coeffs_concatenated - Mean))
        rmsd = np.sqrt(np.mean((coeffs_concatenated - Mean) ** 2))
        zero_crossing_rate = np.sum(np.diff(np.sign(coeffs_concatenated)) != 0) / (2 * len(coeffs_concatenated))
        spectral_centroid = np.sum(np.arange(len(coeffs_concatenated)) * np.abs(np.fft.fft(coeffs_concatenated))) / np.sum(np.abs(np.fft.fft(coeffs_concatenated)))
        spectral_flatness = np.exp(np.mean(np.log(np.abs(np.fft.fft(coeffs_concatenated))))) / np.mean(np.abs(np.fft.fft(coeffs_concatenated)))
        ##
        psd_mean, psd_max, psd_min, psd_var, centroid_freq, mean_sq_freq = PSD_function(coeffs_concatenated)
        approximate_entropy = ApEn(coeffs_concatenated, 2, 0.2)
        sample_en = sample_entropy(coeffs_concatenated, 3)
        shannon_en = shannon_entropy(coeffs_concatenated)
        ##

        result_array[i, :] = [Mean, St_Dev, Median, Variance, Kurtosis, Skewness, q1, q3, rms_value, maximum, \
                              minimum, signal_range, iqr, cv, mad, rmsd, zero_crossing_rate, spectral_centroid, \
                              spectral_flatness,centroid_freq, psd_mean, psd_max, psd_min, psd_var, \
                              approximate_entropy, sample_en, shannon_en]

    result_array = pd.DataFrame(result_array)
    return result_array

from scipy.stats import kurtosis, skew
from scipy.signal import welch

def calculate_frequency_features(signal):
    # Calculate frequency domain features for a given signal
    freqs, psd = welch(signal)
    fs = 125
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    fft_mag = np.abs(fft_signal)
    psd = (fft_mag ** 2) / n**2

    f = np.fft.fftfreq(n, 1/fs)
    fmax = fs / 2  # Nyquist frequency

    # Centroid Frequency (CF)
    cf_num = np.sum(f * psd)
    cf_den = np.sum(psd)
    centroid_freq = cf_num / cf_den if cf_den != 0 else 0

    # PSD Mean, Max, Min, and Variance
    psd_mean = np.mean(psd)
    psd_max = np.max(psd)
    psd_min = np.min(psd)
    psd_variance = np.var(psd)

    # Root Mean Square Frequency (RMSF)
    rmsf_num = np.sqrt(np.sum(f**2 * psd))
    rmsf_den = np.sum(psd)
    rmsf = rmsf_num / rmsf_den if rmsf_den != 0 else 0

    # Amplitude and Frequency of Peaks (assuming finding the highest peak)
    peak_amplitude = np.max(psd)
    peak_frequency = f[np.argmax(psd)]

    # Power at Given Frequency Range (e.g., 0-2 Hz)
    idx_0_to_2Hz = np.where((f >= 0) & (f <= 2))[0]
    power_0_to_2Hz = np.sum(psd[idx_0_to_2Hz])

    # Ratio of Powers at 0-5Hz and 0-2.25Hz
    idx_0_to_5Hz = np.where((f >= 0) & (f <= 5))[0]
    idx_0_to_2_25Hz = np.where((f >= 0) & (f <= 2.25))[0]
    ratio_powers = np.sum(psd[idx_0_to_5Hz]) / np.sum(psd[idx_0_to_2_25Hz])

    return [centroid_freq, psd_mean, psd_max, psd_min, psd_variance, rmsf, peak_amplitude, peak_frequency, power_0_to_2Hz, ratio_powers]

def freq_domain(X_filtered):
    fs = 125
    X_freq_domain = []
    rows, columns = X_filtered.shape

    for i in range(rows):
        row_features = []
        # Extract features from the original signal
        signal = X_filtered[i]
        row_features.extend(calculate_frequency_features(signal))
        X_freq_domain.append(row_features)

    # Convert the list to a DataFrame
    columns = ['CentroidFrequency', 'PSDMean', 'PSDMax', 'PSDMin', 'PSDVariance', 'RMSF', 'PeakAmplitude', 'PeakFrequency', 'Power0to2Hz', 'RatioPowers']
    X_freq_domain_df = pd.DataFrame(X_freq_domain, columns=columns)

    return X_freq_domain_df

import neurokit2 as nk

def neuro(X_demo):
    rows, columns = X_demo.shape
    fid_feat = []
    for i in range(rows):
        ppg = X_demo[i,:]
        # Process the PPG signal
        ppg_signals, info = nk.ppg_process(ppg, sampling_rate=125)

        # Create an epoch for the entire signal duration
        epochs = nk.epochs_create(ppg_signals, events=[0], epochs_start=0, epochs_end=5, sampling_rate=125)

        # Analyze the PPG epoch
        analyze_ppg = nk.ppg_analyze(epochs, sampling_rate=125)

        # Extract the features, removing 'Label' and 'Event_Onset' columns, and converting to a list
        features_list = analyze_ppg.drop(columns=['Label', 'Event_Onset']).values.flatten().tolist()

        # Append the features to fid_feat
        fid_feat.append(features_list)
    fid_feat = pd.DataFrame(fid_feat)
    return fid_feat

top_features_names = [3, 5, 6, 9, 10, 13, 14, 18, 22, 25, 26, 27, 30, 32, 35, 39, 42, 44, 48, 
56, 78, 83, 85, 86, 95, 100, 110, 113, 127, 131, 133, 140, 154, 159, 160]

def extract(signal):
    signal = signal.reshape(1,-1)
    filt = butterworth(signal)
    filt = butterworth(filt)
    br = remove_baseline_wander(filt,125)
    nor = normalize(br)
    first, second, third = derivatives(nor)

    # Fiducial
    fid = neuro(nor)
    
    # Stat
    stat = statistical(nor)
    stat =  pd.concat([stat, statistical(first)], axis = 1, ignore_index = True)
    stat =  pd.concat([stat, statistical(second)], axis = 1, ignore_index = True)
    stat =  pd.concat([stat, statistical(third)], axis = 1, ignore_index = True)

    # Freq
    freq = freq_domain(nor)
    freq = pd.concat([freq,freq_domain(first)], axis = 1, ignore_index = True)
    freq = pd.concat([freq,freq_domain(second)], axis = 1, ignore_index = True)
    freq = pd.concat([freq,freq_domain(third)], axis = 1, ignore_index = True)
    
    # Entropy
    ent = en_fn(nor)
    ent = pd.concat([ent, en_fn(first)], axis = 1, ignore_index = True)
    ent = pd.concat([ent, en_fn(second)], axis = 1, ignore_index = True)
    ent = pd.concat([ent, en_fn(third)], axis = 1, ignore_index = True)

    # WPT
    wpt = wpt_features(nor)
    wpt = pd.concat([wpt, wpt_features(first)], axis = 1, ignore_index = True)
    wpt = pd.concat([wpt, wpt_features(second)], axis = 1, ignore_index = True)
    wpt = pd.concat([wpt, wpt_features(third)], axis = 1, ignore_index = True)

    final = pd.concat([fid,freq,ent], axis = 1, ignore_index = True)
    final = nm.transform(final)
    high_value = 1e3
    final = np.nan_to_num(final, nan=high_value)
    final = final[:,top_features_names]
    
    return final

def print_oled(x):
    font1 = ImageFont.truetype("FreeMono.ttf", 28)
    font2 = ImageFont.truetype("FreeMono.ttf", 22)
    
    def draw_text(draw, text1, text2):
        draw.text((0, 0), text1, font=font2, fill="white")
        draw.text((0, 30), text2, font=font1, fill="white")

    if x == 0:
        text1 = "Unauthor\nized"
        text2 = ""
    elif x == 1:
        text1 = "Authorized"
        text2 = "Debasmita"
    elif x == 2:
        text1 = "Authorized"
        text2 = "Mousumi"
    elif x == 3:
        text1 = "Authorized"
        text2 = "Priya"
    elif x == 4:
        text1 = "Authorized"
        text2 = "Sahana"
    elif x == 5:
        text1 = "Authorized"
        text2 = "Shwashwat"
    elif x == 6:
        text1 = "Authorized"
        text2 = "Tuhin"
    else:
        text1 = "Authorized"
        text2 = "Tushita"
        
    device.clear()  # Clear the display
    time.sleep(0.5)  # Display for 0.5 seconds
    with canvas(device) as draw:
        draw_text(draw, text1, text2)
    

    # time.sleep(0.5)  # Pause for 0.5 seconds before next blink


from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import ImageFont, ImageDraw
import time 
	
# Create an I2C interface with the detected address 0x3C
serial = i2c(port=1, address=0x3C)

# Create the SSD1306 OLED display device
device = ssd1306(serial)

# Setup SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

n = 625

def analogInput(channel_number):
    adc = spi.xfer2([0x01, 0xA0, 0x00])  # for PPG
    data = ((adc[1] & 15) << 8) + adc[2]
    return data



try:
    while True:
        ypoints = [0] * n
        xpoints = [0] * n
        
        for i in range(n):
            xpoints[i] = i
            ypoints[i] = round((analogInput(0) * (3.30 / 4095.0)), 3)
            time.sleep(0.008) 
            #print(i+1, ypoints[i])
        try: 
            # Prediction
            start_time = time.time()
            signal = np.array(ypoints)
            feat = extract(signal)
            pred = ad.predict(feat)
            if pred == 1:print_oled(0)
            else:
                cl_pred = clf.predict(feat)
                print_oled(cl_pred+1)
            end_time = time.time()
            print("Latency:", (end_time - start_time))
        except ValueError as e:
            if "cannot convert float NaN to integer" in str(e):
                pred = 1  # Consider this as y_pred = 1
            else:
                raise e  # Raise the exception if it's not the specific error
except KeyboardInterrupt:
    print("Program stopped by user")
