
import tensorflow as tf


def tf_wav2mfcc(signals, sr=16000, frame_length=1024, frame_step=256, fft_length=1024):
	"""
	This function converts the wav file of sampleing rate sr into MFCCs(Mel-Frequency Cepstral Coefficients)
	It is highly optimised and can be moved over to GPU for parallelization.
	Args:
		signals: waveform
		sr: samping rate defaults is 16000Hz
		frame_length: each frame length default is 1024
		frame_step: each frame step to take in FT default is 256
		fft_length: FFT len default is 1024
	returns:
		log mel spectrograms
	"""
	stfts = tf.contrib.signal.stft(signals, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
	magnitude_spectrograms = tf.abs(stfts)
	num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
	lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000, 80
	linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz, upper_edge_hertz)
	mel_spectrograms = tf.tensordot( magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
	log_offset = 1e-6
	log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
	log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, axis=3)
	return log_mel_spectrograms