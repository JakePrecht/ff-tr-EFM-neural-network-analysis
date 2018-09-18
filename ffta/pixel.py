"""pixel.py: Contains pixel class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2018, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import logging
import numpy as np
from scipy import signal as sps
from scipy import optimize as spo
from scipy import interpolate as spi

from ffta.utils import noise
from ffta.utils import cwavelet
from ffta.utils import parab
from ffta.utils import fitting
from ffta.utils import dwavelet
from ffta.utils import load
#import nitime.timeseries as ts
#from nitime.analysis.spectral import MorletWaveletAnalyzer, SpectralAnalyzer

from matplotlib import pyplot as plt

from numba import autojit
#I commented out below line because it was causing probelms and I don't use peak detect-Jake
#from utils.peakdetect import get_peaks

class Pixel(object):
    """
    Signal Processing to Extract Time-to-First-Peak.

    Extracts Time-to-First-Peak (tFP) from digitized Fast-Free Time-Resolved
    Electrostatic Force Microscopy (FF-trEFM) signals [1-2]. It includes a few
    types of frequency analysis:
        a) Hilbert Transform
        b) Wavelet Transform
        c) Hilbert-Huang Transform (EMD)

    Parameters
    ----------
    signal_array : (n_points, n_signals) array_like
        2D real-valued signal array, corresponds to a pixel.
    params : dict
        Includes parameters for processing. The list of parameters is:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)
        drive_freq = float (in Hz)

        roi = float (in seconds)
        window = string (see documentation of scipy.signal.get_window)
        bandpass_filter = int (0: no filtering, 1: FIR filter, 2: IIR filter)
        filter_bandwidth = float (default: 5kHz)
        n_taps = integer (default: 1799)
        wavelet_analysis = bool (0: Hilbert method, 1: Wavelet Method)
        wavelet_parameter = int (default: 5)
        recombination = bool (0: Data are for Charging up, 1: Recombination)
        fit_phase = bool (0: fit to frequency, 1: fit to phase)

    Attributes
    ----------
    n_points : int
        Number of points in a signal.
    n_signals : int
        Number of signals to be averaged in a pixel.
    signal_array : (n_signals, n_points) array_like
        Array that contains original signals.
    signal : (n_points,) array_like
        Signal after phase-locking and averaging.
    tidx : int
        Index of trigger in time-domain.
    phase : (n_points,) array_like
        Phase of the signal, only calculated with Hilbert Transform method.
    cwt_matrix : (n_widths, n_points) array_like
        Wavelet matrix for continuous wavelet transform.
    inst_freq : (n_points,) array_like
        Instantenous frequency of the signal.
    tfp : float
        Time from trigger to first-peak, in seconds.
    shift : float
        Frequency shift from trigger to first-peak, in Hz.

    Methods
    -------
    analyze()
        Analyzes signals and returns tfp, shift and inst_freq.

    See Also
    --------
    line: Line processing for FF-trEFM data.
    simulate: Simulation for synthetic FF-trEFM data.
    scipy.signal.get_window: Windows for signal processing.

    Notes
    -----
    Frequency shift from wavelet analysis is not in Hertz. It should be used
    with caution.

    analyze() does not raise an exception if there is one, however it logs the
    exception if logging is turned on. This is implemented this way to avoid
    crashing in the case of exception when this method is called from C API.

    References
    ----------
    .. [1] Giridharagopal R, Rayermann GE, Shao G, et al. Submicrosecond time
       resolution atomic force microscopy for probing nanoscale dynamics.
       Nano Lett. 2012;12(2):893-8.
       [2] Karatay D, Harrison JA, et al. Fast time-resolved electrostatic
       force microscopy: Achieving sub-cycle time resolution. Rev Sci Inst.
       2016;87(5):053702

    Examples
    --------
    >>> from ffta import pixel, utils
    >>>
    >>> signal_file = '../data/SW_0000.ibw'
    >>> params_file = '../data/parameters.cfg'
    >>>
    >>> signal_array = utils.load.signal(signal_file)
    >>> n_pixels, params = utils.load.configuration(params_file)
    >>>
    >>> p = pixel.Pixel(signal_array, params)
    >>> tfp, shift, inst_freq = p.analyze()

    """

    def __init__(self, signal_array, params, fit=True):
        
        # Create parameter attributes for optional parameters.
        # They will be overwritten by following for loop if they exist.
        self.n_taps = 1499
        self.Q = 500
        self.filter_bandwidth = 5000
        self.wavelet_analysis = False
        self.wavelet_parameter = 5
        self.recombination = False
        self.phase_fitting = False
        self.EMD_analysis = False

        # Assign the fit parameter.
        self.fit = fit
        self.fit_form = 'PRODUCT'

        # Read parameter attributes from parameters dictionary.
        for key, value in params.items():

            setattr(self, key, value)

        # Assign values from inputs.
        self.signal_array = signal_array
        self.tidx = int(self.trigger * self.sampling_rate)
        
        # Set dimensions correctly
        # Three cases: 1) 2D (has many averages) 2) 1D (but set as 1xN) and 3) True 1D
        if len(signal_array.shape) == 2:
            if signal_array.shape[0] != 1:
                self.n_signals, self.n_points = signal_array.shape
                self._n_points_orig = signal_array.shape[1]
            else:
                self.n_signals = 1
                self.signal_array = self.signal_array[0,:]
                self.n_points = self.signal_array.shape[0]
                self._n_points_orig = self.signal_array.shape[0]
        else:
            self.n_signals = 1
            self.n_points = signal_array.shape[0]
            self._n_points_orig = signal_array.shape[0]

        # Keep the original values for restoring the signal properties.
        self._tidx_orig = self.tidx
        self.tidx_orig = self.tidx

        # Initialize attributes that are going to be assigned later.
        self.signal = None
        self.phase = None
        self.inst_freq = None
        self.tfp = None
        self.shift = None
        self.cwt_matrix = None

        return

    def clear_filter_flags(self):
        """Removes flags from parameters for setting filters"""
        
#        self.window = 0
        self.bandpass_filter = 0
        
        return

    def remove_dc(self):
        """Removes DC components from signals."""

        if self.n_signals != 1:
        
            for i in range(self.n_signals):
            
                self.signal_array[i] -= self.signal_array[i,:].mean()

        return

    def phase_lock(self):
        """Phase-locks signals in the signal array. This also cuts signals."""

        # Phase-lock signals.
        self.signal_array, self.tidx = noise.phase_lock(self.signal_array, self.tidx,
                                                        np.ceil(self.sampling_rate / self.drive_freq))

        # Update number of points after phase-locking.
        self.n_points = self.signal_array.shape[0]

        return

    def average(self):
        """Averages signals."""

        if self.n_signals != 1: # if not multi-signal, don't average
            self.signal = self.signal_array.mean(axis=0)
            
        else:
            self.signal = self.signal_array
            
        return

    def check_drive_freq(self):
        """Calculates drive frequency of averaged signals, and check against
           the given drive frequency."""

        n_fft = 2 ** int(np.log2(self.tidx))  # For FFT, power of 2.
        dfreq = self.sampling_rate / n_fft  # Frequency separation.

        # Calculate drive frequency from maximum power of the FFT spectrum.
        signal = self.signal[:n_fft]
        fft_amplitude = np.abs(np.fft.rfft(signal))
        drive_freq = fft_amplitude.argmax() * dfreq

        # Difference between given and calculated drive frequencies.
        difference = np.abs(drive_freq - self.drive_freq)

        # If difference is too big, reassign. Otherwise, continue. != 0 for accidental DC errors
        if difference >= dfreq and drive_freq != 0:

            self.drive_freq = drive_freq

        return

    def apply_window(self):
        """Applies the window given in parameters."""
        
        self.signal *= sps.get_window(self.window, self.n_points)

        return

    def dwt_denoise(self):
        """Uses DWT to denoise the signal prior to processing."""

        rate = self.sampling_rate
        lpf = self.drive_freq * 0.1
        self.signal, _, _ = dwavelet.dwt_denoise(self.signal,lpf,rate/2,rate)

    def fir_filter(self):

        """Filters signal with a FIR bandpass filter."""
        # Calculate bandpass region from given parameters.

        nyq_rate = 0.5 * self.sampling_rate
        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / nyq_rate
        freq_high = (self.drive_freq + bw_half) / nyq_rate

        band = [freq_low, freq_high]

        # Create taps using window method.
        try:
            taps = sps.firwin(int(self.n_taps), band, pass_zero=False,
                          window='blackman')
        except:
            print('band=', band)
            print('nyq=',nyq_rate)
            print('drive=',self.drive_freq)

        self.signal = sps.fftconvolve(self.signal, taps, mode='same')

        # Shifts trigger due to causal nature of FIR filter
        self.tidx -= (self.n_taps - 1) / 2

        return

    def iir_filter(self):
        """Filters signal with two Butterworth filters (one lowpass,
        one highpass) using filtfilt. This method has linear phase and no
        time delay. Do not use for production."""

        # Calculate bandpass region from given parameters.
        nyq_rate = 0.5 * self.sampling_rate
        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / nyq_rate
        freq_high = (self.drive_freq + bw_half) / nyq_rate

        # Do a high-pass filtfilt operation.
        b, a = sps.butter(9, freq_low, btype='high')
        self.signal = sps.filtfilt(b, a, self.signal)

        # Do a low-pass filtfilt operation.
        b, a = sps.butter(9, freq_high, btype='low')
        self.signal = sps.filtfilt(b, a, self.signal)

        return

    def hilbert_transform(self):
        """Gets the analytical signal doing a Hilbert transform."""

        self.signal = sps.hilbert(self.signal)

        return

    def calculate_phase(self, correct_slope=True):
        """Gets the phase of the signal and correct the slope by removing
        the drive phase."""

        # Unwrap the phase.
        self.phase = np.unwrap(np.angle(self.signal))

        if correct_slope:

            # Remove the drive from phase.
            #self.phase -= (2 * np.pi * self.drive_freq *
            #               np.arange(self.n_points) / self.sampling_rate)

            # A curve fit on the initial part to make sure that it worked.
            start = int(0.3 * self.tidx)
            end = int(0.7 * self.tidx)
            fit = self.phase[start:end]

            xfit = np.polyfit(np.arange(start, end), fit, 1)

            # Remove the fit from phase.
            self.phase -= (xfit[0] * np.arange(self.n_points)) + xfit[1]

        return

    def calculate_inst_freq(self):
        """Calculates the first derivative of the phase using Savitzky-Golay
        filter."""

        dtime = 1 / self.sampling_rate  # Time step.

        # Do a Savitzky-Golay smoothing derivative
        # using 5 point 1st order polynomial.
        self.inst_freq = sps.savgol_filter(self.phase, 5, 1, deriv=1,
                                           delta=dtime)

        # Bring trigger to zero.
        self.tidx = int(self.tidx)
        self.inst_freq -= self.inst_freq[self.tidx]

        return

    def calculate_amplitude(self):
        """Calculates the amplitude of the analytic signal. Uses pre-filter
        signal to do this."""
#       
        if self.n_signals != 1:
            self.signal_orig = self.signal_array.mean(axis=0)
        
        self.signal_orig = sps.hilbert(self.signal_orig)
        self.amp = np.abs(self.signal_orig)

        return

    def find_minimum(self):
        """Finds when the minimum of instantenous frequency happens."""

        # Cut the signal into region of interest.
        ridx = int(self.roi * self.sampling_rate)
        cut = self.inst_freq[self.tidx:(self.tidx + ridx)]

        # Define a spline to be used in finding minimum.
        x = np.arange(ridx)
        y = cut

        func = spi.UnivariateSpline(x, y, k=4, ext=3)

        # Find the minimum of the spline using TNC method.
        res = spo.minimize(func, cut.argmin(),
                           method='TNC', bounds=((0, ridx),))
        idx = res.x[0]

        # Do index to time conversion and find shift.
        self.tfp = idx / self.sampling_rate
        self.shift = func(0) - func(idx)

        return

    def fit_freq_product(self):
        """Fits the frequency shift to an approximate functional form using
        an analytical fit with bounded values."""

        # Calculate the region of interest and if filtered move the fit index.
        ridx = int(self.roi * self.sampling_rate)

        fidx = self.tidx

        # Make sure cut starts from 0 and never goes over.
        cut = self.inst_freq[fidx:(fidx + ridx)] - self.inst_freq[fidx]
        t = np.arange(cut.shape[0]) / self.sampling_rate

        # Fit the cut to the model.
        popt = fitting.fit_bounded_product(self.Q, self.drive_freq, t, cut)

        #A, tau1, tau2 = popt
        A, tau1, tau2 = popt

        # Analytical minimum of the fit.
        #self.tfp = tau2 * np.log((tau1 + tau2) / tau2)
        #self.shift = -A * np.exp(-self.tfp / tau1) * np.expm1(-self.tfp / tau2)

        # For diagnostic purposes.
        self.cut = cut
        self.popt = popt
        self.best_fit = -A*(np.exp(-t/tau1)-1)*np.exp(-t/tau2)

        self.tfp = np.argmin(self.best_fit)/self.sampling_rate
        self.shift = np.min(self.best_fit)
        
        self.rms = np.sqrt(np.mean(np.square(self.best_fit - cut)))

        return

    def fit_freq_sum(self):
        """Fits the frequency shift to an approximate functional form using
        an analytical fit with bounded values."""

        # Calculate the region of interest and if filtered move the fit index.
        ridx = int(self.roi * self.sampling_rate)
        fidx = self.tidx

        # Make sure cut starts from 0 and never goes over.
        cut = self.inst_freq[fidx:(fidx + ridx)] - self.inst_freq[fidx]

        t = np.arange(cut.shape[0]) / self.sampling_rate

        # Fit the cut to the model.
        popt = fitting.fit_bounded_sum(self.Q, self.drive_freq, t, cut)
        A1, A2, tau1, tau2 = popt

        # For diagnostic purposes.
        self.cut = cut
        self.popt = popt
        self.best_fit = A1*(np.exp(-t/tau1)-1) - A2*np.exp(-t/tau2)

        self.tfp = np.argmin(self.best_fit)/self.sampling_rate
        self.shift = np.min(self.best_fit)

        return

    def fit_freq_exp(self):
        """Fits the frequency shift to a single exponential in the case where
        there is no return to 0 Hz offset (if drive is cut)."""

        ridx = int(self.roi * self.sampling_rate)
        fidx = self.tidx

        # Make sure cut starts from 0 and never goes over.
        cut = self.inst_freq[fidx:(fidx + ridx)] - self.inst_freq[fidx]

        t = np.arange(cut.shape[0]) / self.sampling_rate

        # Fit the cut to the model.
        popt  = fitting.fit_bounded_exp(t, cut)

        # For diagnostics
        A, y0, tau = popt
        self.cut = cut
        self.popt = popt
        self.best_fit = A*(np.exp(-t/tau)) + y0

        self.shift = A
        self.tfp = tau

        return

    def fit_phase(self):
        """Fits the phase to an approximate functional form using an
        analytical fit with bounded values."""

        # Calculate the region of interest and if filtered move the fit index.
        ridx = int(self.roi * self.sampling_rate)

        fidx = self.tidx

        # Make sure cut starts from 0 and never goes over.
        # -1 on cut is because of sign error in generating phase
        cut = -1*(self.phase[fidx:(fidx + ridx)] - self.phase[fidx])
        t = np.arange(cut.shape[0]) / self.sampling_rate

        # Fit the cut to the model.
        popt = fitting.fit_bounded_phase(self.Q, self.drive_freq, t, cut)

        A, tau1, tau2 = popt

        # Analytical minimum of the fit.
        self.tfp = tau2 * np.log((tau1 + tau2) / tau2)
        self.shift = A * np.exp(-self.tfp / tau1) * np.expm1(-self.tfp / tau2)

        # For diagnostic purposes.
        postfactor = (tau2 / (tau1 + tau2)) * np.exp(-t / tau2) - 1

        self.cut = cut
        self.popt = popt
        self.best_fit = -A * np.exp(-t / tau1) * np.expm1(-t / tau2 )
        self.best_phase = A * tau1 * np.exp(-t / tau1)*postfactor + A * tau1 * (1 - tau2/(tau1 + tau2))

        return

    def restore_signal(self):
        """Restores the signal length and position of trigger to original
        values."""

        # Difference between current and original values.
        d_trig = int(self._tidx_orig - self.tidx)
        d_points = int(self._n_points_orig - self.n_points)

        # Check if the signal length can accomodate the shift or not.
        if d_trig >= d_points:

            # Pad from left and set the original length.
            self.inst_freq = np.pad(self.inst_freq, (d_trig, 0), 'edge')
            self.inst_freq = self.inst_freq[:self._n_points_orig]

        else:

            # Calculate how many points is needed for padding from right.
            pad_right = d_points - d_trig
            self.inst_freq = np.pad(self.inst_freq, (d_trig, pad_right),
                                    'edge')

        # Set the public variables back to original values.
        self.tidx = self._tidx_orig
        self.n_points = self._n_points_orig

        return

    @autojit
    def __get_cwt__(self):
        """Generates the CWT using Morlet wavelet. Returns a 2D Matrix."""

        w0 = self.wavelet_parameter
        wavelet_increment = 0.5  # Reducing this has little benefit.

        cwt_scale = ((w0 + np.sqrt(2 + w0 ** 2)) /
                     (4 * np.pi * self.drive_freq / self.sampling_rate))

        widths = np.arange(cwt_scale * 0.5, cwt_scale * 1.5,
                           wavelet_increment)

        cwt_matrix = cwavelet.cwt(self.signal, dt=1, scales=widths, p=w0)
        self.cwt_matrix = np.abs(cwt_matrix)

        return w0, wavelet_increment, cwt_scale

    def calculate_cwt_freq_old(self):
        """Conventional ridge-finding spectrogram approach"""
        w0, wavelet_increment, cwt_scale = self.__get_cwt__()

        _, n_points = np.shape(self.cwt_matrix)
        inst_freq = np.empty(n_points)

        for i in xrange(n_points):

            cut = self.cwt_matrix[:, i]
            inst_freq[i], _ = parab.fit(cut, np.argmax(cut))

        inst_freq = (inst_freq * wavelet_increment + 0.9 * cwt_scale)
        inst_freq = ((w0 + np.sqrt(2 + w0 ** 2)) /
                     (4 * np.pi * inst_freq[:] / self.sampling_rate))

        self.inst_freq = inst_freq - inst_freq[self.tidx]

        return

    def calculate_cwt_freq(self):
        """Neuroimaging package Morlet analyzer version. Does not yield
            a spectrogram at the end, though, just analytic signal.
        """

        # Generate necessary tools for wavelet transform.
        t1 = ts.TimeSeries(self.signal,sampling_rate=self.sampling_rate)
        self.wavelet = MorletWaveletAnalyzer(t1, freqs=self.drive_freq,
                                             sd_rel=(self.filter_bandwidth /
                                             self.drive_freq))
              
        phase = np.unwrap(self.wavelet.phase.data)

        self.inst_freq = sps.savgol_filter(phase, int(self.n_taps),
                                           1, deriv=1, delta=1e-7)

        return

    def EMD_signal(self):
        """Uses Empirical Mode Decomposition to denoise and analyze the
        input signal."""

        signal = self.signal
        imfs = []

        tt = np.arange(0, len(signal), 1)

        x1 = signal
        sd = 1

        # loop controls how many EMD modes
        modes = 1
        for i in xrange(modes):

            # Continuously adjusts signal until offset within 0.1 f.o.m.
            while sd > .1:

                maxpeaks, minpeaks = get_peaks(x1)

                fmax = spi.UnivariateSpline(maxpeaks, x1[maxpeaks], k=3)
                fmin = spi.UnivariateSpline(minpeaks, x1[minpeaks], k=3)
                fmax.set_smoothing_factor(0)
                fmin.set_smoothing_factor(0)

                smean = (fmax(tt) + fmin(tt)) / 2.0

                x2 = x1 - smean

                # figure of merit for EMD decomposition
                sd = np.sum((x1 - x2)**2) / np.sum(x1**2)

                x1 = x2

            imfs.append(x1)
            signal = signal - x1

        self.signal = imfs[0]

        return

    def EMD_inst_freq(self):
        """Calculates instantaneous frequency from EMD Mode 0."""

        savgolc = int(self.n_taps)

        self.inst_freq = sps.savgol_filter(self.phase, savgolc, 1, deriv=1,
                                           delta=1/self.sampling_rate)
        self.inst_freq = self.inst_freq/(2*np.pi) - self.drive_freq

        # Restores length
        self.inst_freq = np.pad(self.inst_freq, ((savgolc-1)/2,0),'constant')
        self.inst_freq = self.inst_freq[:len(self.signal)]

        # Bring trigger to zero.
        self.tidx = int(self.tidx)
        self.inst_freq -= self.inst_freq[self.tidx]

        return

    def plot(self, newplot=True, c1='r', c2='g'):
        """ Quick visualization of best_fit and cut."""
        
        if newplot:
            plt.figure()
        
        plt.plot(self.cut, c1+'-')
        plt.plot(self.best_fit, c2+'--')
                
        return

    def analyze(self):
        """
        Analyzes the pixel with the given method.

        Returns
        -------
        tfp : float
            Time from trigger to first-peak, in seconds.
        shift : float
            Frequency shift from trigger to first-peak, in Hz.
        inst_freq : (n_points,) array_like
            Instantenous frequency of the signal.

        Notes
        -----
        It does not raise an exception if there is one, however it logs
        the exception if logging is turned on. This is implemented this way to
        avoid crashing in the case of exception when this method is called
        from C API.

        """
        #logging.basicConfig(filename=r'C:\Users\Asylum User\Documents\ffta\ffta\error.log', level=logging.DEBUG)
        
        try:
            # Remove DC component, first.
            self.remove_dc()

            # Phase-lock signals.
            #self.phase_lock()

            # Average signals.
            self.average()

            # Remove DC component again, introduced by phase-locking.
            self.remove_dc()

            # Check the drive frequency.
            self.check_drive_freq()

            # DWT Denoise
            #self.dwt_denoise()

            if self.EMD_analysis:

                # Calculate signal by Hilbert-Huang transform.
                self.EMD_signal()

                # Get the analytical signal doing a Hilbert transform.
                self.hilbert_transform()

                # Calculate the phase from analytic signal.
                self.calculate_phase()

                # Calculate the instantaneous frequency
                self.EMD_inst_freq()

            elif self.wavelet_analysis:

                # Calculate instantenous frequency using wavelet transform.
                self.calculate_cwt_freq()

            else:
                # Hilbert transform method

                # Apply window.
                if self.window != 0:
                    
                    self.apply_window()

                # Filter the signal with a filter, if wanted.
                if self.bandpass_filter == 1:

                    self.fir_filter()

                elif self.bandpass_filter == 2:

                    self.iir_filter()

                # Get the analytical signal doing a Hilbert transform.
                self.hilbert_transform()

                # Calculate the phase from analytic signal.
                self.calculate_phase()

                # Calculate instantenous frequency.
                self.calculate_inst_freq()

            # If it's a recombination image invert it to find minimum.
            if self.recombination:

                self.inst_freq = self.inst_freq * -1

            # Find where the minimum is.
            if self.fit:

                if self.phase_fitting:

                    self.fit_phase()

                else:
                   
                    if self.fit_form == 'PRODUCT':

                        self.fit_freq_product()

                    elif self.fit_form == 'SUM':

                        self.fit_freq_sum()

                    elif self.fit_form == "EXP":
                        
                        self.fit_freq_exp()

            else:

                self.find_minimum()

            # Restore the length.
            self.restore_signal()

        # If caught any exception, set everything to zero and log it.
        except Exception as exception:
            self.tfp = 0
            self.shift = 0
            self.inst_freq = np.zeros(self._n_points_orig)

            logging.exception(exception, exc_info=True)

        if self.phase_fitting:

            return self.tfp, self.shift, self.phase

        else:

            if self.tfp == 5e-7:
                return np.NaN, np.NaN, self.inst_freq
                
            return self.tfp, self.shift, self.inst_freq
        