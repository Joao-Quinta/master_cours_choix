import pandas as pd
import numpy as np
import biosppy.signals.tools as pt
import biosppy.signals.ecg as ecg
import scipy.stats as sstats
import scipy.signal as ssig
import itertools
import collections


def modulateSignal(signal, carrier, factor=1):
    """
        Args:
            signal: array, [N]
                the signal to modulate
            carrier: array, [N]
                the carrier signal to use for the modulation
        Returns: array, [N]
            representing the amplitude modulated signal using the given carrier and factor
    """
    return signal+signal*carrier*factor


# Simple mapping to classes
def bin_signal(signal, bins, balanceBins=False):
    """
        Args:
            signal: array, [N]
                the signal to bin into classes
            bins: int, or array [bins]
                if int - how many classes to split it to
                if array - list of bin left edges
            balanceBins: boolean, optional
                if False, bin edges will be uniformly distributed on the range of the signal,
                 else will optimize bin edges such that each bin contains roughly equal number of samples
        Returns: array, [N]
            representing the class (bin) that each sample of the signal belongs to
    """
    def assignClasses(sig, bedg):
        signalClass = np.zeros(np.shape(signal))
        for i in range(1,len(bedg)):
             signalClass[sig>=bedg[i]] = i

        return signalClass

    
    if isinstance(bins, collections.Iterable):
        #TODO check dimensions of bins

        sigMin = min(signal)
        if bins[0]!=sigMin:
            bins.insert(0,sigMin)
        binEdges = bins

    elif not balanceBins:
        sigMin = min(signal)
        sigMax = max(signal)
        binEdges = [(i/bins)*(sigMax-sigMin)+sigMin for i in range(bins)]

    else:
        def split_metric(classSizes):
            me = np.mean(classSizes)
            metric = 0
            for sz in classSizes:
                metric = metric+(sz-me)**2

            return metric
        
        def eval_edge_balance(sig, bedg):
            clSizes = [0]*len(bedg)
            for i in range(len(bedg)-1):
                clSizes[i] = ((sig>=bedg[i]) & (sig<bedg[i+1])).sum()
            clSizes[-1] = len(sig) - sum(clSizes) # the rest is the last class

            return split_metric(clSizes)


        sigSorted = np.sort(signal)

        binSize = int(len(sigSorted)/bins)
       # binEdgeIx = min(*i) for i in zip([j*binSize for j in range(bins)], [len(sigSorted)-1]*bins)
        binEdgesIx = np.minimum([j*binSize for j in range(bins)], [len(sigSorted)-1]*bins)
        # in case the bin edge is not good (lies within group with same values) adjust
        sigSorted_d = np.diff(sigSorted)
        binEdgesAltIx = [0]*bins
        for i in range(1,bins): # first edge is always at 0
            if sigSorted_d[binEdgesIx[i]]==0 or (binEdgesIx[i]>0 and sigSorted_d[binEdgesIx[i]-1] == 0):
                for ix in range(binEdgesIx[i],len(sigSorted_d),1):
                    if sigSorted_d[ix] != 0:
                        ix=ix+1
                        break
                binEdgesAltIx[i] = ix

        # if there is ambiguity in edges
        if sum(binEdgesAltIx)>0:
            # generate combination masks of edges
            binEdgesOptionMasks = [list(i) for i in itertools.product([0, 1], repeat=bins)]
            binEdgesOptionMasks = binEdgesOptionMasks[0:int(len(binEdgesOptionMasks)/2)] # first edge is always 0 so we can exclude all masks that do not conform
            # compare the options and find the best
            bestEdges = [0]*bins
            bestMetr = np.Inf
            for optMask in binEdgesOptionMasks:
                optEdges = [0]*bins
                for i in range(len(optMask)):
                    if optMask[i]>0 and binEdgesAltIx[i]>0:
                        optEdges[i] = sigSorted[binEdgesAltIx[i]]
                    else:
                        optEdges[i] = sigSorted[binEdgesIx[i]]
                optMetr = eval_edge_balance(signal, optEdges)
                if(optMetr<bestMetr):
                    bestEdges = optEdges
                    bestMetr = optMetr

            binEdges = bestEdges

        else:
            binEdges = [sigSorted[ix] for ix in binEdgesIx]



    return assignClasses(signal, binEdges)

def filter_ecg(signal, sampling_rate):
    order = int(0.3 * sampling_rate)
    filtered, _ ,_  = pt.filter_signal(
        signal=signal,
        ftype='FIR',
        band='bandpass',
        order=order,
        frequency=[3, 45],
        sampling_rate=sampling_rate)
    return filtered


def filter_eda(signal, sampling_rate):
    # filter signal
    aux, _, _ = pt.filter_signal(
        signal=signal,
        ftype='butter',
        band='lowpass',
        order=4,
        frequency=5,
        sampling_rate=sampling_rate)

    # smooth
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = pt.smoother(
        signal=aux,
        kernel='boxzen',
        size=sm_size,
        mirror=True)
    return filtered


def filter_rsp(signal, sampling_rate):
    filtered, _, _ = pt.filter_signal(
        signal=signal,
        ftype='butter',
        band='bandpass',
        order=2,
        frequency=[0.1, 1],  # 1 is choosen as sudy on sports indicates that most respiration do not exeed 60 rpm
        sampling_rate=sampling_rate)
    return filtered


def extractFeatures(signal, sampling_rate, features, win_duration, win_delay=0, step=1):
    """
        Args:
            signal: array, [N]
                the signal to extract features from
            sampling_rate: int
                the sampling rate of the signal
            features: list
                list of features to extract ['min', 'max', 'mean', 'median', 'mode', 'variance']
            win_duration: int
                window duration in seconds
            win_delay: int
                window delay in seconds. If 0 then the window is starts on the current sample
            step: int
                step size in seconds
        Returns: pandas DataFrame, [N*len(features)]
            containing the features of the signal using a rolling window with the provided parameters
            """

    if not features and not all(isinstance(feature, basestring) for feature in features):
        return signal
    
    supportedFeatures = ['min', 'max', 'mean', 'median', 'mode', 'variance','sum']
    
    featfuncs = [np.mean]*len(features)
    for i,feature in enumerate(features):
        if feature == 'median':
            featfuncs[i] = np.nanmedian
        elif feature == 'min':
            featfuncs[i] = np.nanmin
        elif feature == 'max':
            featfuncs[i]= np.nanmax
        elif feature == 'mean':
            featfuncs[i] = np.nanmean
        elif feature == 'mode':
            featfuncs[i] = lambda x: sstats.mode(x)[0]
        elif feature == 'variance':
            featfuncs[i] = np.nanvar
        elif feature == 'sum':
                featfuncs[i] = np.nansum

    win_length = win_duration * sampling_rate

    win_offset = win_delay * sampling_rate
    step_samples = step*sampling_rate

    if len(signal)<win_length+win_offset:
        print('signal length is smaller than window, aborting!')
        return signal     
    

    # extract features into the columns
    win_starts = np.arange(win_offset,len(signal)+win_offset,step_samples)
    # win_starts = win_starts[(win_starts>=0) & (win_starts<=len(signal))]
    # init columns
    signalFeat = pd.DataFrame()
    for feature in features:
        if feature in supportedFeatures:
            signalFeat[feature] = np.zeros_like(win_starts)
    # signalFeat = signalFeat.reset_index()
    # print(win_starts)
    for i,win_start in enumerate(win_starts):
        win_end = win_start+win_length
        win_start = min(len(signal),max(0,win_start))
        win_end = min(len(signal),max(0,win_end))
    
        for f,feature in enumerate(features):
            if win_end-win_start < 1:
                signalFeat.loc[i,feature] = 0
            else:
                signalFeat.loc[i,feature] = featfuncs[f](signal[win_start:win_end])


    return signalFeat


def _get_heart_rate(beats=None, sampling_rate=1000., smooth=False, size=3):
    """Compute instantaneous heart rate from an array of beat indices.
    Parameters
    ----------
    beats : array
        Beat location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    smooth : bool, optional
        If True, perform smoothing on the resulting heart rate.
    size : int, optional
        Size of smoothing window; ignored if `smooth` is False.
    Returns
    -------
    index : array (2*N)
        Heart rate start (first col) and stop (last col) location indices.
    heart_rate : array
        Instantaneous heart rate (bpm).
    """

    # check inputs
    if beats is None:
        raise TypeError("Please specify the input beat indices.")

    if len(beats) < 2:
        raise ValueError("Not enough beats to compute heart rate.")

    # compute heart rate
    idx = np.array([beats[:-1], beats[1:]])  # start and stop times of hr
    hr = sampling_rate * (60. / np.diff(beats))

    # Herre some physiological limits (>40 <200 bps) were applied in biosppy
    # I do not want that here -> removed

    # smooth with moving average
    if smooth and (len(hr) > 1):
        hr, _ = pt.smoother(signal=hr, kernel='boxcar', size=size, mirror=True)

    return idx, hr


def compute_hr_from_ecg(filtered_ecg, sampling_rate, smooth=True, size=3, times=None):
    # construct time vector is it is not given in parameter
    if times is None:
        length = len(filtered_ecg)
        T = (length - 1) / sampling_rate
        times = np.linspace(0, T, length, endpoint=True)

    elif len(times) != len(filtered_ecg):
        raise ValueError('Size mistmatch - ecg and times must have the same size')


    # segment
    rpeaks, = ecg.christov_segmenter(signal=filtered_ecg, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = ecg.correct_rpeaks(signal=filtered_ecg, rpeaks=rpeaks,
        sampling_rate=sampling_rate, tol=0.05)

    # compute heart rate
    hr_idx, hr = _get_heart_rate(beats=rpeaks, sampling_rate=sampling_rate,
        smooth=smooth, size=size)

    # get hr times
    ts_hr = times[hr_idx]

    return interpolate_hr(hr,ts_hr,times)


def interpolate_hr(hr, ts, new_ts, method='interp'):
    """Interpolates HR samples with start end time stamps
    Args:
        hr: array, N
            heart rate values
        ts: array, 2*N
            start and stop times of heart rate
        new_ts: array, M
            points to which HR should be interpolated
        method: string
            method used for interpolation, can be:
            - interp: uses numpy
            - flat (under dev.)
    Returns
        hr_new: heart rate interpolated at times new_ts
    """
    if method == 'interp':
        # Assumes that HR is at the middle of the interval
        ts = ts.mean(axis=0)
        new_hr = np.interp(new_ts, ts, hr)

    elif method == 'flat':
        raise NotImplementedError('method flat')

    return new_hr


def compute_HR_features(signals, sampling_rate, welch_win_duration=10):
    """
    Args:
        signals: arrray, [N*S]
            S heart rate signals sampled at fixed frequency with N samples
            Can also manage one dimensional array with one signal
        sampling_rate:
            fixed sampling rate
        welch_win_duration:
            duration in seconds of the segments used for welch computations
    Retuns: panda DataFrame [S*F]
        containing the F features for S signals
    """
    if signals.ndim == 1:
        signals = np.expand_dims(signals, axis=-1)

    df = pd.DataFrame()

    # Compute standard statistics
    df['mean'] = np.mean(signals, axis=0)
    df['mean_norm'] = np.mean(signals - signals[0], axis=0)
    df['HRV'] = np.std(signals, axis=0)

    # Compute low, medium, high and ration frequ
    welch_win_samples = welch_win_duration*sampling_rate
    if signals.shape[0] >= welch_win_samples:
        f, P = ssig.welch(signals.T, fs=sampling_rate, nperseg=welch_win_samples)
        df['LF'] = np.sum(P[:, f <= 0.15], axis=1)
        df['HF'] = np.sum(P[:, (0.15 < f) & (f < 0.4)], axis=1)
    else:  # not enough sample for frequencies computations
        df['LF'] = np.NaN
        df['HF'] = np.NaN
    df['LF/HF'] = df['LF'] / df['HF']

    # TODO: Compute multiscale entropy

    return df


def compute_EDA_features(signals, sampling_rate):
    """
    Args:
        signals: arrray, [N*S]
            S EDA signals sampled at fixed frequency with N samples
            Can also manage one dimensional array with one signal
        sampling_rate:
            fixed sampling rate
    Retuns: panda DataFrame [S*F]
        containing the F features for S signals
    """
    if signals.ndim == 1:
        signals = np.expand_dims(signals, axis=-1)

    df = pd.DataFrame()

    df['mean'] = np.mean(signals, axis=0)
    df['std'] = np.std(signals, axis=0)
    df['quartile3'] = np.quantile(signals, 0.75, axis=0)
    df['median'] = np.quantile(signals, 0.5, axis=0)
    df['quartile1'] = np.quantile(signals, 0.25, axis=0)

    norm_sig = signals - signals[0]
    df['mean_norm'] = np.mean(norm_sig, axis=0)
    df['quartile3_norm'] = np.quantile(norm_sig, 0.75, axis=0)
    df['median_norm'] = np.quantile(norm_sig, 0.5, axis=0)
    df['quartile1_norm'] = np.quantile(norm_sig, 0.25, axis=0)

    # Compute percentage of signal increase, not considering NaN values
    diff_sig = np.diff(signals, axis=0)
    diff_nan = np.isnan(diff_sig)
    diff_sig[diff_nan] = 0
    with np.errstate(invalid='ignore'):
        df['per_increase'] = np.sum(diff_sig > 0, axis=0) / np.sum(~diff_nan, axis=0)

    # TODO: include more complex stuff such as tonic / phasic separation and peak counts

    return df


def compute_Respiration_features(signals, sampling_rate, welch_win_duration=10):
    """
    Args:
        signals: arrray, [N*S]
            S respiration signals sampled at fixed frequency with N samples
            Can also manage one dimensional array with one signal
        sampling_rate:
            fixed sampling rate
        welch_win_duration:
            duration in seconds of the segments used for welch computations
    Retuns: panda DataFrame [S*F]
        containing the F features for S signals
    """
    if signals.ndim == 1:
        signals = np.expand_dims(signals, axis=-1)

    df = pd.DataFrame()

    # TODO: have common statistical moments for all/many signals ?
    df['mean'] = np.mean(signals, axis=0)
    df['std'] = np.std(signals, axis=0)
    df['kurtosis'] = sstat.kurtosis(signals, axis=0)
    df['skew'] = sstat.skew(signals, axis=0)

    # Compute frequency features
    welch_win_samples = welch_win_duration*sampling_rate
    if (signals.shape[0] < welch_win_samples) or np.any(np.isnan(signals)): # missing samples for frequency computation
        df['main_freq'] = np.NaN
    else:  
        f, P = ssig.welch(signals.T, fs=sampling_rate, nperseg=welch_win_samples)
        df['main_freq'] = f[np.argmax(P, axis=1)]


    # TODO: Compute frequency features (see TEAP)

    return df
