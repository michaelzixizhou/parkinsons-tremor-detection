def detect_peak_frequency(window , fs=100, low_freq=3, high_freq=8, ar_order=6):
    """
    Detects peak frequency using an autoregressive model. 

    Args: 
        window: Windowed segment of accelerometer data.
        fs: Sampling frequency.
        low_freq: Lower frequency bound for tremor detection.
        high_freq: Upper frequency bound for tremor detection.
        ar_order: Order of the autoregressive model. 

    Return:
        Peak frequency if within tremor range, otherwise 1.
    """

    pass
