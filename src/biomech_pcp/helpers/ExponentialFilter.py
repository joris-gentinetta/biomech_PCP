# ExponentialFilter.py implements exponential smoothing

import numpy as np


class ExponentialFilterArr:
    # this creates an array of Bessel filters for processing multiple signals
    def __init__(self, numChannels, smoothingFactor, defaultValue=1):
        self.numChannels = numChannels

        if type(smoothingFactor) in [int, float]:
            self.alpha = [smoothingFactor] * numChannels
        elif len(smoothingFactor) == 1:  # a 2D array or a single number, repeat them
            self.alpha = numChannels * smoothingFactor
        else:
            raise ValueError(
                f"Invalid smoothing factor {smoothingFactor} provided. Must be an integer or a list of integers."
            )

        if type(defaultValue) is int:
            self.defaultValue = defaultValue
        else:
            raise ValueError(
                f"Invalid default value {defaultValue} provided. Must be an integer."
            )

        assert len(self.alpha) == numChannels, (
            f"smoothingFactor input size ({len(self.alpha)}) does not match number of channels ({numChannels})"
        )
        assert all([0 <= a <= 1 for a in self.alpha]), (
            "All smoothing factors must be between 0 and 1"
        )

        self.filters = defaultValue * np.ones((numChannels, 1))

    def resetFilters(self):
        self.filters = self.defaultValue * np.ones((self.numChannels, 1))

    def resetFilterByIndex(self, channel):
        if channel >= self.numChannels or channel < 0:
            raise ValueError(
                f"Invalid channel index {channel} provided. Must be between 0 and {self.numChannels - 1}"
            )

        self.filters[channel] = self.defaultValue

    def filter(self, sig):
        self.filters = np.multiply(self.alpha, sig) + np.multiply(
            (1.0 - np.asarray(self.alpha)), self.filters
        )

        return self.filters.copy()

    def filterByIndex(self, sig, channel):
        if channel >= self.numChannels or channel < 0:
            raise ValueError(
                f"Invalid channel index {channel} provided. Must be between 0 and {self.numChannels - 1}"
            )

        self.filters[channel] = (
            self.alpha[channel] * sig
            + (1 - self.alpha[channel]) * self.filters[channel]
        )

        return self.filters[channel]
