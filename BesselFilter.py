# "BesselFilter.py implements the Bessel filters

from scipy import signal
import numpy as np

class BesselFilterArr():
    # this creates an array of Bessel filters for processing multiple signals
    def __init__(self, numChannels, order, critFreqs, fs, filtType):
        self.numChannels = numChannels

        if filtType not in ['bandstop', 'lowpass', 'highpass']:
            raise ValueError(f"Invalid filter type {filtType} provided (options are 'lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’).")

        if type(critFreqs) is int:
            freqs = [critFreqs]*numChannels
        elif len(critFreqs) == 1: # a 2D array or a single number, repeat them
            freqs = numChannels*critFreqs
        elif len(critFreqs) == 2 and not numChannels == 2: # a 2 element array for a bandstop, repeat it
            freqs = [critFreqs for _ in range(numChannels)]
        else:
            freqs = critFreqs

        assert len(freqs) == numChannels, f'critFreqs input size ({len(freqs)}) does not match number of channels ({numChannels})'

        self.filters = {'sos': [], 'zi': []}
        for i in range(self.numChannels):
            filter_sos = signal.bessel(N=order, Wn=freqs[i], btype=filtType, output='sos', fs=fs, analog=False)
            
        # self.filters = {'sos': [], 'zi': []}
        # for _ in range(self.numChannels):
        #     filter_sos = signal.bessel(N=order, Wn=critFreqs, btype=filtType, output='sos', fs=fs, analog=False)
            filter_zi = signal.sosfilt_zi(filter_sos)

            self.filters['sos'].append(filter_sos)
            self.filters['zi'].append(filter_zi)

    def resetFilters(self):
        for i in range(self.numChannels):
            thisFilter = self.filters['sos'][i]
            newInit = signal.sosfilt_zi(thisFilter)
            self.filters['zi'][i] = newInit

    def filter(self, sig):
        filterOut = np.zeros_like(sig)

        for i in range(self.numChannels):
            out, self.filters['zi'][i] = signal.sosfilt(self.filters['sos'][i], sig[i, :], zi=self.filters['zi'][i])
            filterOut[i, :] = out

        return filterOut
    
    def filterByIndex(self, sig, channel):

        filterOut = np.zeros_like(sig)

        for i in channel:
            out, self.filters['zi'][i] = signal.sosfilt(self.filters['sos'][i], sig, zi=self.filters['zi'][i])
            filterOut = out

        return filterOut
    
    def offlineFilter(self, sig):
        filterOut = np.zeros_like(sig)

        for i in range(self.numChannels):
            b, a = signal.sosfreqz(self.filters['sos'][i])
            out = signal.filtfilt(b, a, sig[i, :])
            filterOut[i, :] = out

        return filterOut

    def getFilter(self, channelNum):
        return self.filters['sos'][channelNum], self.filters['zi'][channelNum]

    def printFilters(self):
        for i in range(self.numChannels):
            print(f'Channel {i}: {self.getFilter(i)}')
