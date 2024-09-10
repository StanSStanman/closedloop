import numpy as np
import scipy as sp
from numba import jit
import timeit


@jit("float64(float64[:], float64[:])", nopython=True, cache=True)
def corr(signal, template):
    return np.corrcoef(signal, template)[-1, 0]
    

def cond_A(signal, template, threshold=.5):
    # Signal should be a 2D array of shape (channels, t_points)
    N = signal.shape[1]

    # corr(S(t), T)
    corr_t0 = np.apply_along_axis(corr, 0, signal, template)
    # corr(S(t+1), T)
    corr_t1 = np.roll(corr_t0, -1)
    
    # corr(S(t+1), T) - corr(S(t), T) -> d_corr
    d_corr = (corr_t1[:N-1] - corr_t0[:N-1])
    # sum(((d_corr / |d_corr|) * .5) +.5) / N - 1
    coeff = np.sum(((d_corr / abs(d_corr)) * .5) + 0.5) / (N - 1)

    print(coeff)

    if coeff >= threshold:
        return True
    else:
        return False
    

def cond_B(signal, template, threshold=.75, sign=1.): # sign is 1 or -1
    N = signal.shape[1]
    corr_t0 = np.apply_along_axis(corr, 0, signal, template)
    coeff = np.sum((((sign*(0 - corr_t0)) / abs(0 - corr_t0)) * .5) + .5) / N

    print(coeff)

    if coeff >= threshold:
        return True
    else:
        return False


def cond_C(signal, threshold=.5):
    # The signal should pass some steps before passing through this function:
    #   - Referencing with respect to the average
    #   - Selecting the channels surrounding the target site
    #   - Averaging across these channels
    #   - Filter between .15 - 2. Hz
    # Thus in this case 'signal' is a vector of length N samples
    N = len(signal)
    sig_t1 = np.roll(signal, -1)
    d_sig = (sig_t1[:N-1] - signal[:N-1])
    coeff = np.sum((((d_sig) / abs(d_sig)) * .5) + .5) / (N - 1)

    print(coeff)

    if coeff >= threshold:
        return True
    else:
        return False


def cond_D(signal, threshold=.75, sign=1.):
    # See notes on cond_C
    N = len(signal)
    coeff = np.sum((((sign*(0 - signal)) / abs(0 - signal)) * .5) + .5) / N

    print(coeff)

    if coeff >= threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    chans = 256
    samples = 25

    a = np.random.uniform(-1, 1, (chans, samples))
    b = np.random.uniform(-1, 1, (chans))
    c = np.random.uniform(-1, 1, (samples))
    cond_A(a, b)
    start = time.time()
    cond_A(a, b)
    end_1 = time.time()
    cond_B(a, b, sign=1)
    end_2 = time.time()
    cond_C(c)
    end_3 = time.time()
    cond_D(c)
    end_4 = time.time()

    print('f1:', end_1 - start) # around 0.001 s -> improved with jit 0.0002 s
    print('f2:', end_2 - end_1) # around 0.001 s -> improved with jit 0.0002 s
    print('f3:', end_3 - end_2) # around 4.2e-5 s
    print('f4:', end_4 - end_3) # around 1.6e-5 s

    cycles = 1.3 # how many sine cycles
    resolution = 50 # how many datapoints to generate

    length = np.pi * 2 * cycles
    my_wave = np.sin(np.arange(0, length, length / resolution))
    my_wave[0] = 1e-10
    my_wave *= 10

    plt.plot(my_wave)
    plt.plot(my_wave*10)
    plt.show()

    cond_C(my_wave)
    cond_D(my_wave, sign=1.)

