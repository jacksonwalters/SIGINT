# SIGINT
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

A collection of Python scripts serving as an introduction to SIGINT (signal intelligence) techniques.

- `amplitude_modulation.py`: generate a sequence of amplitude-moduled pulses, use filtering and autocorrelation to estimate PRI (pulse repitition interval)
- `basic_workflow.py`: generate pulse train, add Gaussian noise, use autocorrelation to estimate PRI
- `chirped_pulses.py`: simulates a noisy train of linear chirped pulses, applies a matched filter and autocorrelation to analyze it, and then estimates the PRI and pulse width from the matched filter output
- `irregular_pulse_detection.py`: simulates a noisy pulse train with random PRI jitter, applies matched filtering and autocorrelation, and then estimates the mean and variability of the PRI from the detected pulses
- `gaussian_jittered.py`: simulates a noisy pulse train with Gaussian PRI jitter, detects the pulses using matched filtering, and estimates the mean and standard deviation of the PRI (with outlier rejection) while visualizing the signal, matched filter output, and PRI statistics over time
- `multi_emitter_noisy_jitter.py`: simulates two noisy pulse trains with different PRIs and jitter, combines them, and uses autocorrelation with peak detection and harmonic rejection to estimate the fundamental PRIs of both emitters
- `noisy_multi_emitter_jitter.py`: simulates two overlapping pulse trains with different PRIs in noise and uses smoothed autocorrelation with peak detection and harmonic rejection to estimate their fundamental PRIs
- `time_difference_of_arrival.py`: simulates two noisy pulse trains received at different times and estimates the delay between them using cross-correlation
- `two_pulse_trains.py`: simulates two overlapping pulse trains with different PRIs and uses autocorrelation with harmonic rejection to estimate their fundamental repetition intervals
