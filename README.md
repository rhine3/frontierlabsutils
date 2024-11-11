# Acoustic synchronization utilities for Frontier Labs recorders

If you're using Frontier Labs recorders for acoustic localization, these tools can help you synchronize your recorders. 

## About
Frontier Labs recorders with the localization firmware can be used to create time-synchronized recordings for acoustic localization. These recorders use an onboard GPS receiver to timestamp the beginning and end of each recording. However, these recordings are not synchronized until they are post-processed.

This repository contains code for post-processing Frontier Labs recordings to synchronize them. It uses the timestamps saved in the filename of each recording to resample the recordings. It also uses the `loclog.txt` files saved by the recorders to account for "dropped buffers," i.e. the occasionally dropped audio data that sometimes happens when using time-synchronizing recorders. 

## How to use
The files included in this repository are:
* The utility library itself: [`frontierlabsutils.py`](frontierlabsutils.py)
* Example recording resampling using the utilities: [`example_sync.py`](example_sync.py)
* Example trimming of resampled recordings using the utilities: [`example_trim.py`](example_trim.py)

I developed this library for my own application, so it isn't quite "plug and play." To use it, you will have to change at least the following things:
* the function that returns a list of all the names of the recorders in your array ([`get_recorder_list`](https://github.com/rhine3/frontierlabsutils/blob/594d194b37e025bfbc211392ef5a2ede06bfc302/frontierlabsutils.py#L35))
* the hard-coded path to the original audio data for your array ([`DATA_DIR`](https://github.com/rhine3/frontierlabsutils/blob/594d194b37e025bfbc211392ef5a2ede06bfc302/frontierlabsutils.py#L31))
* path wildcards in some functions ([`get_recording_path`](https://github.com/rhine3/frontierlabsutils/blob/594d194b37e025bfbc211392ef5a2ede06bfc302/frontierlabsutils.py#L127), [`get_overflows`](https://github.com/rhine3/frontierlabsutils/blob/594d194b37e025bfbc211392ef5a2ede06bfc302/frontierlabsutils.py#L252), [`get_loclog_contents`](https://github.com/rhine3/frontierlabsutils/blob/594d194b37e025bfbc211392ef5a2ede06bfc302/frontierlabsutils.py#L348)). `MIN231x` was the prefix I added to all of the recorder names, which were deployed in rows and named A1-A7, B1-B7, etc., so the original recording paths include `MIN231x` in them in addition to the recording name A1-G7. 
* how to get recordings from the same recording period ([`get_all_times`](https://github.com/rhine3/frontierlabsutils/blob/594d194b37e025bfbc211392ef5a2ede06bfc302/frontierlabsutils.py#L210))

## Disclaimer
I'm not associated with Frontier Labs - they are developing their own software that both synchronizes recordings and can be used to localize sounds. That software wasn't available when I made these utilities. Please see their website/contact them for updates on it - https://www.frontierlabs.com.au/post/acoustic-localisation

## Other advice
Some other personal advice for how to get the most out of the Frontier Labs recorders for localization:
* Make sure you have the localization firmware! Reach out to Frontier Labs here - https://www.frontierlabs.com.au/post/acoustic-localisation
* Dropped buffers may be associated with SD cards that are older, or haven’t been reformatted recently, so consider reformatting your SD cards before using them
* Your recordings should be less than 1 hour long in order to be able to get <1ms sync accuracy (the recorders take a single time stamp in the middle of the recording and use the current clock speed to back-calculate the beginning and end time of the recordings). The longer the recordings are, the less accurate the synchronization will be, especially at the start and end of the recording. I made 10-minute long recordings.
* Try to position your ARUs so they have a clear view of the sky, otherwise their batteries may drain more quickly while trying to get a GPS fix. The recorders may lose battery faster than you expect, and some may lose it faster than others depending on their position. I would check your recorders after a few days of recording and see if any seem to be losing battery faster; you may want to prioritize checking/recharging those recorders more frequently
* Consider doing a test deployment with your schedule (e.g. in a park or backyard) before your actual deployment, and check that the start/end times of the recordings are accurately listed in the recording filenames
* The firmware I used wasn’t set up to use 2 microphones - our recorders only made single-channel recordings even though they had two mics attached.

To automatically localize sounds in your recordings, check out our lab's automated localization pipeline implemented in the OpenSoundscape Python package: https://opensoundscape.readthedocs.io/en/latest/tutorials/acoustic_localization.html 
