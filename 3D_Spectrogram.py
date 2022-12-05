import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import sys
from scipy import interpolate
import statsmodels.api as sm
import librosa
import os
import math
from math import pi





class Visualizer(object):
    """ A class to visualize a given song"""

    def __init__(self, path):
        """
        Initializes the QtGui.
        Extracts primary audio features to initialize the first frame.
        Manipulates the audio features for required visualization
        Defines the presentation parameters
        Performs data extension for a smooth start
        Initializes the graph items for animation
        Parameters
        ----------
        path : str
            Path of the audio file (audio file should be in .wav format)
        """

        # Initializes the QtGui
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.setGeometry(0, 0, 1920, 1080) # Set to "portrait"
        self.w.setWindowTitle('Music Visualizer')
        self.Vector = QtGui.QVector3D
        self.w.setCameraPosition(pos=None , distance=900, azimuth=270, elevation=50)
        self.w.show()

        # Primary feature extraction from the audio
        self.time_series, self.sample_rate = librosa.load("file_path")
        self.hop_length = 64 # Fidelity (# of "lines")
        self.stft = np.abs(librosa.stft(self.time_series, hop_length=self.hop_length, n_fft=2048))
        self.oenv = librosa.onset.onset_strength(y=self.time_series, sr=self.sample_rate,
                                                 hop_length=self.hop_length)
        self.onset_frames = librosa.onset.onset_detect(onset_envelope=self.oenv, sr=self.sample_rate)
        self.beat_boolean = np.array([1 if i in self.onset_frames else 0 for i in
                                      range(0, len(librosa.times_like(self.stft)))])
        self.spectral_centroid = librosa.feature.spectral_centroid(y=self.time_series, sr=self.sample_rate)
        self.chroma = librosa.feature.chroma_stft(S=self.stft, sr=self.sample_rate)
        self.n_chromas = self.chroma.shape[0]

        # Extracting and Manipulating features for visualization
        self.tempo_mult = 20
        self.spectr_mult = 120 # Originally "60" (Controls height of amplitude)
        self.chroma_tracer_offset_height = self.tempo_mult + self.spectr_mult * 1.8
        self.tempo_final = self.get_tempogram()
        self.spectrogram_final = self.get_spectrogram()
        self.spectro_beat_final = self.get_spectrogram()
        self.camera_x = self.get_camera_x_position()
        self.chroma_tracer_z = self.get_chroma_tracer_z()

        # Defining the presentation parameters
        self.tempo_chunks = self.tempo_final.shape[0]
        self.spectro_beat_chunks = self.spectro_beat_final.shape[0]
        self.window_length = 400
        self.matrix_offset = 1

        # Extending the data to make the initial representation of the first timestamp seem continuous
        self.tempo_final = self.data_extender(self.tempo_final)
        self.chroma = self.data_extender(self.chroma)
        self.spectro_beat_final = self.data_extender(self.spectro_beat_final)
        self.beat_boolean = self.data_extender(self.beat_boolean)
        self.camera_x = self.data_extender(self.camera_x)
        self.chroma_tracer_z = self.data_extender(self.chroma_tracer_z)
        

        # Initializing the relevant data generators
        self.tempo_gen = self.data_sample_gen(self.tempo_final, self.matrix_offset)
        self.chroma_gen = self.data_sample_gen(self.chroma, self.matrix_offset, op_data_as_cood=False)
        self.specto_beat_gen = self.data_sample_gen(self.spectro_beat_final, self.matrix_offset, along_y=True)
        self.beat_gen = self.data_sample_gen(self.beat_boolean, self.matrix_offset, ip_data_1d=True)
        self.cam_x_gen = self.data_sample_gen(self.camera_x, self.matrix_offset, ip_data_1d=True)
        self.chroma_tracer_z_gen = self.data_sample_gen(self.chroma_tracer_z, self.matrix_offset, ip_data_1d=True)

        # Getting the first set of data sample
        #self.tempo_sample = next(self.tempo_gen)
        #self.chroma_sample = next(self.chroma_gen)
        self.specto_beat_sample = next(self.specto_beat_gen)
        self.beat_sample = next(self.beat_gen)
        self.cam_x_sample = next(self.cam_x_gen)
        self.chroma_tracer_z_sample = next(self.chroma_tracer_z_gen)

        # Sets the folder to store the frames
        self.total_frames = self.tempo_final.shape[1]
        self.img_path = os.path.abspath('file_path')
        os.makedirs(self.img_path, exist_ok=True)

        # Initializing Dictionaries to store the graph items
        self.traces = {}
        self.tempo_traces = {}
        self.circles = {}
        self.chroma_tracers = {}
        self.spectro_beat_tracers = {}

        # Initializing a tracker for observing the run
        self.tracker_gen = self.tracker(self.total_frames)
        next(self.tracker_gen)

        # Initializes all the graph items
        self.initialize_visuals()

    

    def initialize_visuals(self):

        # Spectrobeat visuals initialization
        start_pt = 0
        for i in range(self.window_length):
            pts = np.zeros((384, 3))
            #print(pts[2:3])
            start_pt += start_pt
            self.spectro_beat_tracers[i] = gl.GLLinePlotItem(pos=pts,
                                                             color=pg.intColor(index=1, maxHue=i),
                                                             antialias=True,
                                                             width=self.beat_sample[i])
            self.w.addItem(self.spectro_beat_tracers[i])

    def update(self):
        """
        This will update the graph items which were initialized by initialize_visuals()
        Every update will be saved as an image
        """

        # update tracker
        tracker_id = next(self.tracker_gen)

        # Updating the Spectro Beat array
        new_specto_beat_sample = next(self.specto_beat_gen)
        new_beat_sample = next(self.beat_gen)
        start_pt = 0
        for i in range(self.window_length):
            #start = np.random.random_integers(0, 1, 4)
            pts = new_specto_beat_sample[start_pt:start_pt + self.spectro_beat_chunks]
            start_pt += self.spectro_beat_chunks
            self.spectro_beat_tracers[i].setData(pos=pts,
                                                 color=(pg.mkColor(255, 0, 0, 50)),
                                                 antialias=True,
                                                 width=1)
        # Updating the Camera
        new_cam_x_sample = next(self.cam_x_gen)
                                               
        #self.w.grabFramebuffer().save(os.path.join(self.img_path, f'img_{tracker_id}.png'))
    
    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(5)
        self.start()
    
    def data_extender(self, data):
        """
        This helps in adding additional data at the very beginning to ensure that the first frame isn't blank
        :param data: 1D or 2D numpy array
        :return: extended data
        """
        if len(data.shape) == 2:
            #apt = np.zeros((3, 384)).T
            arr = np.clip(data[:, 0], 0 , 0)
            append_this = np.tile(arr, (self.window_length - 1, 1)).T # Creates an array: (data[:, 0], [399, 1])
            #print(np.shape(data[:, 0]))
            return np.hstack((append_this, data))
        elif len(data.shape) == 1:        
            return np.hstack((np.ones(self.window_length - 1) * data[0], data))
        else:
            raise ValueError("Can't handle data with more than 2 dimensions")
            

    def data_sample_gen(self, data, offset, op_data_as_cood=True, along_y=False, ip_data_1d=False):
        """
        This is a generator which gives us an updated window of the original data which needs to be plotted
        :param data: data from which the sample needs to be created
        :param offset: the steps by which the new window needs to be offset
        :param op_data_as_cood: if True this provides data with coordinates. if False it provides just the data-points
        :param along_y: if True this provides data with coordinates such that the GLLinePlotItem plots line along
        the y-axis
        :param ip_data_1d: True if the input data is 1D
        :return: yields the relevant data sample
        """
        starter_ind = 0
        while True:
            if ip_data_1d:
                sample = data[starter_ind: starter_ind + self.window_length]
                yield sample
                starter_ind = (starter_ind + offset) % len(data)
            else:
                sample = data.T[starter_ind: starter_ind + self.window_length].T
                if op_data_as_cood:
                    if along_y:
                        xx, yy = np.meshgrid(np.arange(sample.shape[0]), np.arange(sample.shape[1]))
                        sample_cood = np.vstack((xx.ravel(), yy.ravel(), sample.ravel(order="F"))).T
                        yield sample_cood
                    else:
                        xx, yy = np.meshgrid(np.arange(sample.shape[1]), np.arange(sample.shape[0]))
                        sample_cood = np.vstack((yy.ravel(), xx.ravel(), sample.ravel())).T
                        yield sample_cood
                else:
                    yield sample
                starter_ind = (starter_ind + offset) % data.shape[1]

    def get_tempogram(self):
        """
        This scales the original tempogram and then multiplies it with a multiplier
        :return: Manipulated Tempogram
        """
        tempogram = librosa.feature.tempogram(onset_envelope=self.oenv, sr=self.sample_rate,
                                              hop_length=self.hop_length, norm=np.inf)
        tempogram_scaled = (tempogram - np.min(tempogram)) / \
                           (np.max(tempogram) - np.min(tempogram))
        return tempogram_scaled * self.tempo_mult

    def get_spectrogram(self):
        """
        The original spectrogram is first mean filtered. Then it's dimensions are edited so as to match that of the
        tempogram. This is done using scaling and 2d interpolation. It is finally multiplied by a multiplier
        :return: Manipulated Spectrogram
        """
        spectrogram = librosa.amplitude_to_db(self.stft, amin=0.1, ref=np.max)  # Spectrogram Added "amin" parameter to flatten
        spectrogram = librosa.decompose.nn_filter(spectrogram,
                                                  aggregate=np.mean)  # Spectrogram Mean Filtered
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        sampling_freqs = np.logspace(start=9/6, # Originally "0"
                                     stop=np.log(freqs[-1]) / np.log(10),
                                     num=self.tempo_final.shape[0], # Originally "0" (less wide)
                                     endpoint=True)
        sampling_freqs_indices = (sampling_freqs - np.min(sampling_freqs)) / \
                                 (np.max(sampling_freqs) - np.min(sampling_freqs)) * spectrogram.shape[0]
        freq_indices = np.arange(0, spectrogram.shape[0])
        time_indices = np.arange(0, spectrogram.shape[1])
        f = interpolate.interp2d(time_indices, freq_indices, spectrogram, kind='linear')
        spectrogram = f(time_indices, sampling_freqs_indices)
        spectrogram = (spectrogram - np.min(spectrogram)) / \
                      (np.max(spectrogram) - np.min(spectrogram)) # Flip "min" and "max" to invert
        return spectrogram * self.spectr_mult
        
        

    def get_camera_x_position(self):
        """
        This gives us the x-coordinates of the camera position which follows the spectral centroid. The spectral
        centroid is smoothened (LOWESS) to avoid quick transitions which might be strenuous for the viewer
        :return: x-coordinates of the camera
        """
        cent_scaled = (self.spectral_centroid[0] - np.min(self.spectral_centroid[0])) / \
                      (np.max(self.spectral_centroid[0]) - np.min(self.spectral_centroid[0])) * \
                      self.tempo_final.shape[0]
        lowess = sm.nonparametric.lowess(cent_scaled,
                                         np.arange(0, len(cent_scaled)),
                                         frac=0.1, return_sorted=False)
        camera_x = (lowess - np.min(lowess)) / \
                   (np.max(lowess) - np.min(lowess)) * \
                   self.tempo_final.shape[0]
        return camera_x        
    def get_chroma_tracer_z(self):
        """
        This provides the z coordinates of the chroma tracer. The chroma tracer flows over the tempogram with an offset
        :return: z coordinates of the chroma tracer
        """
        chroma_tracer_z = np.array([self.tempo_final[min(int(round(j)), self.tempo_final.shape[0] - 1), i]
                                    for i, j in enumerate(self.camera_x)])
        chroma_tracer_z_lowess = sm.nonparametric.lowess(chroma_tracer_z,
                                                         np.arange(0, len(chroma_tracer_z)),
                                                         frac=0.01, return_sorted=False)
        chroma_tracer_z = (chroma_tracer_z_lowess - np.min(chroma_tracer_z_lowess)) / \
                          (np.max(chroma_tracer_z_lowess) - np.min(chroma_tracer_z_lowess)) * self.tempo_mult
        return chroma_tracer_z + self.chroma_tracer_offset_height

    @staticmethod
    def start():
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    @staticmethod
    def points_in_circum(x, y, z, r, n=100):
        """
        Give center coordinates. Function will return pts corresponding to a circle in the XZ-Plane
        """
        return np.array([(math.cos(2 * pi / n * i) * r + x, y, math.sin(2 * pi / n * i) * r + z)
                         for i in range(0, n + 1)])

    @staticmethod
    def tracker(frames):
        """
        Generator for tracking the frame number
        :param frames: No of frames before the codes stops
        :return: yields the current frame number
        """
        starter = 0
        while starter < frames:
            print(starter)
            yield starter
            starter += 1
        sys.exit("Image generation complete!")


if __name__ == "__main__":
    audio_path = "file_path"
    t = Visualizer(audio_path)
    t.animation()
