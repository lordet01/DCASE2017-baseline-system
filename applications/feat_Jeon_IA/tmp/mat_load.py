import scipy.io
import subprocess
import numpy

mel_spectrum = scipy.io.loadmat('./tmp_melfeat.mat')
mel_spectrum = mel_spectrum['mel_spectrum']

feature_matrix = []
mel_spectrum = numpy.log(mel_spectrum + 0.000000001)
mel_spectrum = mel_spectrum.T

feature_matrix.append(mel_spectrum)

matname = 'feat_my_again.mat'
feat_dic = {'feat': feature_matrix}
scipy.io.savemat(matname, mdict={'feat_dic': feature_matrix})