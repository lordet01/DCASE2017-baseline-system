import scipy.io
import subprocess
import numpy
audio_file = 'D:\\DLproj\\DCASE2017-baseline-system\\applications\\data\\TUT-rare-sound-events-2017-development\\generated_data\\mixtures_devtest_0367e094f3f5c81ef017d128ebff4a3c\\audio\\mixture_devtest_babycry_000_3f4d0f979789b888f9830c3ec396ac41.wav'
process = subprocess.call(['.\\feat_MATLAB.bat', '\''+audio_file+'\''])
# "start /wait cmd /c .\\feat_MATLAB.bat "+'\''+audio_file+'\''

mel_spectrum = scipy.io.loadmat('./feat_Jeon_IA/tmp/tmp_melfeat.mat')
mel_spectrum = mel_spectrum['mel_spectrum']


log_mel = numpy.log(mel_spectrum + 0.0000000000001);
print log_mel
