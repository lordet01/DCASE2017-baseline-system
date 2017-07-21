import cPickle as pickle
import numpy as np
from numpy import array

filename = 'sequence_mixture_devtest_gunshot_492_b549f585888faf1672438c394b039d1b.cpickle'
#filename = 'sequence_mixture_devtest_gunshot_492_b549f585888faf1672438c394b039d1b_Jeon.cpickle'
with open(filename, 'rb') as fp:
	feat_dic = pickle.load(fp)
	#print feat_dic['mel_spectrum']

	
for key, value in feat_dic.iteritems() :
    print key	

#print feat_dic['meta'] 
	
feat = feat_dic['feat'] 
feat = array(feat)
print feat.shape
print np.amax(feat)
print np.amin(feat)

import scipy.io
matname = 'feat_org.mat'
scipy.io.savemat(matname, mdict={'feat_dic': feat})