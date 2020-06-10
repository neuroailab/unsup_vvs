import h5py
import numpy

from compute_stat_PCA import write_to_tfrs

h5path = '/data2/chengxuz/vm_response/response_f2bpslen.hdf5'
fin = h5py.File(h5path, 'r')

IT_ave = numpy.asarray(fin['IT_ave'])

write_to_tfrs(IT_ave, 
        tfrpath='/data2/chengxuz/vm_response/tfrecords/',
        each_key='IT_ave',
        num_per_tfr=256,
        name_pat='train_%i.tfrecords',
        )

V4_ave = numpy.asarray(fin['V4_ave'])
write_to_tfrs(V4_ave, 
        tfrpath='/data2/chengxuz/vm_response/tfrecords/',
        each_key='V4_ave',
        num_per_tfr=256,
        name_pat='train_%i.tfrecords',
        )

h5path = '/data2/chengxuz/vm_response/response_f2bpslen_val.hdf5'
fin = h5py.File(h5path, 'r')

IT_ave = numpy.asarray(fin['IT_ave'])

write_to_tfrs(IT_ave, 
        tfrpath='/data2/chengxuz/vm_response/tfrecords/',
        each_key='IT_ave',
        num_per_tfr=256,
        name_pat='test_%i.tfrecords',
        )

V4_ave = numpy.asarray(fin['V4_ave'])
write_to_tfrs(V4_ave, 
        tfrpath='/data2/chengxuz/vm_response/tfrecords/',
        each_key='V4_ave',
        num_per_tfr=256,
        name_pat='test_%i.tfrecords',
        )
