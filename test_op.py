from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIGenericIterator

import numpy as np

import nvidia.dali.plugin_manager as plugin_manager
plugin_manager.load_library('./build/libjpeg2dct.so')


def common_pipeline(jpegs, labels):
    dct_y, dct_cb, dct_cr = fn.jpeg_to_dct(jpegs, normalize=True, channels=3, device='cpu')
    dct_y = fn.resize(
        dct_y.gpu(),
        dtype=types.FLOAT,
        size=(56, 56),
        interp_type=types.INTERP_LINEAR)
    dct_cb = fn.resize(
        dct_cb.gpu(),
        dtype=types.FLOAT,
        size=(56, 56),
        interp_type=types.INTERP_LINEAR)
    dct_cr = fn.resize(
        dct_cr.gpu(),
        dtype=types.FLOAT,
        size=(56, 56),
        interp_type=types.INTERP_LINEAR)
    images = fn.cat(dct_y, dct_cb, dct_cr, axis=2)
    images = fn.transpose(images, perm=[2, 0, 1])
    return images, labels


@pipeline_def
def mxnet_reader_pipeline(path, num_gpus):
    jpegs, labels = fn.readers.mxnet(
        path=[path+"train.rec"],
        index_path=[path+"train.idx"],
        random_shuffle=True,
        shard_id=Pipeline.current().device_id,
        num_shards=num_gpus,
        name='Reader')

    return common_pipeline(jpegs, labels)


if __name__ == '__main__':
    from tqdm import tqdm
    db_folder = '/mnt/swmeng/DCTNet_Mxnet/data/'
    N = 1             # number of GPUs
    BATCH_SIZE = 128  # batch size per GPU
    label_range = (0, 999)
    pipes = [mxnet_reader_pipeline(path=db_folder,
        batch_size=BATCH_SIZE, num_threads=8, device_id=device_id, num_gpus=N) for device_id in range(N)]
    dali_iter = DALIGenericIterator(
        pipes,
        [('data', DALIGenericIterator.DATA_TAG),
         ('label', DALIGenericIterator.LABEL_TAG)],
        reader_name='Reader')

    for i, data in enumerate(tqdm(dali_iter)):
        # Testing correctness of labels
        for d in data:
            label = d.label[0].asnumpy()
            image = d.data[0]
            ## labels need to be integers
            assert(np.equal(np.mod(label, 1), 0).all())
            ## labels need to be in range pipe_name[2]
            assert((label >= label_range[0]).all())
            assert((label <= label_range[1]).all())
    print("OK")
