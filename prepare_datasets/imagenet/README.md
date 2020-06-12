# Data preparation

## Create tfrecords for ImageNet

Prepare the ImageNet data as the raw JPEG format used in pytorch ImageNet training (see [link](https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py)).
Then run the following command:
```
python build_tfrs.py --save_dir /path/to/imagenet/tfrs --img_folder /path/to/imagenet/raw/folder
```
For convenience, please name the saved `/path/to/imagenet/tfrs` as `/path/to/datasets/TFRecord_Imagenet_standard/image_label_full_widx`

## Create subsets of ImageNet

First create a pickle file including all labels in the same order of the tfrecords. Run the following command:
```
python get_all_label.py --save_path /path/to/label/pkl --load_dir /path/to/imagenet/tfrs
```

Then create corresponding label and index npy files for p% (p={1, 2, 3, 4, 5, 6, 10, 20, 50}). For example, when p=1:
```
python make_balance_partIN.py --lbl_save_path /path/to/part/label/npy --idx_save_path /path/to/part/index/npy --lbl_pkl_path /path/to/label/pkl --per_cat_img 13
```

Here `per_cat_img` parameter should be set as `p% * 1300`. These npy files will be used to train Local Label Propagation models.
For Few-Label controls and Mean Teacher models, tfrecords of subsets are needed. They can be created by running the following command:

```
python sample_tfr_by_index.py --save_dir /path/to/part/tfr/dir --index_path /path/to/part/index/npy --load_dir  /path/to/imagenet/tfrs --img_per_file 130
```

We recommend that `img_per_file` parameter should be 130 when `p<=10`, when `p>10`, it should scale with p, for example, it should be 650 when `p=50`.
For convenience, please name the saved `/path/to/part/tfr/dir` like `/path/to/datasets/TFRecord_Imagenet_standard/image_label_p01_balanced` (the example for p=1).
