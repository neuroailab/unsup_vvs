import pymongo as pm
import gridfs
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import pylab

from scipy import misc
import os, sys
import time

import sklearn.linear_model
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.cluster import KMeans

import math
from matplotlib.backends.backend_pdf import PdfPages

import PIL.Image
import IPython.display as ip_disp
from cStringIO import StringIO

DEFAULT_BATCHSIZE = 128
CACHE_DICT = {}

# Add parent directory to the system path so we can import from there.
parent, _ = os.path.split(os.getcwd())
if parent not in sys.path:
    sys.path.append(parent)

from tfutils_reader import (
    TfutilsReader, load_model,
    load_validation_results,
    load_loss_results,
    )
from embedding_stats import MemoryBank
import image_utils
from instance_stats import analysis


def display_img_array(*img_arrays, **kwargs):
    fmt = kwargs['fmt'] if 'fmt' in kwargs else 'jpeg'
    imgs = []
    for img_array in img_arrays:
        img_array = np.uint8(img_array)
        f = StringIO()
        PIL.Image.fromarray(img_array).save(f, fmt)
        img = ip_disp.Image(data=f.getvalue())
        imgs.append(img)
    ip_disp.display(*imgs)


def color_sequence(rgb, n):
    color_seq = np.stack([
        (rgb[c] * np.array(range(n)) / float(n)) for c in [0,1,2]
    ], axis=1)
    return np.array([tuple(c) for c in color_seq])


def plot_trajectory(traj, plot, components=(0,1),
                    projection_vecs=None,
                    color=(1, 0, 0), smoothed=True):
    if smoothed:
        traj = np.apply_along_axis(
            lambda x: analysis.smooth(x, conv_len=10),
            axis=0, arr=traj)
    if projection_vecs is None:
        top, _, projection_vecs = analysis.pca_analysis(traj)
        #print top

    projected_pts = []
    comp1, comp2 = components
    for pt in traj:
        coord1 = np.dot(pt, projection_vecs[comp1])
        coord2 = np.dot(pt, projection_vecs[comp2])
        projected_pts.append((coord1, coord2))

    xs, ys = zip(*projected_pts)
    plot.scatter(xs, ys, c=color_sequence(color, len(xs)))


def plot_tsne(data, **kwargs):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(data)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], **kwargs)

def plot_mds(data, **kwargs):
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    X_mds = clf.fit_transform(data)
    plt.scatter(X_mds[:,0], X_mds[:,1], **kwargs)


class ValidationPlot(object):
    def __init__(self, xlabel="Epoch", ylabel="Validation Performance"):
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel=xlabel, ylabel=ylabel)
        self.data = {}

    def add(self, label, xy_pairs):
        self.data[label] = xy_pairs

    def plot(self, start=None, end=None):
        start = start if start is not None else 0
        end = end if end is not None else float('inf')
        for label, xy_pairs in self.data.iteritems():
            xy_pairs = [(x,y) for x,y in xy_pairs if x >= start and x <= end]
            if len(xy_pairs) == 0:
                continue
            xs, ys = zip(*xy_pairs)
            self.ax.plot(xs, ys, label=label)
        self.ax.legend(loc='lower right')


class LossPlot(object):
    def __init__(self, xlabel="Epoch", ylabel="Loss Value", steps_per_epoch=10009):
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel=xlabel, ylabel=ylabel)
        self.steps_per_epoch = steps_per_epoch
        self.data = {}

    def add(self, label, values, start_epoch=0):
        self.data[label] = (values, start_epoch)

    def plot(self, start=None, end=None):
        start = start if start is not None else 0
        end = end if end is not None else float('inf')
        for label, (values, start_epoch) in self.data.iteritems():
            s = max(start - start_epoch, 0) * self.steps_per_epoch
            e = max(end - start_epoch, 0) * self.steps_per_epoch
            if s >= e:
                continue
            values = values[s:e]
            idxs = [i / float(self.steps_per_epoch) for i in range(len(values))]
            self.ax.plot(idxs, values, label=label)
        self.ax.legend(loc='upper right')







def show_train_learnrate(
        curr_expid,
        conn,
        dbname='combinet-test',
        colname='combinet',
        start_N=50,
        with_dataset=None,
        batch_watch_start=0,
        batch_watch_end=None,
        do_conv=False,
        conv_len=100,
        new_figure=True,
        batch_size=DEFAULT_BATCHSIZE,
        batch_offset=0,
        max_step=None,
        label_now=None,
        loss_key='loss',
        big_loss_thres=50,
        ):

    # Weird setting in tfutils, the records are stored in colname + .files
    colname = colname + '.files'

    # Used for legend
    if label_now is None:
        label_now = curr_expid

    # Get the query through exp_id
    find_res = conn[dbname][colname].find(
                {'exp_id': curr_expid, 'train_results': {'$exists': True}})

    # Sort the results based on steps, for overlapped steps, use the last record
    find_res = sorted(find_res, key = lambda x: x['step'])
    new_find_res = []
    for curr_indx in xrange(len(find_res)-1):
        if find_res[curr_indx]['step'] == find_res[curr_indx+1]['step']:
            continue
        new_find_res.append(find_res[curr_indx])
    new_find_res.append(find_res[len(find_res)-1])
    find_res = new_find_res

    # Only use the steps
    if max_step:
        find_res = filter(lambda x: x['step']<max_step, find_res)
    train_vec = np.concatenate(
            [
                [(_r[loss_key], _r['learning_rate'])
                    for _r in r['train_results']]
                for r in find_res])

    # Build a new figure if needed
    if new_figure:
        fig = plt.figure(figsize=(12, 5))

    # For loss plotting
    plt.subplot(1, 2, 1)
    ## Throw away first start_N losses
    ## (as they might be huge, which influences the visualizations)
    _N = start_N
    inter_list = train_vec[_N:, 0]
    inter_list = np.asarray(inter_list)
    ## Filter out the spikes, which can influence the visualizations
    inter_list = inter_list[inter_list < big_loss_thres]
    ## Smooth the loss values
    if do_conv:
        conv_list = np.ones([conv_len])/conv_len
        inter_list = np.convolve(inter_list, conv_list, mode='valid')

    ## The x axis will be per 10k steps
    temp_x_list = np.asarray(range(len(inter_list)))\
            * 1.0 * batch_size / (10000*DEFAULT_BATCHSIZE) + batch_offset
    new_indx_list = temp_x_list > batch_watch_start
    if batch_watch_end:
        new_indx_list = (temp_x_list>batch_watch_start) \
                & (temp_x_list<batch_watch_end)
    ## Do the plotting
    plt.plot(
            temp_x_list[new_indx_list],
            inter_list[new_indx_list],
            label=label_now)
    plt.title('Training loss')
    plt.legend(loc = 'best')

    # For learning rate plot
    plt.subplot(1, 2, 2)
    temp_y_list = train_vec[_N:, 1]
    # This x_list is slightly different from that in losses (convolution)
    temp_x_list_lr = np.asarray(range(len(temp_y_list)))\
            * 1.0 * batch_size / (10000*DEFAULT_BATCHSIZE) + batch_offset
    new_indx_list_lr = temp_x_list_lr > batch_watch_start
    if batch_watch_end:
        new_indx_list_lr = (temp_x_list_lr>batch_watch_start) \
                & (temp_x_list_lr<batch_watch_end)
    plt.plot(
            temp_x_list_lr[new_indx_list_lr],
            temp_y_list[new_indx_list_lr],
            label=label_now)
    plt.title('Learning Rate')


def show_val(
        curr_expid,
        conn,
        dbname='combinet-test',
        colname='combinet',
        gridfs_name='combinet',
        key='loss',
        valid_key='topn',
        big_dict=CACHE_DICT,
        batch_watch_start=0,
        batch_watch_end=None,
        new_figure=True,
        val_N=0,
        batch_size=DEFAULT_BATCHSIZE,
        label_now=None,
        val_step=1.0,
        batch_offset=0,
        do_plot=True,
        ):

    colname = colname + '.files'
    if not label_now:
        label_now = curr_expid

    # Get the query
    find_res = conn[dbname][colname].find(
            {'exp_id': curr_expid,
             'validation_results': {'$exists': True}})
    # Sort using step
    find_res = sorted(find_res, key = lambda x: x['step'])
    if len(find_res)==0:
        return None
    # Remove the duplicate steps
    new_find_res = []
    for curr_indx in xrange(len(find_res)-1):
        if find_res[curr_indx]['step'] == find_res[curr_indx+1]['step'] \
                and find_res[curr_indx]['step']:
            continue
        new_find_res.append(find_res[curr_indx])
    new_find_res.append(find_res[len(find_res)-1])
    find_res = new_find_res

    # Get the validation results
    if len(find_res)==0:
        return
    list_res = filter(lambda x: valid_key in x['validation_results'], find_res)

    test_vec = [r['validation_results'][valid_key][key] for r in list_res]
    x_range = range(len(test_vec))
    _N = val_N
    if new_figure and do_plot:
        plt.figure(figsize=(9, 5))
    x_range = np.asarray(x_range) * val_step * (batch_size)/(DEFAULT_BATCHSIZE)\
            + batch_offset
    x_range = np.asarray(x_range[_N:])
    test_vec = np.asarray(test_vec[_N:])

    choose_indx = x_range > batch_watch_start
    if batch_watch_end is not None:
        choose_indx = choose_indx & (x_range < batch_watch_end)

    if do_plot:
        plt.plot(x_range[choose_indx], test_vec[choose_indx], label = label_now)
    if 'top' in key and  do_plot:
        print(
                curr_expid, key,
                np.max(test_vec[choose_indx]), np.argmax(test_vec))
    if do_plot:
        plt.title('Validation Performance, %s' % key)

    if do_plot:
        return test_vec[_N:]
    else:
        return x_range[choose_indx], test_vec[choose_indx]


def show_instance_perf(
        txt_path='/home/chengxuz/previous_logs/instance_task.log',
        perf_detail_str='50000/50000',
        perf_str='Top1: ',
        mult_num=1,
        offset_epoch=0,
        epoch_range=None,
        ):

    with open(txt_path, 'r') as f:
        data = f.readlines()
    all_perfes = []
    for each_line in data:
        if perf_str in each_line \
                and perf_detail_str in each_line \
                and 'Top5' not in each_line:
            perf_plc = each_line.find(perf_str)
            curr_perf = float(each_line[perf_plc+len(perf_str):]) * mult_num
            all_perfes.append(curr_perf)

    start, end = epoch_range or (0, len(all_perfes))
    end = min(end, len(all_perfes))
    x_perf = range(start, end)
    plt.plot([x + offset_epoch for x in x_perf], [all_perfes[x] for x in x_perf])


def show_tf_inst_loss(
        which_loss='total',
        txt_path='/home/chengxuz/previous_logs/'\
                + 'log_performance_instance_18_fxcp_2.txt',
        end_N=None,
        start_N=None,
        label_now=None,
        offset_N=0,
        color=None
        ):
    with open(txt_path, 'r') as f:
        data = f.readlines()

    mapping_dict = {
            'total': 'Loss:',
            'model': 'Loss model:',
            'noise': 'Loss noise:',
            }

    all_losses = []
    loss_str = mapping_dict[which_loss]
    for each_line in data:
        if loss_str in each_line:
            loss_plc = each_line.find(loss_str)
            curr_loss_str = each_line[loss_plc+len(loss_str):]
            curr_loss_str = curr_loss_str.split(',')[0]
            if which_loss=='total':
                which_one = 0
            elif which_loss=='model':
                which_one = 1
            else:
                which_one = 2
            curr_loss = float(curr_loss_str)
            all_losses.append(curr_loss)
    all_losses = all_losses[::10]
    if end_N is not None:
        all_losses = all_losses[:end_N]
    if start_N is not None:
        all_losses = all_losses[start_N:]
    # Smooth the loss curve
    conv_len = 100
    conv_list = np.ones([conv_len])/conv_len
    all_losses = np.convolve(all_losses, conv_list, mode='valid')

    if not label_now:
        search_str = 'log_performance_instance_'
        label_now = txt_path[txt_path.find(search_str) + len(search_str):]
        label_now = label_now[:-4]
    plot_kwargs = {}
    if color:
        plot_kwargs['color'] = color
    x_indx = np.arange(len(all_losses)) + offset_N
    plt.plot(x_indx, all_losses, label=label_now, **plot_kwargs)
    plt.legend(loc = 'lower left')
