from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np

if sys.version_info.major == 2:  # Python 2.x
  from StringIO import StringIO as BIO
else:  # Python 3.x
  from io import BytesIO as BIO

if importlib.util.find_spec('tensorflow'):
  import tensorflow as tf


class PrintLogger(object):

  def __init__(self):
    """Create a summary writer logging to log_dir."""
    self.name = 'PrintLogger'

  def log(self, string):
    print(string)

  def close(self):
    print('-' * 30 + ' close printer ' + '-' * 30)


class Logger(object):

  def __init__(self, log_dir, seed, create_model_dir=True, use_tf=False, sparse_flag=False):
    """Create a summary writer logging to log_dir."""
    self.seed = int(seed) if not sparse_flag else seed
    self.log_dir = Path(log_dir)
    self.model_dir = Path(log_dir) / 'checkpoint'
    self.log_dir.mkdir(parents=True, exist_ok=True)
    if create_model_dir:
      self.model_dir.mkdir(parents=True, exist_ok=True)
    # self.meta_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    self.use_tf = bool(use_tf)
    self.tensorboard_dir = self.log_dir / ('tensorboard-{:}'.format(time.strftime('%d-%h', time.gmtime(time.time()))))
    # self.tensorboard_dir = self.log_dir / ('tensorboard-{:}'.format(time.strftime( '%d-%h-at-%H:%M:%S', time.gmtime(time.time()) )))
    self.logger_path = self.log_dir / 'seed-{:}-T-{:}.log'.format(self.seed, time.strftime('%d-%h-at-%H-%M-%S',
                                                                                           time.gmtime(time.time())))
    self.logger_file = open(self.logger_path, 'w')

    if self.use_tf:
      self.tensorboard_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
      self.writer = tf.summary.FileWriter(str(self.tensorboard_dir))
    else:
      self.writer = None

  def __repr__(self):
    return (
      '{name}(dir={log_dir}, use-tf={use_tf}, writer={writer})'.format(name=self.__class__.__name__, **self.__dict__))

  def path(self, mode):
    valids = ('model', 'best', 'info', 'log')
    if mode == 'model':
      return self.model_dir / 'seed-{:}-basic.pth'.format(self.seed)
    elif mode == 'best':
      return self.model_dir / 'seed-{:}-best.pth'.format(self.seed)
    elif mode == 'info':
      return self.log_dir / 'seed-{:}-last-info.pth'.format(self.seed)
    elif mode == 'log':
      return self.log_dir
    else:
      raise TypeError('Unknow mode = {:}, valid modes = {:}'.format(mode, valids))

  def extract_log(self):
    return self.logger_file

  def close(self):
    self.logger_file.close()
    if self.writer is not None:
      self.writer.close()

  def log(self, string, save=True, stdout=False):
    if stdout:
      sys.stdout.write(string);
      sys.stdout.flush()
    else:
      print(string)
    if save:
      self.logger_file.write('{:}\n'.format(string))
      self.logger_file.flush()

  def scalar_summary(self, tags, values, step):
    """Log a scalar variable."""
    if not self.use_tf:
      warnings.warn('Do set use-tensorflow installed but call scalar_summary')
    else:
      assert isinstance(tags, list) == isinstance(values, list), 'Type : {:} vs {:}'.format(type(tags), type(values))
      if not isinstance(tags, list):
        tags, values = [tags], [values]
      for tag, value in zip(tags, values):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

  def image_summary(self, tag, images, step):
    """Log a list of images."""
    import scipy
    if not self.use_tf:
      warnings.warn('Do set use-tensorflow installed but call scalar_summary')
      return

    img_summaries = []
    for i, img in enumerate(images):
      # Write the image to a string
      try:
        s = StringIO()
      except:
        s = BytesIO()
      scipy.misc.toimage(img).save(s, format="png")

      # Create an Image object
      img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                 height=img.shape[0],
                                 width=img.shape[1])
      # Create a Summary value
      img_summaries.append(tf.Summary.Value(tag='{}/{}'.format(tag, i), image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=img_summaries)
    self.writer.add_summary(summary, step)
    self.writer.flush()

  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""
    if not self.use_tf: raise ValueError('Do not have tensorflow')
    import tensorflow as tf

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    self.writer.add_summary(summary, step)
    self.writer.flush()


def time_for_file():
  ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
  return '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def time_string_short():
  ISOTIMEFORMAT='%Y%m%d'
  string = '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def time_print(string, is_print=True):
  if (is_print):
    print('{} : {}'.format(time_string(), string))

def convert_secs2time(epoch_time, return_str=False):    
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)  
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  if return_str:
    str = '[{:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    return str
  else:
    return need_hour, need_mins, need_secs

def print_log(print_string, log):
  #if isinstance(log, Logger): log.log('{:}'.format(print_string))
  if hasattr(log, 'log'): log.log('{:}'.format(print_string))
  else:
    print("{:}".format(print_string))
    if log is not None:
      log.write('{:}\n'.format(print_string))
      log.flush()


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.0
    self.avg = 0.0
    self.sum = 0.0
    self.count = 0.0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __repr__(self):
    return ('{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__))


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""

  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0, 'total_epoch should be greater than 0 vs {:}'.format(total_epoch)
    self.total_epoch = total_epoch
    self.current_epoch = 0
    self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
    self.epoch_losses = self.epoch_losses - 1
    self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
    self.epoch_accuracy = self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
      self.total_epoch, idx)
    self.epoch_losses[idx, 0] = train_loss
    self.epoch_losses[idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == self.epoch_accuracy[idx, 1]

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain:
      return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:
      return self.epoch_accuracy[:self.current_epoch, 1].max()

  def plot_curve(self, save_path):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    title = 'the accuracy/loss curve of train/val'
    dpi = 100
    width, height = 1600, 1000
    legend_fontsize = 10
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis * 50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis * 50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)