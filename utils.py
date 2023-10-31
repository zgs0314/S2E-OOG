import numpy as np
import os
import shutil
import time
import sys
import logging
import logging.handlers as handlers

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp

class SizedTimedRotatingFileHandler(handlers.TimedRotatingFileHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size, or at certain
    timed intervals
    """
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None,
                 delay=0, when='h', interval=1, utc=False):
        # If rotation/rollover is wanted, it doesn't make sense to use another
        # mode. If for example 'w' were specified, then if there were multiple
        # runs of the calling application, the logs from previous runs would be
        # lost if the 'w' is respected, because the log file would be truncated
        # on each run.
        if maxBytes > 0:
            mode = 'a'
        handlers.TimedRotatingFileHandler.__init__(
            self, filename, when, interval, backupCount, encoding, delay, utc)
        self.maxBytes = maxBytes

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        if self.stream is None:                 # delay was set...
            self.stream = self._open()
        if self.maxBytes > 0:                   # are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  #due to non-posix-compliant Windows feature
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1
        t = int(time.time())
        if t >= self.rolloverAt:
            return 1
        return 0


def init_logger(logger_name='', log_file='', log_level='', print_console=False):

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)


    error_handler = logging.StreamHandler(sys.stdout)
    error_handler.setLevel(logging.ERROR)

    # create a logging format
    formatter = logging.Formatter('%(name)s-logging.%(levelname)s-%(thread)d-%(asctime)s-%(message)s')
    handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    daily_handler=SizedTimedRotatingFileHandler(log_file, when='midnight')
    daily_handler.setLevel(logging.INFO)
    daily_handler.setFormatter(formatter)

    # add the handlers to the logger
    #logger.addHandler(handler)
    logger.addHandler(error_handler)
    logger.addHandler(daily_handler)

    if print_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def generate_search_space(trans_num_layer, use_raw):
    from itertools import product
    TRANS_OPTION = ['linear', 'rel-linear', 'conv', 'rel-conv', 'gate' ]# 'rel-linear-conv' 
    AGG_OPTION = ['sum', 'mean', 'max', 'min', 'attention']
    TRANS_OPTION_2 = ['linear', 'conv', 'gate']
    arch_list = []
    for trans_num in range(1, trans_num_layer+1):
        if trans_num == 1:
            res = list(product(TRANS_OPTION))
        elif trans_num == 2:
            res = list(product(TRANS_OPTION, TRANS_OPTION))
        elif trans_num == 3:
            res = list(product(TRANS_OPTION, TRANS_OPTION, TRANS_OPTION))
        elif trans_num == 4:
            res = list(product(TRANS_OPTION, TRANS_OPTION, TRANS_OPTION, TRANS_OPTION))
        elif trans_num == 5:
            res = list(product(TRANS_OPTION, TRANS_OPTION, TRANS_OPTION, TRANS_OPTION, TRANS_OPTION))
        else:
            pass
        res = [list(r) for r in res]
        for r in res:
            if use_raw:
                for agg in AGG_OPTION:
                    for transop2 in TRANS_OPTION_2:
                        res_copy = r.copy()
                        res_copy.append(agg)
                        res_copy.append(transop2)
                        arch_list.append(res_copy)
            else:
                for agg in AGG_OPTION:
                    res_copy = r.copy()
                    res_copy.append(agg)
                    arch_list.append(res_copy)
    arch_str_list = ['||'.join(arch) for arch in arch_list]
    return arch_str_list