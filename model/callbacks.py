from fastNLP.core.callback import Callback
from fastNLP import DataSet, Tester
import fitlog
from copy import deepcopy


class FitlogCallback(Callback):

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=1, log_exception=True,
                 raise_threshold=0, better_dev_eval=True):

        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        self.raise_threshold = raise_threshold

        assert isinstance(log_loss_every, int) and log_loss_every>=0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        self.better_dev_eval = better_dev_eval

    def on_train_begin(self):
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.trainer.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.kwargs.get('test_use_tqdm', self.trainer.use_tqdm),
                                sampler=self.trainer.kwargs.get('test_sampler', None))
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

    def on_backward_begin(self, loss):
        if self._log_loss_every >0:
            self._avg_loss += loss.item()
            if self.step %self._log_loss_every==0:
                fitlog.add_loss(self._avg_loss /self._log_loss_every *self.update_every, name='loss', step=self.step, epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        indicator, indicator_val = _check_eval_results(eval_result, metric_key=metric_key)
        if indicator_val < self.raise_threshold:
            raise RuntimeError("The program has been running off.")

        if len(self.testers) > 0:
            do_eval = True
            if self.better_dev_eval:
                if not better_result:
                    do_eval = False
            if do_eval:
                for idx, (key, tester) in enumerate(self.testers.items()):
                    try:
                        eval_result = tester.test()
                        if self.verbose != 0:
                            self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                            self.pbar.write(tester._format_eval_results(eval_result))
                        fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                        if idx == 0:
                            indicator, indicator_val = _check_eval_results(eval_result, metric_key=self.trainer.metric_key)
                            if indicator_val>self.best_test_metric_sofar:
                                self.best_test_metric_sofar = indicator_val
                                self.best_test_epoch = self.epoch
                                self.best_test_sofar = eval_result

                        if better_result:
                            self.best_dev_test = eval_result
                            self.best_dev_epoch = self.epoch
                            fitlog.add_best_metric(eval_result, name=key)
                    except Exception as e:
                        self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                        raise e

    def on_train_end(self):
        if self.best_test_sofar:
            line1 = "Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_test_sofar, self.best_test_epoch)
            self.logger.info(line1)
            fitlog.add_to_line(line1)
        if self.best_dev_test:
            line2 = "Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_dev_test, self.best_dev_epoch)
            self.logger.info(line2)
            fitlog.add_to_line(line2)
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


def _check_eval_results(metrics, metric_key=None):

    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]

        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val


from fastNLP import WarmupCallback as FWarmupCallback
import math
class WarmupCallback(FWarmupCallback):
    def __init__(self, warmup=0.1, schedule='constant'):

        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        elif schedule == 'inverse_square':
            self.get_lr = self._get_inverse_square_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_inverse_square_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((math.sqrt(progress) - 1.) / (math.sqrt(self.warmup) - 1.), 0.)


