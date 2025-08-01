import os
import re



def early_stopping(log_value, best_value, stopping_step, best_epoch, epoch, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_epoch = epoch
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(best_epoch, best_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop, best_epoch
