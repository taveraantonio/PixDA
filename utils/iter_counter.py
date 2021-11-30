import time
import datetime


# Helper class that keeps track of training iterations
class IterationCounter:
    def __init__(self, args):
        self.args = args
        self.total_steps = args.early_stop
        self.total_steps_so_far = 0
        self.start_iter_time = 0
        self.last_iter_time = 0
        self.time_per_iter = 0
        self.time_per_training = 0
        self.epochs = 0
        self.validate_epoch = False
        self.training_end = False

    # return the iterator for the training
    def training_steps(self):
        return range(self.total_steps_so_far, self.total_steps+1)

    # record the training start and resume iteration number if continuing training
    def record_training_start(self, resume_iter=0, kd=False):
        self.training_end = False
        self.start_iter_time = time.time()
        self.total_steps_so_far = resume_iter
        self.last_iter_time = time.time()
        if kd:
            self.total_steps = self.total_steps_so_far + self.args.fine_tuning_steps

    # record one iteration
    def record_one_iteration(self):
        current_time = time.time()
        self.time_per_iter = current_time - self.last_iter_time
        self.last_iter_time = current_time
        self.total_steps_so_far += 1

    def is_training_end(self):
        return self.training_end

    # record the end of training
    def record_training_end(self):
        self.training_end = True
        current_time = time.time()
        self.time_per_training = current_time - self.start_iter_time
        print('End of training \t Time Taken: %d sec' % self.time_per_training)

    def record_one_epoch(self):
        self.epochs += 1
        self.validate_epoch = True

    def total_epochs(self):
        return self.epochs

    # print some statistics related to time of training
    def print_statistics(self):
        print('Training time per iteration: \t %.4f sec' % self.time_per_iter)
        print('Time requested to complete training: \t %s' % (str(datetime.timedelta(seconds=self.time_per_iter*self.args.early_stop))))

    def needs_saving(self):
        return self.total_steps_so_far != 0 and (self.total_steps_so_far % self.args.save_freq) == 0

    def needs_printing(self):
        return self.total_steps_so_far != 0 and (self.total_steps_so_far % self.args.print_freq) == 0

    def needs_validating(self):
        if self.validate_epoch and self.epochs % self.args.eval_epoch == 0:
            self.validate_epoch = False
            return True
        else:
            return self.total_steps_so_far != 0 and (self.total_steps_so_far % self.args.eval_freq) == 0

    def needs_validating_kd(self):
        return self.total_steps_so_far != 0 and (self.total_steps_so_far % self.args.eval_freq) == 0
