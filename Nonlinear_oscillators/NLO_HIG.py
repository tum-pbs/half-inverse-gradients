import numpy as np
import tensorflow as tf
import time
import functools
from gpinv_source import gpinv

def timing_dec(func):
    @functools.wraps(func)
    def wrapper_timing(*args, **kwargs):
        t0 = time.time()
        results = func(*args, **kwargs)
        duration = time.time()- t0
        return results,duration
    return wrapper_timing


def ig_gpinv(jacy, lossgrad, trunc):
    inv = gpinv(jacy, rcond=trunc)
    ig = tf.tensordot(inv, lossgrad, axes=(1, 0))
    return ig

def ig_pc(jacy, lossgrad, trunc):
    inv = tf.linalg.pinv(jacy, rcond=trunc)
    ig = tf.tensordot(inv, lossgrad, axes=(1, 0))
    return ig


class NLO_optimization_framework():
    def __init__(self, model,solver,loss_function):
        self.model = model
        self.solver = solver
        self.loss_function = loss_function

        self.initial_weights = model.get_weights()
        self.weight_shapes = [weight_tensor.shape for weight_tensor in model.trainable_weights]

    def reset_weights(self):
        self.model.set_weights(self.initial_weights)

    def set_data_set(self, x_data_train, y_data_train, x_data_test=None, y_data_test=None):
        self.x_train = x_data_train
        self.y_train = y_data_train
        self.N = x_data_train.shape[0]  
        self.y_dim = y_data_train.shape[1]
        if x_data_test is not None:
            self.test = True
            self.x_test = x_data_test
            self.y_test = y_data_test
            self.N_test = x_data_test.shape[0]  
        else:
            self.test = False

    def set_inversion_parameters(self, linear_solve,truncation):

        self.truncation = truncation

        self.linear_solve = lambda x,y: linear_solve(x,y,self.truncation)

    def set_training_parameters(self, opt_mode, optimizer, batch_size, learning_rate,
                                stopping_criteria, max_number):
        # opt_mode: 'GD' or 'IG'
        # stopping_criteria: 'epochs' or 'sim_time'

        self.opt_mode = opt_mode

        self.max_number = max_number
        self.stopping_criteria = stopping_criteria

        self.batch_size = batch_size
        assert self.N % batch_size == 0, 'N / bs '
        self.number_of_batches = self.N // batch_size

        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.learning_rate)


    def computation(self,x_batch,y_batch):
        control_batch = self.model(y_batch)
        y_prediction_batch = self.solver(x_batch,control_batch)
        loss, loss_batch = self.loss_function(y_batch,y_prediction_batch)
        return loss, loss_batch, y_prediction_batch

    @tf.function
    @timing_dec
    def compute_derivatives(self,x_batch,y_batch):

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss, loss_batch, y_prediction_batch = self.computation(x_batch,y_batch)

            loss_per_dp = loss / self.batch_size

        if self.opt_mode == 'GD':
            grad = tape.gradient(loss_per_dp, self.model.trainable_variables)
            return grad
        elif self.opt_mode == 'IG':
            jacy = tape.jacobian(y_prediction_batch, self.model.trainable_variables, experimental_use_pfor=True)
            loss_grad = tape.gradient(loss_per_dp, y_prediction_batch)
            return jacy, loss_grad
        else:
            return 0

    @timing_dec
    def process_derivatives(self,derivatives):
        if self.opt_mode == 'GD':
            return derivatives

        elif self.opt_mode=='IG':
            jacy_batch, loss_grad_batch = derivatives

            flat_jacy_list = [tf.reshape(jac, (self.batch_size * self.y_dim, -1)) for jac in jacy_batch]
            flat_jacy = tf.concat(flat_jacy_list, axis=1)
            flat_grad = tf.reshape(loss_grad_batch, (-1,))

            processed_derivatives = self.linear_solve(flat_jacy, flat_grad)

            update_list = []
            l1 = 0
            for k, shape in enumerate(self.weight_shapes):
                l2 = l1 + np.prod(shape)
                upd = processed_derivatives[l1:l2]
                upd = np.reshape(upd, shape)
                update_list.append(upd)
                l1 = l2

            return update_list

        else:
            return 0

    @timing_dec
    def apply_update(self,update_list):
        self.optimizer.apply_gradients(zip(update_list, self.model.trainable_weights))

    #@timing_dec
    def mini_batch_update(self, x_batch, y_batch):

        # compute derivatives
        derivatives, t_cd = self.compute_derivatives(x_batch, y_batch)

        # process derivatives
        update_list, t_pd = self.process_derivatives(derivatives)

        # update application
        _, t_au = self.apply_update(update_list)

        return t_cd, t_pd, t_au

    @timing_dec
    def epoch_update(self):
        ts = []
        for batch_index in range(self.number_of_batches):
            position = batch_index * self.batch_size
            x_batch = self.x_train[position:position + self.batch_size]
            y_batch = self.y_train[position:position + self.batch_size]
            ts.append(self.mini_batch_update(x_batch, y_batch))

        return (np.array(ts).sum(axis=0))

    def eval(self, print_cond=True):
        if print_cond==True: print('Ep: ', self.epoch,' SimTime: ',self.simulation_time,' EpDuration: ',self.t_epoch )

        train_loss = self.computation(self.x_train,self.y_train)[0]
        train_loss_per_dp = train_loss / self.N
        self.results_train_loss.append(train_loss_per_dp)
        if print_cond == True: print('TrainL:', train_loss_per_dp.numpy())

        if self.test == True:
            test_loss = self.computation(self.x_test, self.y_test)[0]
            test_loss_per_dp = test_loss / self.N
            self.results_test_loss.append(test_loss_per_dp)
            if print_cond==True: print('TestL:',test_loss_per_dp.numpy())



    def reset_results(self):
        self.epochs = []
        self.simulation_times = []
        self.time_per_epoch = []
        self.results_train_loss = []
        self.results_test_loss = []

    def results_as_dict(self):
        res = {}
        res['epochs'] = np.array(self.epochs)
        res['train_loss'] = np.array(self.results_train_loss)
        res['test_loss'] = np.array(self.results_test_loss)
        res['epoch_time'] = np.array(self.time_per_epoch)
        res['simulation_time'] = np.array(self.simulation_times)
        return res

    def start_training(self, new_training=True):

        if new_training:
            self.reset_weights()
            self.reset_results()
            init_time = time.time()

            self.simulation_time = time.time()- init_time
            self.simulation_times.append(self.simulation_time)
            self.epoch = 0
            self.t_epoch = 0
            self.epochs.append(self.epoch)

            self.eval()

        current_number = 0

        while current_number<self.max_number:

            ts, self.t_epoch = self.epoch_update()

            self.epoch += 1
            self.simulation_time += self.t_epoch

            self.time_per_epoch.append(self.t_epoch)
            self.epochs.append(self.epoch)
            self.simulation_times.append(self.simulation_time)

            self.eval()

            if self.stopping_criteria == 'sim_time':
                current_number += self.t_epoch
            elif self.stopping_criteria == 'epochs':
                current_number = self.epoch


        total_training_time = time.time() - init_time
        print('Total simulation time: ',total_training_time, ' Total epochs: ', self.epoch)

        results = self.results_as_dict()
        return results


        


