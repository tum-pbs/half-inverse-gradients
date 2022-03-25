from world_batch import World_batch, l2_loss
from NLO_HIG import *
from ODEs_batch import coup_nonlin_osc_batch
from int_methods_batch import rk4_batch



class simulation:
    def __init__(self):
        pass

    def set_physics(self, Nx, Nt, dt):
        self.Nx = Nx
        self.Nt = Nt
        self.dt = dt


    def set_optimization(self, opt_mode, optimizer,truncation,inversion):
        self.opt_mode = opt_mode
        self.optimizer = optimizer
        self.truncation = truncation
        self.inversion = inversion

    def set_learning(self, N, batch_size, learning_rate, stopping_criteria, max_number):
        self.N = N
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.stopping_criteria = stopping_criteria
        self.max_number = max_number

    def set_seed(self, seed):
        self.seed = seed

    def get_data(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        x = np.random.rand(2*self.N, 2 * self.Nx).astype(np.float32)
        y = x

        x_train,x_test=np.split(x,2)
        y_train,y_test=np.split(y,2)

        return x_train, y_train,x_test,y_test



    def get_model_and_solver(self):

        world = World_batch(coup_nonlin_osc_batch, rk4_batch)
        solver = world.get_solver_dp(self.dt, self.Nt)

        act = tf.keras.activations.relu
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2*self.Nx)),
            tf.keras.layers.Dense(20, activation=act),
            tf.keras.layers.Dense(20, activation=act),
            tf.keras.layers.Dense(20, activation=act),
            tf.keras.layers.Dense(self.Nt, activation='linear')
        ])
        model.summary()
        return model, solver

    def get_sim_par_dict(self):

        sim_par = {'Nx':self.Nx,
                   'Nt':self.Nt,
                   'dt':self.dt,
                   'opt_mode':self.opt_mode,
                   'optimizer':self.optimizer.__name__,
                   'truncation':self.truncation,
                   'N':self.N,
                   'batch_size':self.batch_size,
                   'learning_rate':self.learning_rate,
                   'stopping_criteria':self.stopping_criteria,
                   'max_number':self.max_number,
                   'seed':self.seed}

        return sim_par


    def start(self):
        x_train,y_train,x_test,y_test = self.get_data()
        model, solver = self.get_model_and_solver()

        NLO = NLO_optimization_framework(model, solver, l2_loss)
        NLO.set_data_set(x_train, y_train,x_test,y_test)
        NLO.set_training_parameters(self.opt_mode, self.optimizer, self.batch_size,
                                    self.learning_rate, self.stopping_criteria,
                                    self.max_number)

        ig = None
        if self.inversion =="GN":
            ig =  ig_pc
        if self.inversion =="HIG":
            ig = ig_gpinv         

        NLO.set_inversion_parameters(ig,self.truncation)

        results = NLO.start_training()
        sim_par = self.get_sim_par_dict()
        results['sim_par'] = sim_par
        return results, sim_par







