from solver import *
import pickle
from QMC_HIG import *



class simulation:
    def __init__(self):
        pass

    def set_physics(self, Nx, Nt, dt):
        self.Nx = Nx
        self.Nt = Nt
        self.dt = dt

        self.xa = 0
        self.xb = 2

    def set_optimization(self, opt_mode, optimizer,truncation):
        self.opt_mode = opt_mode
        self.optimizer = optimizer
        self.truncation = truncation

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

        xf = np.zeros((self.N*2, self.Nx - 2, 2), dtype=np.float32)
        for n in range(self.N*2):
            xf[n,:, 0] = es(1, self.Nx - 2)

        a = es(2, self.Nx - 2)
        b = es(3, self.Nx - 2)

        yf = np.zeros((self.N*2, self.Nx - 2, 2), dtype=np.float32)
        for n in range(self.N*2):
            c1 = np.random.normal()
            c2 = np.random.normal()
            p1 = np.random.uniform(0,2*np.pi)
            p2 = np.random.uniform(0,2*np.pi)
            norm = np.sqrt(c1**2+c2**2)

            f1 = c1/norm 
            f2 = c2/norm 

            state = f1*a+f2*b

            yf[n, :, 0] = np.real(state)
            yf[n, :, 1] = np.imag(state)

        x_train,x_test=np.split(xf,2)
        y_train,y_test=np.split(yf,2)

        return x_train, y_train,x_test,y_test


    def get_model_and_solver(self):
        world = World_batch(qm_step_batch, 0.0, 2.0, self.Nx)
        solver = world.get_solver_dp(self.dt, self.Nt)

        act = tf.keras.activations.tanh
        non = 20
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(14, 2)),
            tf.keras.layers.Reshape((28,)),
            tf.keras.layers.Dense(non, activation=act),
            tf.keras.layers.Dense(non, activation=act),
            tf.keras.layers.Dense(non, activation=act),
            tf.keras.layers.Dense(self.Nt, activation='linear')
        ])
        model.summary()
        return model, solver

    def get_sim_par_dict(self):
        sim_par = {'Nx':self.Nx,
                   'Nt':self.Nt,
                   'dt':self.dt,
                   'xa':self.xa,
                   'xb':self.xb,
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
        x_train, y_train,x_test,y_test = self.get_data()
        model, solver = self.get_model_and_solver()
        opt_mode_2 = self.opt_mode

        inv_opt = ig_gpinv

        QMC = QMC_optimization_framework(model, solver, ip_loss)
        QMC.set_data_set(x_train, y_train,x_test,y_test)
        QMC.set_training_parameters(opt_mode_2, self.optimizer, self.batch_size,
                                    self.learning_rate, self.stopping_criteria,
                                    self.max_number)
        QMC.set_inversion_parameters(inv_opt,self.truncation)

        results = QMC.start_training()
        sim_par = self.get_sim_par_dict()
        results['sim_par'] = sim_par
        return results, sim_par







