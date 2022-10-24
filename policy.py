import numpy as np
import tensorflow as tf
import ray.experimental
import datetime
import sklearn
from ray.experimental.tf_utils import TensorFlowVariables
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Policy(object):
    """ Policy Neural Network """

    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult,  clipping_range=0.2, temp=2.0):
        """
        :param obs_dim: num observation dimensions
        :param act_dim: num action dimensions
        :param kl_targ: target KL divergence between pi_old and pi_new
        :param hid1_mult: size of first hidden layer, multiplier of obs_dim
        :param clipping_range:
        :param temp: temperature parameter
        """
        self.temp = temp
        self.beta = 3  # dynamically adjusted D_KL loss multiplier
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.epochs = 3
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clipping_range = clipping_range

        self._build_graph()
        #self._init_session()



    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._loss_train_op()
            self._loss_initial_op()
            self.init = tf.global_variables_initializer()
            self.sess = tf.Session(graph=self.g)
            self.variables = TensorFlowVariables(self.loss, self.sess)
            self.sess.run(self.init)


    def _placeholders(self):
        """ Define placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')

        self.act_ph = tf.placeholder(tf.int32, (None, len(self.act_dim)), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta') # learning rate:
        self.old_act_prob_ph = []
        for i in range(len(self.act_dim)):# split the output layer over queuing stations
            self.old_act_prob_ph.append(tf.placeholder(tf.float32, (None, self.act_dim[i]), 'old_act_prob'+str(i)))



    def _policy_nn(self):
        """ Neural Network architecture for policy approximation function
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        hid3_size = len(self.act_dim) * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size
        self.lr = 9. * 10**(-4)  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                 stddev=np.sqrt(1 / hid1_size)), name="h2")

        out_last = []
        act_prob_out = []
        for i in range(len(self.act_dim)):
            out_last.append(tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3"+str(i)))

            act_prob_out.append( tf.layers.dense(tf.divide(out_last[i], self.temp), self.act_dim[i], tf.nn.softmax,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)), name="act_prob"+str(i)))

        self.act_prob_out = [act_prob_out[i] for i in range(len(act_prob_out))]

    def _logprob(self):
        """
        Calculate probabilities using previous step's model parameters and new parameters being trained.
        """
        act_probs = []
        act_probs_old = []
        for i in range(len(self.act_dim)):
            act_probs.append(self.act_prob_out[i])
            act_probs_old.append(self.old_act_prob_ph[i])
        for i in range(len(self.act_dim)):
            # probabilities of actions which agent took with policy
            act_probs[i] = tf.reduce_sum(act_probs[i] * tf.one_hot(indices=self.act_ph[:,i], depth=act_probs[i].shape[1]), axis=1)


            # probabilities of actions which agent took with old policy
            act_probs_old[i] = tf.reduce_sum(act_probs_old[i] * tf.one_hot(indices=self.act_ph[:,i], depth=act_probs_old[i].shape[1]), axis=1)



        self.act_probs_old = np.prod(act_probs_old, axis=0)
        self.act_probs = np.prod(act_probs, axis = 0)


    def _kl_entropy(self):
        """
        Calculate KL-divergence between old and new distributions
        """

        self.entropy = 0
        self.kl = 0
        for i in range(len(self.act_dim)):
            kl = tf.reduce_sum(self.act_prob_out[i] * (tf.log(tf.clip_by_value(self.act_prob_out[i], 1e-10, 1.0))
                                                       - tf.log(tf.clip_by_value(self.old_act_prob_ph[i], 1e-10, 1.0))), axis=1)
            entropy = tf.reduce_sum(self.act_prob_out[i] * tf.log(tf.clip_by_value(self.act_prob_out[i], 1e-10, 1.0)), axis=1)
            self.entropy += -tf.reduce_mean(entropy, axis=0)# sum of entropy of pi(obs)
            self.kl += tf.reduce_mean(kl, axis=0)


    def _loss_train_op(self):
        """
        Calculate the PPO loss function
        """

        ratios = tf.exp(tf.log(self.act_probs) - tf.log(self.act_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clipping_range, clip_value_max=1 + self.clipping_range)
        loss_clip = tf.minimum(tf.multiply(self.advantages_ph, ratios), tf.multiply(self.advantages_ph, clipped_ratios))

        self.loss = -tf.reduce_mean(loss_clip)#- self.entropy*0.0001
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)


    def sample(self, obs, stochastic=True):
        """
        :param obs: state
        :return: if stochastic=True returns pi(a|x), else returns distribution with prob=1 on argmax[ pi(a|x) ]
        """

        feed_dict = {self.obs_ph: obs}
        if stochastic:
            return self.sess.run(self.act_prob_out, feed_dict=feed_dict)
        else:
            determ_prob = []
            for i in range(len(self.act_dim)):
                pr = self.sess.run(self.act_prob_out[i], feed_dict=feed_dict)
                inx = np.argmax(pr)
                ar = np.zeros(self.act_dim[i])
                ar[inx] = 1
                determ_prob.extend([ar[np.newaxis]])
            return determ_prob


    def update(self, observes, actions, advantages, logger):
        """
        Policy Neural Network update
        :param observes: states
        :param actions: actions
        :param advantages: estimation of antantage function at observed states
        :param logger: statistics accumulator
        """

        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.lr_ph: self.lr * self.lr_multiplier}

        old_act_prob_np = self.sess.run(self.act_prob_out, feed_dict)  # actions probabilities w.r.t the current policy
        for i in range(len(self.act_dim)):
            feed_dict[self.old_act_prob_ph[i]] = old_act_prob_np[i]

        loss = 0
        kl = 0
        entropy = 0
        for e in range(self.epochs): # training
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                print('early stopping: D_KL diverges badly')
                break

        # actions probabilities w.r.t the new and old (current) policies
        act_probs, act_probs_old = self.sess.run([self.act_probs, self.act_probs_old], feed_dict)
        ratios = np.exp(np.log(act_probs) - np.log(act_probs_old))
        if self.clipping_range is not None:
            clipping_range = self.clipping_range
        else:
            clipping_range = 0

        logger.log({'PolicyLoss': loss,
                    'Clipping' : clipping_range,
                    'Max ratio': max(ratios),
                    'Min ratio': min(ratios),
                    'Mean ratio': np.mean(ratios),
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier })





    def run_episode(self, network, scaler, time_steps, cycles_num, skipping_steps, initial_state, rpp = False):
        """
        One episode simulation
        :param network: queuing network
        :param scaler: normalization values
        :param time_steps: max number of time steps
        :param skipping_steps: number of steps for which control is fixed
        :param initial_state: initial state for the episode
        :return: collected data
        """


        policy_buffer = {} # save action disctribution of visited states

        total_steps = 0 # count steps
        action_optimal_sum = 0 # count actions that coinside with the optimal policy
        total_zero_steps = 0 # count states for which all actions are optimal


        observes = np.zeros((time_steps, network.buffers_num))
        actions = np.zeros((time_steps, network.stations_num), 'int8')
        actions_glob = np.zeros((time_steps,  ), 'int8')
        rewards = np.zeros((time_steps, 1))
        unscaled_obs = np.zeros((time_steps, network.buffers_num), 'int32')
        unscaled_last = np.zeros((time_steps, network.buffers_num), 'int32')
        array_actions = []
        for i in range(network.stations_num):
            array_actions.append(np.zeros((time_steps, self.act_dim[i])))


        t = 0
        av_cycle_length = 0
        for cyc in range(cycles_num):
            scale, offset = scaler.get()

            ##### modify initial state according to the method of intial states generation######
            if scaler.initial_states_procedure =='previous_iteration':
                if sum(initial_state[:-1]) > 300 :
                    initial_state = np.zeros(network.buffersNum+1, 'int8')
                state = np.asarray(initial_state[:-1],'int32')
            else:
                state = np.asarray(initial_state, 'int32')

            ###############################################################
            cycle_length = 0

            init = True
            while init or np.sum(state)!=0: # run until visit to the empty state (regenerative state)
                init = False
                cycle_length += 1

                # if t % 50000 == 0:
                #     print('have not reached, current t is: {}, cyc is:  ', t, cyc)
                unscaled_obs[t] = state
                state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations

                ###### compute action distribution according to Policy Neural Network for state###


                if tuple(state) not in policy_buffer:
                    if rpp:
                        act_distr = network.random_proportional_policy_distr(state)
                    else:
                        act_distr = self.sample([state_input])
                    policy_buffer[tuple(state)] = act_distr
                distr = policy_buffer[tuple(state)][0][0] # distribution for each station

                array_actions[0][t] = distr
                for ar_i in range(1, network.stations_num):
                    # iterate through each station, get all possible action combinations
                    distr = [a * b for a in distr for b in policy_buffer[tuple(state)][ar_i][0]]
                    array_actions[ar_i][t] = policy_buffer[tuple(state)][ar_i][0]
                distr = distr / sum(distr)
                ############################################

                act_ind = np.random.choice(len(distr), 1, p=distr) # sample action according to distribution 'distr'
                action_full = network.dict_absolute_to_binary_action[act_ind[0]]
                action_for_server = network.dict_absolute_to_per_server_action[act_ind[0]]

                ######### check optimality of the sampled action ################
                # if len(state)==3 and state[0]<140 and state[1]<140 and state[2]<140:
                #     if state[0]==0 or state[2]==0:
                #         total_zero_steps += 1

                #     action_optimal = network.comparison_policy[tuple(state)]
                #     if all(action_full == action_optimal) or state[0]==0 or state[2]==0:
                #         action_optimal_sum += 1

                # else:
                #     action_optimal = network.comparison_policy[tuple(state>0)]
                #     if all(action_full == action_optimal):
                #         action_optimal_sum += 1
                #######################



                rewards[t] = -np.sum(state)

                unscaled_last[t] = state
                state = network.next_state(state, action_full)
                actions[t] = action_for_server
                observes[t] = state_input
                actions_glob[t] = act_ind[0]


                t+=1
            av_cycle_length = 1/(cyc+1)*cycle_length + cyc/(cyc+1)*av_cycle_length



        total_steps += len(actions[:t])
        # record simulation

        trajectory = {#'observes': observes,
                          'actions': actions[:t],
                          'actions_glob': actions_glob[:t],
                          'rewards': rewards[:t] / skipping_steps,
                          'unscaled_obs': unscaled_obs[:t],
                          'unscaled_last': unscaled_last[:t]
                      }


        print('Network:', network.network_name + '.', 'time of an episode:',
               'Average cost:', -np.mean(trajectory['rewards']),
                'Average cycle lenght:', av_cycle_length)

        return trajectory, total_steps, action_optimal_sum, total_zero_steps, array_actions

    def run_episode_alg3(self, network, scaler, time_steps, cycles_num, skipping_steps, initial_state, rpp = False):
        """
        One episode simulation
        :param network: queuing network
        :param scaler: normalization values
        :param time_steps: max number of time steps
        :param skipping_steps: number of steps for which control is fixed
        :param initial_state: initial state for the episode
        :return: collected data
        """


        policy_buffer = {} # save action disctribution of visited states

        total_steps = 0 # count steps
        action_optimal_sum = 0 # count actions that coinside with the optimal policy
        total_zero_steps = 0 # count states for which all actions are optimal


        observes = np.zeros((time_steps, network.buffers_num))
        actions = np.zeros((time_steps, network.stations_num), 'int8')
        actions_glob = np.zeros((time_steps,  ), 'int8')
        rewards = np.zeros((time_steps, 1))
        unscaled_obs = np.zeros((time_steps, network.buffers_num), 'int32')
        unscaled_last = np.zeros((time_steps, network.buffers_num), 'int32')
        array_actions = []
        for i in range(network.stations_num):
            array_actions.append(np.zeros((time_steps, self.act_dim[i])))


        t = 0
        av_cycle_length = 0
        for cyc in range(cycles_num):
            if t >= time_steps:
                break
            scale, offset = scaler.get()

            ##### modify initial state according to the method of intial states generation######
            if scaler.initial_states_procedure =='previous_iteration':
                if sum(initial_state[:-1]) > 300 :
                    initial_state = np.zeros(network.buffersNum+1, 'int8')
                state = np.asarray(initial_state[:-1],'int32')
            else:
                state = np.asarray(initial_state, 'int32')

            ###############################################################
            cycle_length = 0

            init = True
            while init or np.sum(state)!=0: # run until visit to the empty state (regenerative state)
                if t >= time_steps:
                    break
                init = False
                cycle_length += 1

                # if t % 50000 == 0:
                #     print('have not reached, current t is: {}, cyc is:  ', t, cyc)
                unscaled_obs[t] = state
                state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations

                ###### compute action distribution according to Policy Neural Network for state###
                if tuple(state) not in policy_buffer:
                    if rpp:
                        act_distr = network.random_proportional_policy_distr(state)
                    else:
                        act_distr = self.sample([state_input])
                    policy_buffer[tuple(state)] = act_distr
                distr = policy_buffer[tuple(state)][0][0] # distribution for each station

                array_actions[0][t] = distr
                for ar_i in range(1, network.stations_num):
                    # iterate through each station, get all possible action combinations
                    distr = [a * b for a in distr for b in policy_buffer[tuple(state)][ar_i][0]]
                    array_actions[ar_i][t] = policy_buffer[tuple(state)][ar_i][0]
                distr = distr / sum(distr)
                ############################################

                act_ind = np.random.choice(len(distr), 1, p=distr) # sample action according to distribution 'distr'
                action_full = network.dict_absolute_to_binary_action[act_ind[0]]
                action_for_server = network.dict_absolute_to_per_server_action[act_ind[0]]

                rewards[t] = -np.sum(state)

                unscaled_last[t] = state
                state = network.next_state(state, action_full)
                actions[t] = action_for_server
                observes[t] = state_input
                actions_glob[t] = act_ind[0]


                t+=1
            av_cycle_length = 1/(cyc+1)*cycle_length + cyc/(cyc+1)*av_cycle_length

        total_steps += len(actions[:t])
        # record simulation

        trajectory = {#'observes': observes,
                          'actions': actions[:t],
                          'actions_glob': actions_glob[:t],
                          'rewards': rewards[:t] / skipping_steps,
                          'unscaled_obs': unscaled_obs[:t],
                          'unscaled_last': unscaled_last[:t]
                      }


        print('Network:', network.network_name + '.', 'time of an episode:',
               'Average cost:', -np.mean(trajectory['rewards']),
                'Average cycle lenght:', av_cycle_length)
        for i in range(len(array_actions)):
            array_actions[i] = array_actions[i][:t]

        return trajectory, total_steps, action_optimal_sum, total_zero_steps, array_actions



    def policy_performance_cycles(self, network, scaler, initial_state, id, batch_num = 1000000, stochastic=True):


        length_batch = np.zeros(batch_num)
        cost_batch = np.zeros(batch_num)
        policy_buffer = {}




        scale, offset = scaler.get()


        for batch in range(batch_num):

            if batch % 10000 ==1 and id==1:
                print(int(batch/ batch_num*100), '% is done')

            if scaler.initial_states_procedure =='previous_iteration':
                if sum(initial_state[:-1]) > 300 :
                    initial_state = np.zeros(network.buffers_num+1, 'int8')


                state = np.asarray(initial_state[:-1],'int32')
            else:
                state = np.asarray(initial_state, 'int32')




            k = 0
            while np.sum(state)!=0 or k==0:
                k += 1

                state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations


                if tuple(state) not in policy_buffer:

                    act_distr = self.sample([state_input], stochastic)
                    policy_buffer[tuple(state)] = act_distr
                distr = policy_buffer[tuple(state)][0][0]  # distribution for each station


                for ar_i in range(1, network.stations_num):
                    distr = [a * b for a in distr for b in policy_buffer[tuple(state)][ar_i][0]]

                distr = distr / sum(distr)
                act_ind = np.random.choice(len(distr), 1, p=distr)

                action_full = network.dict_absolute_to_binary_action[act_ind[0]]

                #average_performance = 1/(t+1) * np.sum(state) + t / (t+1) * average_performance
                length_batch[batch] +=1
                cost_batch[batch] +=np.sum(state)



                state = network.next_state(state, action_full)




        average_performance = np.mean(cost_batch)/ np.mean(length_batch)

        s11 = np.var(length_batch)
        s22 = np.var(cost_batch)
        s12 = 1/ (batch_num-1) * np.sum((length_batch - np.mean(length_batch))*(cost_batch - np.mean(cost_batch)))

        s2 = np.mean(cost_batch)**2*s11/(np.mean(length_batch)**4) + s22/(np.mean(length_batch)**2) - 2*np.mean(cost_batch)*s12/(np.mean(length_batch)**3)
        ci =  1.96*np.sqrt(s2 / batch_num)


        #optimal_ratio = action_optimal_sum / total_steps

        print(id, ' average_' + str(average_performance)+'+-' +str(ci))
        return average_performance, id, ci


    def policy_performance_sixclasses(self, network, scaler, initial_state, id, episode_length = 5*10**6, stochastic=True):
        batch_num = 50

        cost_batch = [] # the cost for each step in the episode_length
        policy_buffer = {}

        scale, offset = scaler.get()

        state = np.asarray(initial_state, 'int32')

        arr_cnt = 0 # number of arrival events that have occurred
        while arr_cnt < episode_length: # we sample until the episode_length arr event occurs
            if arr_cnt % (episode_length/100) ==1 and id==1:
                print(int(arr_cnt/ episode_length*100), '% is done')

            # calculate action taking distribution
            state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations
            if tuple(state) not in policy_buffer:
                act_distr = self.sample([state_input], stochastic)
                policy_buffer[tuple(state)] = act_distr
            distr = policy_buffer[tuple(state)][0][0]  # distribution for each station
            for ar_i in range(1, network.stations_num):
                # iterate through each station, get all possible action combinations
                distr = [a * b for a in distr for b in policy_buffer[tuple(state)][ar_i][0]]
            distr = distr / sum(distr)

            # Sample the action to take, record current step's cost
            act_ind = np.random.choice(len(distr), 1, p=distr)
            action_full = network.dict_absolute_to_binary_action[act_ind[0]]
            cost_batch.append(np.sum(state))

            # Get next state; determine whether an arrival happened
            state, is_arrival = network.next_state_nClassesPerformance(state, action_full)
            if is_arrival:
                arr_cnt += 1

        # Calculate mean and CI
        cost_batch = np.array(cost_batch)
        average_performance = np.mean(cost_batch)

        batch_len = len(cost_batch) // batch_num
        rem = len(cost_batch) % batch_num
        reshaped_costs = np.reshape(cost_batch[:-rem], (batch_num, batch_len))
        batch_mean = np.mean(reshaped_costs, axis=1) # the mean for each batch (batch_num,)
        batch_mean[-1] = (batch_mean[-1] * batch_len + np.sum(cost_batch[-rem:])) / (batch_len + rem) # reinclude remainder
        ci_down_up = np.percentile(batch_mean, [2.5, 97.5])
        ci = (ci_down_up[1] - ci_down_up[0]) / 2

        print(id, ' average_' + str(average_performance)+'+-' +str(ci))
        return average_performance, id, ci


    def _loss_initial_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_init = optimizer.minimize(self.kl)

    def initilize_rpp(self, observes, action_distr, batch_size=256):
        """
        Policy Neural Network update
        :param observes: states
        :param actions: actions
        :param advantages: estimation of antantage function at observed states
        :param logger: statistics accumulator
        """

        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches

        for e in range(20):
            x_train, *y_train = sklearn.utils.shuffle(observes, *action_distr)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.beta_ph: self.beta,
                             self.lr_ph: self.lr * self.lr_multiplier,
                             self.act_ph: 0 * observes[:, 0:2],
                             self.advantages_ph: 0 * observes[:, 0]}
                for i in range(len(self.act_dim)):
                    feed_dict[self.old_act_prob_ph[i]] = y_train[i][start:end, :]

                self.sess.run(self.train_init, feed_dict)



    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def get_obs_dim(self):
        return self.obs_dim

    def get_hid1_mult(self):
        return self.hid1_mult

    def get_act_dim(self):
        return self.act_dim

    def get_weights(self):
        return self.variables.get_weights()

    def set_weights(self, weights):
        # Set the weights in the network.
        self.variables.set_weights(weights)

    def get_kl_targ(self):
        return self. kl_targ
