import sys, os
import numpy as np
import time
import gym
import tensorflow as tf
from spinup.utils.logx import EpochLogger

from optimistic_actor_critic.array_observation.oac_disc.core import *
from optimistic_actor_critic.plot_progress import plot_progress

# configure gpu use and supress tensorflow warnings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""
Store the observations in ring buffer type array of size m
"""
class StateBuffer:
    def __init__(self,m):
        self.m = m

    def init_state(self, init_obs):
        self.current_state = np.concatenate([init_obs]*self.m, axis=0)
        return self.current_state

    def append_state(self, obs):
        new_state = np.concatenate( (self.current_state, obs), axis=0)
        self.current_state = new_state[obs.shape[0]:]
        return self.current_state

"""
Process features of the environment
"""
def process_observation(o, obs_dim, observation_type):
    if observation_type == 'Discrete':
        o = np.eye(obs_dim)[o]
    return o

def process_action(a, act_dim):
    one_hot_a = np.eye(act_dim)[a]
    return one_hot_a

def process_reward(reward):
    # apply clipping here if needed
    return reward

"""
Linear annealing from start to stop value based on current step and max_steps
"""
def linear_anneal(current_step, start=0.1, stop=1.0, steps=1e6):
    if current_step<=steps:
        eps = stop + (start - stop) * (1 - current_step/steps)
    else:
        eps=start
    return eps

"""
Clip gradient whilst handling None error
"""
def ClipIfNotNone(grad, grad_clip_val):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -grad_clip_val, grad_clip_val)


"""

Discrete Optimistic Actor-Critic

"""
def oac(env_fn, actor_critic=mlp_actor_critic,
                logger_kwargs=dict(),
                network_params=dict(),
                rl_params=dict()):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # control params
    seed                = rl_params['seed']
    epochs              = rl_params['epochs']
    steps_per_epoch     = rl_params['steps_per_epoch']
    replay_size         = rl_params['replay_size']
    batch_size          = rl_params['batch_size']
    start_steps         = rl_params['start_steps']
    max_ep_len          = rl_params['max_ep_len']
    save_freq           = rl_params['save_freq']
    render              = rl_params['render']

    # rl params
    gamma               = rl_params['gamma']
    polyak              = rl_params['polyak']
    lr                  = rl_params['lr']
    state_hist_n        = rl_params['state_hist_n']
    grad_clip_val       = rl_params['grad_clip_val']

    # entropy params
    alpha                = rl_params['alpha']
    target_entropy_start = rl_params['target_entropy_start']
    target_entropy_stop  = rl_params['target_entropy_stop']
    target_entropy_steps = rl_params['target_entropy_steps']

    # optimistic exploration params
    use_opt             = rl_params['use_opt']
    beta_UB             = rl_params['beta_UB']
    beta_LB             = rl_params['beta_LB']
    delta               = rl_params['delta']
    opt_lr              = rl_params['opt_lr']
    max_opt_steps       = rl_params['max_opt_steps']

    train_env, test_env = env_fn(), env_fn()
    obs_space = train_env.observation_space
    act_space = train_env.action_space

    try:
        obs_dim = obs_space.n
        observation_type = 'Discrete'
    except AttributeError as e:
        obs_dim = obs_space.shape[0]
        observation_type = 'Box'

    act_dim = act_space.n

    # set the seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    train_env.seed(seed)
    train_env.action_space.np_random.seed(seed)
    test_env.seed(seed)
    test_env.action_space.np_random.seed(seed)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim*state_hist_n, act_dim=act_dim, size=replay_size)

    # init a state buffer for storing last m states
    train_state_buffer = StateBuffer(m=state_hist_n)
    test_state_buffer  = StateBuffer(m=state_hist_n)

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim*state_hist_n, act_dim, obs_dim*state_hist_n, None, None)

    # alpha and entropy setup
    max_target_entropy = tf.log(tf.cast(act_dim, tf.float32))
    target_entropy_prop_ph =  tf.placeholder(dtype=tf.float32, shape=())
    target_entropy = max_target_entropy * target_entropy_prop_ph

    log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)

    if alpha == 'auto': # auto tune alpha
        alpha = tf.exp(log_alpha)
    else: # fixed alpha
        alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=alpha)

    # Main outputs from computation graph
    # with tf.variable_scope('main'):
    #     mu, pi, action_probs, log_action_probs, action_logits, q1_logits, q2_logits, q1_a, q2_a = actor_critic(x_ph, a_ph, **network_params)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, action_probs, log_action_probs, action_logits, q1_logits, q2_logits  = actor_critic(x_ph, a_ph, **network_params)

    with tf.variable_scope('main', reuse=True):
        _, _, action_probs_next, log_action_probs_next, _, _, _  =  actor_critic(x2_ph, a_ph, **network_params)

    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, q1_logits_targ, q2_logits_targ = actor_critic(x2_ph, a_ph, **network_params)

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['log_alpha',
                                                       'main/pi',
                                                       'main/q1',
                                                       'main/q2',
                                                       'main'])
    print("""\nNumber of parameters:
             alpha: %d,
             pi: %d,
             q1: %d,
             q2: %d,
             total: %d\n"""%var_counts)
             
    if use_opt:
        # Optimistic Exploration
        mu_Q    = (q1_logits + q2_logits) / 2.0
        sigma_Q = tf.math.abs(q1_logits - q2_logits) / 2.0

        Q_UB = mu_Q + beta_UB * sigma_Q
        Q_LB = mu_Q + beta_LB * sigma_Q

        Q_UB_sm = tf.nn.softmax(Q_UB, axis=-1) # needed to make EV and penalty proportional for optimisation

        R = tf.get_variable('R', dtype=tf.float32, shape=[1,act_dim], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        assign_R = R.assign(action_logits) # initialises P as the same "pessimistic" action distribution
        P = tf.nn.softmax(R, axis=-1)

        expected_value = tf.reduce_sum( tf.multiply(P, Q_UB_sm) )
        KL_P_PT = tf.reduce_sum( tf.multiply(P, tf.log( tf.divide(P, action_probs) ) ) )
        penalty = KL_P_PT - delta
        relu_penalty = tf.nn.relu(penalty)
        penalised_opt_function = - expected_value + relu_penalty

        optpi_optimizer = tf.train.AdamOptimizer(learning_rate=opt_lr)
        train_optpi_op  = optpi_optimizer.minimize(penalised_opt_function, var_list=get_vars('R'))

        optimistic_policy_dist = tf.distributions.Categorical(probs=P)
        optimistic_pi = optimistic_policy_dist.sample()
    else:
        optimistic_pi = pi # use standard SAC policy
        Q_LB = tf.minimum(q1_logits, q2_logits)

    # Min Double-Q:
    min_q_logits_targ  = tf.minimum(q1_logits_targ, q2_logits_targ)

    # Targets for Q regression
    q_backup = r_ph + gamma*(1-d_ph)*tf.stop_gradient( tf.reduce_sum(action_probs_next * (min_q_logits_targ - alpha * log_action_probs_next), axis=-1))

    # critic losses
    q1_a  = tf.reduce_sum(tf.multiply(q1_logits, a_ph), axis=1)
    q2_a  = tf.reduce_sum(tf.multiply(q2_logits, a_ph), axis=1)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_a)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_a)**2)
    value_loss = q1_loss + q2_loss

    # policy loss
    pi_backup = tf.reduce_sum(action_probs * ( alpha * log_action_probs - Q_LB ), axis=-1)
    pi_loss = tf.reduce_mean(pi_backup)

    # alpha loss for temperature parameter
    pi_entropy = -tf.reduce_sum(action_probs * log_action_probs, axis=-1)
    alpha_backup = tf.stop_gradient(target_entropy - pi_entropy)
    alpha_loss   = -tf.reduce_mean(log_alpha * alpha_backup)

    # Policy train op
    # (has to be separate from value train op, because q1_logits appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-04)
    if grad_clip_val is not None:
        gvs = pi_optimizer.compute_gradients(pi_loss,  var_list=get_vars('main/pi'))
        capped_gvs = [(ClipIfNotNone(grad, grad_clip_val), var) for grad, var in gvs]
        train_pi_op = pi_optimizer.apply_gradients(capped_gvs)
    else:
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-04)
    with tf.control_dependencies([train_pi_op]):
        if grad_clip_val is not None:
            gvs = value_optimizer.compute_gradients(value_loss, var_list=get_vars('main/q'))
            capped_gvs = [(ClipIfNotNone(grad, grad_clip_val), var) for grad, var in gvs]
            train_value_op = value_optimizer.apply_gradients(capped_gvs)
        else:
            train_value_op = value_optimizer.minimize(value_loss, var_list=get_vars('main/q'))

    alpha_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-04)
    with tf.control_dependencies([train_value_op]):
        train_alpha_op = alpha_optimizer.minimize(alpha_loss, var_list=get_vars('log_alpha'))

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, q1_a, q2_a,
                pi_entropy, target_entropy,
                alpha_loss, alpha,
                train_pi_op, train_value_op, train_alpha_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                outputs={'mu': mu, 'pi': pi, 'q1_a': q1_a, 'q2_a': q2_a})


    def get_action(state, deterministic=False):

        # # record data for printing
        # _ =  sess.run(assign_R, feed_dict={x_ph: [state]})
        # ins = sess.run([action_probs, Q_UB, P, KL_P_PT], feed_dict={x_ph: [state]})

        if deterministic:
            act_op = mu
        else:
            if use_opt:
                # run a few optimisation steps to set optimistic policy
                _ =  sess.run(assign_R, feed_dict={x_ph: [state]})
                for i in range(max_opt_steps):
                    _ = sess.run([train_optpi_op], feed_dict={x_ph: [state]})

            act_op = optimistic_pi

        # # print difference between pessimistic and optimistic policy probabilities
        # outs = sess.run([P, KL_P_PT], feed_dict={x_ph: [state]})
        #
        # print('ap:     ', ins[0])
        # print('Q:      ', ins[1])
        # print('P_in:   ', ins[2])
        # print('P_out:  ', outs[0])
        # print('KL_in:  ', ins[3])
        # print('KL_out: ', outs[1])
        # print('')

        return sess.run(act_op, feed_dict={x_ph: [state]})[0]

    def reset(env, state_buffer):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        o = process_observation(o, obs_dim, observation_type)
        r = process_reward(r)
        state = state_buffer.init_state(init_obs=o)
        return o, r, d, ep_ret, ep_len, state

    def test_agent(n=10, render=True):
        global sess, mu, pi, q1_a, q2_a
        for j in range(n):
            o, r, d, ep_ret, ep_len, test_state = reset(test_env, test_state_buffer)

            if render: test_env.render()

            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(test_state, True))
                o = process_observation(o, obs_dim, observation_type)
                r = process_reward(r)
                test_state = test_state_buffer.append_state(o)
                ep_ret += r
                ep_len += 1

                if render: test_env.render()

            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        if render: test_env.close()

    start_time = time.time()
    o, r, d, ep_ret, ep_len, state = reset(train_env, train_state_buffer)
    total_steps = steps_per_epoch * epochs

    target_entropy_prop = linear_anneal(current_step=0, start=target_entropy_start, stop=target_entropy_stop, steps=target_entropy_steps)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > start_steps:
            a = get_action(state)
        else:
            a = train_env.action_space.sample()

        # Step the env
        o2, r, d, _ = train_env.step(a)
        o2 = process_observation(o2, obs_dim, observation_type)
        a = process_action(a, act_dim)
        r = process_reward(r)
        next_state = train_state_buffer.append_state(o2)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(state, a, r, next_state, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        state = next_state

        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph:  batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph:  batch['acts'],
                             r_ph:  batch['rews'],
                             d_ph:  batch['done'],
                             target_entropy_prop_ph: target_entropy_prop
                            }

                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0],
                             LossQ1=outs[1],    LossQ2=outs[2],
                             Q1Vals=outs[3],    Q2Vals=outs[4],
                             PiEntropy=outs[5], TargEntropy=outs[6],
                             LossAlpha=outs[7], Alpha=outs[8])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len, state = reset(train_env, train_state_buffer)


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # update target entropy every epoch
            target_entropy_prop = linear_anneal(current_step=t, start=target_entropy_start, stop=target_entropy_stop, steps=target_entropy_steps)

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': train_env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(n=10,render=render)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('PiEntropy', average_only=True)
            logger.log_tabular('TargEntropy', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    plot_progress(os.path.join(logger_kwargs['output_dir'],'progress.txt'), show_plot=False)

if __name__ == '__main__':

    from spinup.utils.run_utils import setup_logger_kwargs

    network_params = {
        'hidden_sizes':[64, 64],
        'activation':'relu',
        'policy':kl_policy
    }

    rl_params = {
        # env params
        # 'env_name':'FrozenLake-v0',
        'env_name':'CartPole-v1',
        # 'env_name':'Taxi-v2',
        # 'env_name':'MountainCar-v0',
        # 'env_name':'Acrobot-v1',
        # 'env_name':'LunarLander-v2',

        # control params
        'seed': int(5),
        'epochs': int(50),
        'steps_per_epoch': 2000,
        'replay_size': 100000,
        'batch_size': 256,
        'start_steps': 0,
        'max_ep_len': 500,
        'save_freq': 5,
        'render': False,

        # rl params
        'gamma': 0.99,
        'polyak': 0.995,
        'lr': 0.0003,
        'state_hist_n': 1 ,
        'grad_clip_val':None,

        # entropy params
        'alpha': 'auto',
        'target_entropy_start':0.3, # proportion of max_entropy
        'target_entropy_stop':0.3,
        'target_entropy_steps':1e5,

        # optimistic exploration params
        'use_opt':True,
        'beta_UB': 1,
        'beta_LB': -1,
        'delta': 0.1,
        'opt_lr':0.15,
        'max_opt_steps':5,
    }


    saved_model_dir = '../../saved_models'
    logger_kwargs = setup_logger_kwargs(exp_name='oac_disc_' + rl_params['env_name'], seed=rl_params['seed'], data_dir=saved_model_dir, datestamp=False)
    env = gym.make(rl_params['env_name'])

    oac(lambda:env, actor_critic=mlp_actor_critic,
                    logger_kwargs=logger_kwargs,
                    network_params=network_params,
                    rl_params=rl_params)
