import sys, os
import numpy as np
import tensorflow as tf
import time

# configure gpu use and supress tensorflow warnings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# parameters for setting up
dim = 3
opt_steps = 500
opt_lr = 0.15
delta_feed = np.array([0.1])
logits = np.array([3,5,2], dtype=np.float32) # logits as would be ouput from NN
pessimistic_P_feed = np_softmax(logits)      # action probabilites after softmax
Q_feed  = np.array([100.0,200.0,400.0])      # predicted Q values

# define optimisation
delta_ph = tf.placeholder(dtype=tf.float32, shape=1)
pessimistic_P_ph = tf.placeholder(dtype=tf.float32, shape=dim)
Q_ph  = tf.placeholder(dtype=tf.float32, shape=dim)
Q = tf.nn.softmax(Q_ph, axis=-1) # Q values after softmax needed to make EV and penalty proportional for optimisation

# assign a variable to be optimized to give new optimistic action probabilites
# (initialise with logits as best starting point)
R =  tf.get_variable('R', dtype=tf.float32, shape=dim, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
assign_R = R.assign(logits)
optimistic_P = tf.nn.softmax(R, axis=-1)

# calculate expected value for loss function e.g. EV(P,Q)
expected_value = tf.reduce_sum( tf.multiply(optimistic_P, Q) )

# calculate penalty function e.g. KL(P_opt, P_pess) <= delta
KL_optP_pessP = tf.reduce_sum( tf.multiply(optimistic_P, tf.log( tf.divide(optimistic_P, pessimistic_P_ph) ) ) )
penalty = KL_optP_pessP - delta_ph

# make penalty 0 when negative e.g. no penalty for KL < delta
relu_penalty = tf.nn.relu(penalty)

# define loss function
penalised_opt_function = -expected_value + relu_penalty

# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=opt_lr)
train_op  = optimizer.minimize(penalised_opt_function, var_list=get_vars('R'))

# initialise tf sess
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())
sess.run(assign_R)

# get values before optimisation
inits = sess.run([R, optimistic_P, expected_value, penalty, KL_optP_pessP],
               feed_dict={pessimistic_P_ph:pessimistic_P_feed, Q_ph:Q_feed, delta_ph:delta_feed})

# run optimisation
for i in range(opt_steps):
    _ = sess.run([train_op],
                   feed_dict={pessimistic_P_ph:pessimistic_P_feed, Q_ph:Q_feed, delta_ph:delta_feed})

# get values after optimisation
outs = sess.run([R, optimistic_P, expected_value, penalty, KL_optP_pessP],
               feed_dict={pessimistic_P_ph:pessimistic_P_feed, Q_ph:Q_feed, delta_ph:delta_feed})


print('pessimistic_P: ', pessimistic_P_feed)
print('Q:  ',     Q_feed)
print('delta:  ', delta_feed)

print('')

print('R_init: ',   inits[0])
print('P_init:  ',  inits[1])
print('EV_init:  ', inits[2])
print('pt_init:  ', inits[3])
print('kl_init:  ', inits[4])

print('')

print('R_out:  ',  outs[0])
print('P_out:  ',  outs[1])
print('EV_out:  ', outs[2])
print('pt_out:  ', outs[3])
print('kl_out:  ', outs[4])
