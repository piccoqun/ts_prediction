import numpy as np
import tensorflow as tf
import argparse

from workflow_manager import WorkflowManager
from reinforce import Reinforce
import datetime


'''
max_layer in NAS --> number of categories in the workflow (maximum is 5)
num_of_hyperparameters --> number of hyperparameters for each category
'''
def parse_args():
    desc = "TensorFlow implementation of 'Neural Architecture Search with Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--max_layers', default=1)

    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args

'''
    Policy network is a main network for searching optimal architecture
    it uses NAS - Neural Architecture Search recurrent network cell.
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1363
    Args:
        state: current state of required topology
        max_layers: maximum number of layers --> number of nodes in GS
    Returns:
        3-D tensor with new state (new topology)
'''
def policy_network(state, num_of_hyperparameters, max_layers):
    with tf.name_scope("policy_network"):
        # num_of_hyperparameters denotes the number of parameters for each layer
        nas_cell = tf.contrib.rnn.NASCell(num_of_hyperparameters*max_layers)
        outputs, state = tf.nn.dynamic_rnn(
            nas_cell,
            tf.expand_dims(state, -1),
            dtype=tf.float32
        )
        bias = tf.Variable([0.05]*num_of_hyperparameters*max_layers)
        outputs = tf.nn.bias_add(outputs, bias)
        print("outputs: ", outputs, outputs[:, -1:, :],  
              tf.slice(outputs, [0, num_of_hyperparameters*max_layers-1, 0], 
                       [1, 1, num_of_hyperparameters*max_layers]))
        #return tf.slice(outputs, [0, num_of_hyperparameters*max_layers-1, 0], 
        #[1, 1, num_of_hyperparameters*max_layers]) # Returned last output of rnn
        return outputs[:, -1:, :]

def train():
    global args
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    num_of_hyperparameters = 1
    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                           500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    reinforce = Reinforce(sess, optimizer, policy_network, args.max_layers, 
                          global_step, num_of_hyperparameters)
    workflow_manager = WorkflowManager(num_of_hyperparameters,
                 ser_url = None, usr_name = None, password = None)

    MAX_EPISODES = 2500
    step = 0
    state = np.array([[10.0]*num_of_hyperparameters*args.max_layers], dtype=np.float32)
    pre_acc = 0.0
    total_rewards = 0
    min_action = 0
    max_action = 30
    
    for i_episode in range(MAX_EPISODES):
        action = reinforce.get_action(state)
        print("ca:", action)
        if all(ai > min_action for ai in action[0][0]) and all(ai < max_action for ai in action[0][0]):
            reward, pre_acc = workflow_manager.get_reward(action, step, pre_acc)
            print("=====>", reward, pre_acc)
        else:
            reward = -1.0
        total_rewards += reward
        # In our sample action is equal state

        print ('action', action)

        state = action[0]
        reinforce.storeRollout(state, reward)

        print ('state', state)

        step += 1
        ls = reinforce.train_step(1)
        log_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)

def main():
    global args
    args = parse_args()

    train()

if __name__ == '__main__':
    main()