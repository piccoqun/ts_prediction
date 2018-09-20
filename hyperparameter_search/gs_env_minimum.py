import random

class GsEnv:
    def __init__(self, ser_url, usr_name, password):
        self.ser_url = ser_url
        self.user_name = usr_name
        self.password = password
        self.replacements = dict()

    '''
    :parameter end_node_gid: the last node for the dataflow
    :return  a map of action space's type: {$node_gid : node type(str(for node gid), float(for a continous number), int(for discrete numbers)}
    '''
    def init_dataflow_template(self, j_node_gid):
        pass

    '''
    :parameter node_gid: init_dataflow_template's return
    :parameter arr_of_replacement, a array of data(node_gid or float or int) that will replace the target node[node_gid]
    : key: gid
    : value: [para_name,range,ini_value]
    '''
    def set_replace_spaces(self, parameter_node_gid, arr_of_replacements):
        self.replacements[parameter_node_gid] = arr_of_replacements

    '''
    :parameter action_seqs: action_seqs of some type
    :return reward.
    '''
    def get_reward(self, action_seqs):
        print ('action_seqs', action_seqs)
        new_node_gid = 'ABC23422883EFAC88ABC23422883EFAC88' # some how get it.
        return new_node_gid,random.randint(1,10)
    '''
    : key: did
    : value: para_name, value
    '''

    '''
    :parameter actions_seqs: ???
    :return: a map for replacement {target_parameter_gid:(new_node_gid or a float or a int)}
    '''
    def encode_action_seqs_to_replace_choice(self, action_seqs):
        return {}
