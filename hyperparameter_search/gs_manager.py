from gs_env_minimum import GsEnv
class GSManager():
    def __init__(self, num_of_hyperparameters,
                 ser_url = None, usr_name = None, password = None):
        self.num_of_hyperparameters = num_of_hyperparameters
        # define parameters for GS system
        self.ser_url = ser_url
        self.usr_name = usr_name
        self.password = password
    def get_reward(self, action, step, pre_fitness):
        action = [action[0][0][x:x+self.num_of_hyperparameters] for x in range(0, len(action[0][0]), self.num_of_hyperparameters)]
        GS_model = GsEnv(self.ser_url, self.usr_name, self.password)
        new_node_gid, fitness = GS_model.get_reward(action)
        print("!!!!!!fitness:", fitness, pre_fitness)
        if fitness - pre_fitness <= 0.01:
            return fitness, fitness 
        else:
            return 0.01, fitness