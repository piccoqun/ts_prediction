class WorkflowManager():
    def __init__(self, num_of_hyperparameters):
        self.num_of_hyperparameters = num_of_hyperparameters

    def get_reward(self, action, step, pre_fitness):
        # action -- hyperparameters
        action = [action[0][0][x:x+self.num_of_hyperparameters] for x in range(0, len(action[0][0]), self.num_of_hyperparameters)]
        # get_reward(action)
        #fitness = get_reward(action)
        fitness = 0
        print("!!!!!!fitness:", fitness, pre_fitness)
        if fitness - pre_fitness <= 0.01:
            return fitness, fitness 
        else:
            return 0.01, fitness


    def set_parameter(self):
        pass