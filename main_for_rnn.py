import json

def main():
    configs = json.load(open('config.json'))
    # for hyperparameter search, load parameters from a class ModelParam
    if configs['hyperparameter_search']:
        from tools.model_parameters import ModelParam
        ModelParam().setparam()
        configs = json.load(open('config.json'))

    # 1. data
    from data.data_loader import read_index_from_pkl
    data = read_index_from_pkl(configs['data'])
    # (1944, 6)['OpenPrice', 'HighPrice', 'LowPrice','TurnoverVolume', 'TurnoverValue']

    from data.data_processor import DataForModel
    data_processing = DataForModel(data, configs['data'])
    train_gen = data_processing.generate_train_batch(configs['data']['sequence_length'], configs['training']['batch_size'], configs['data']['normalise'])

    # 2. prediction model
    from predictionmodel.lstm_pytorch import LSTM
    model = LSTM(data_processing.data_train.shape[1], hidden_size=3)
    model.train(train_gen, configs['training']['epochs'])

    '''
    x_test, y_test = data_processing.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
    
    
    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
        # plot_results(predictions, y_test)
    
    def plot_results_multiple(predicted_data, true_data, prediction_len):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        #Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
        plt.savefig('reports/prediction_result.png')
        plt.show()
'''

if __name__ == '__main__':
    main()
