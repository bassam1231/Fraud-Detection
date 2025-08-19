from src.configuration.config import config

import src.utils as utils
import src.model_training as training
import src.predict_evaluate as predict

def main():
    print('Loading & Preprocessing Data...')
    df = utils.load_data()
    x_train, x_test, y_train, y_test = utils.preprocess_data(df)

    print(f'Training {config.model.model_name}...')
    model = training.train(x_train, y_train)
    print(f'Training {config.model.model_name} done!')

    print(f'Evaluation:')
    y_pred = predict.predict(model, x_test)
    predict.evaluate(y_test, y_pred)
    print('Done!')

if __name__ == '__main__':
    main()