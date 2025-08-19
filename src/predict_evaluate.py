from sklearn.metrics import classification_report

def predict(model, x_predict):
    if model is None or x_predict is None:
        raise Exception("❌ Model or Data not found, run training first or pass Data!!")

    y_predict = model.predict(x_predict)

    return y_predict

def evaluate(y_test, y_predict):
    if y_test is None or y_predict is None:
        raise Exception("❌ Testing samples or Prediction samples is null.")

    print(classification_report(y_test, y_predict))

def predict_and_evaluate(model, x_test, y_test):
    evaluate(y_test, predict(model, x_test))
