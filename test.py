# from src.utils import get_device
# from src.inference import predict_text_rnn
# from src.training.training import (
#     train_rnn,
#     train_baselines,
#     train_transformer_experiment,
# )

# CSV_PATH = "data/mental_health.csv"

# # 1. Train baselines
# baseline_results = train_baselines(CSV_PATH)
# print(
#     "Baseline metrics:", baseline_results["tfidf_logreg"], baseline_results["tfidf_svm"]
# )

# # 2. Train LSTM
# rnn_result = train_rnn(CSV_PATH, model_type="lstm")
# print("LSTM test metrics:", rnn_result["test_metrics"])

# # 3. Train GRU
# gru_result = train_rnn(CSV_PATH, model_type="gru")
# print("GRU test metrics:", gru_result["test_metrics"])

# # 4. Train tiny transformer
# transformer_result = train_transformer_experiment(CSV_PATH)
# print("Transformer test metrics:", transformer_result["test_metrics"])


# # After training LSTM:
# model = rnn_result["model"]
# word2idx = rnn_result["word2idx"]
# label_encoder = rnn_result["label_encoder"]
# device = get_device()
# model.to(device)

# # text = "I feel tired, sad, and completely overwhelmed with everything."
# text = "I am dying to go to that concert next week!"
# label, confidence = predict_text_rnn(text, model, word2idx, label_encoder, device)

# print("Input:", text)
# print("Predicted status:", label)
# print("Confidence:", confidence)


lstm_learning_curve = [
    {
        "epoch": 1,
        "train_loss": 1.0382294148668987,
        "val_accuracy": 0.6750189825360668,
        "val_f1_macro": 0.5214905559496948,
    },
    {
        "epoch": 2,
        "train_loss": 0.7156191046015018,
        "val_accuracy": 0.7215894710199949,
        "val_f1_macro": 0.608312603578468,
    },
    {
        "epoch": 3,
        "train_loss": 0.5818280513393399,
        "val_accuracy": 0.7286762844849405,
        "val_f1_macro": 0.6542586444762716,
    },
    {
        "epoch": 4,
        "train_loss": 0.48663177883266845,
        "val_accuracy": 0.7512022272842318,
        "val_f1_macro": 0.6850718342725958,
    },
    {
        "epoch": 5,
        "train_loss": 0.4034108414717462,
        "val_accuracy": 0.7484181219944318,
        "val_f1_macro": 0.6908040922649061,
    },
    {
        "epoch": 6,
        "train_loss": 0.34759006957575145,
        "val_accuracy": 0.7453809162237408,
        "val_f1_macro": 0.6847647999256834,
    },
    {
        "epoch": 7,
        "train_loss": 0.27954537777790006,
        "val_accuracy": 0.7487977727157682,
        "val_f1_macro": 0.69392632538342,
    },
    {
        "epoch": 8,
        "train_loss": 0.2218394943728456,
        "val_accuracy": 0.7477853707922045,
        "val_f1_macro": 0.691167063160982,
    },
]
