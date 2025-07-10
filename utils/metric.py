import evaluate, numpy as np
f1_metric      = evaluate.load("f1")
accuracy_metric= evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds  = np.argmax(logits, axis=-1)
    labels = labels.astype(np.int64)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1":       f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }
