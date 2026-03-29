# utils/common_utils.py

def load_data(file_path):
    """ Load data from a file """
    with open(file_path, 'r') as file:
        return file.read()

def save_data(data, file_path):
    """ Save data to a file """
    with open(file_path, 'w') as file:
        file.write(data)

def calculate_accuracy(true_labels, predicted_labels):
    """ Calculate accuracy of predictions """
    return sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) / len(true_labels)

def calculate_precision(true_positives, false_positives):
    """ Calculate precision of predictions """
    if true_positives + false_positives == 0:
        return 0
    return true_positives / (true_positives + false_positives)

def calculate_recall(true_positives, false_negatives):
    """ Calculate recall of predictions """
    if true_positives + false_negatives == 0:
        return 0
    return true_positives / (true_positives + false_negatives)