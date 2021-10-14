import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from time import time

if __name__ == "__main__":

    sc = SparkContext(appName="SVM")
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]

    time_1 = time()
    d_corpus = sc.textFile(train_dir, 1)  # min partition=1
    valid_lines = d_corpus.filter(lambda x: 'id="' in x and '" url=' in x and '">' in x)
    top_size = 20000
    d_keyAndText = valid_lines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    d_keyAndListOfWords = d_keyAndText.map(lambda x: (x[0], regex.sub(' ', x[1]).lower().split()))
    all_words = d_keyAndListOfWords.flatMap(lambda x: [(word, 1) for word in x[1]])
    all_counts = all_words.reduceByKey(lambda x, y: x + y)
    top_words = all_counts.top(top_size, key=lambda x: x[1])
    top_words_k = sc.parallelize(range(top_size))
    dictionary = top_words_k.map(lambda x: (top_words[x][0], x))
    dictionary.persist(storageLevel=StorageLevel(True, False, False, False))
    dictionary.take(1)
    all_words_with_docid = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    all_dictionary_words = dictionary.join(all_words_with_docid)
    just_doc_and_pos = all_dictionary_words.map(lambda x: (x[1][1], x[1][0]))
    all_dictionary_words_in_each_doc = just_doc_and_pos.groupByKey()


    def build_array(list_of_indices):
        return_val = np.zeros(top_size)
        for index in list_of_indices:
            return_val[index] = return_val[index] + 1
        my_sum = np.sum(return_val)
        return_val = np.divide(return_val, my_sum)
        return return_val


    def get_category(docid):
        if docid[0:2] == "AU":
            return 1
        else:
            return 0


    #  Train TF Array
    train_tf_array = all_dictionary_words_in_each_doc.map(lambda x: (get_category(x[0]), build_array(x[1])))
    train_tf_array.persist(storageLevel=StorageLevel(True, False, False, False))

    test_d_corpus = sc.textFile(test_dir, 1)  # min partition=1
    test_valid_lines = test_d_corpus.filter(lambda x: 'id="' in x and '" url=' in x and '">' in x)
    test_text = test_valid_lines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
    test_words = test_text.map(lambda x: (x[0], regex.sub(' ', x[1]).lower().split()))
    test_words_id = test_words.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    test_id_pos = dictionary.join(test_words_id).map(lambda x: (x[1][1], x[1][0]))
    test_tf_array = test_id_pos.groupByKey().map(lambda x: (get_category(x[0]), build_array(x[1])))
    test_tf_array.persist(storageLevel=StorageLevel(True, False, False, False))

    train_tf_array.take(1)
    test_tf_array.take(1)
    time_2 = time()
    print("===== Task 1 - Using Spark Mllib to train SVM =====")
    print("time cost for data reading and preprocessing: {:.2f} minutes".format((time_2 - time_1) / 60))

    # Load and parse the data
    def parsePoint(values):
        return LabeledPoint(values[0], values[1])

    train_data = train_tf_array.map(parsePoint)
    test_data = test_tf_array.map(parsePoint)
    model = SVMWithSGD.train(train_data, iterations=100, intercept=True)
    time_3 = time()
    print("time cost for data training: {:.2f} minutes".format((time_3 - time_2) / 60))


    def evaluate(input_result):
        true = input_result[0]
        predicted = input_result[1]
        if true == 1 and predicted == 1:
            return "TP", 1
        if true == 1 and predicted == 0:
            return "FN", 1
        if true == 0 and predicted == 1:
            return "FP", 1
        if true == 0 and predicted == 0:
            return "TN", 1


    labelsAndPreds = test_data.map(lambda p: (p.label, model.predict(p.features)))
    final_result = labelsAndPreds.map(evaluate).reduceByKey(lambda x, y: x + y).collect()

    time_4 = time()
    print("time cost for data testing: {:.2f} minutes".format((time_4 - time_3) / 60))

    TP, FN, FP, TN = 1.0, 1.0, 1.0, 1.0
    for each in final_result:
        if each[0] == "TP" and each[1] > 0:
            TP = float(each[1])
            continue
        if each[0] == "FN" and each[1] > 0:
            FN = float(each[1])
            continue
        if each[0] == "FP" and each[1] > 0:
            FP = float(each[1])
            continue
        if each[0] == "TN" and each[1] > 0:
            TN = float(each[1])
            continue

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
    print("accuracy: {}, precision: {}, recall: {}".format(accuracy, precision, recall))
    f1_score = 2.0 * precision * recall / (precision + recall)
    print("f1_score : {}".format(f1_score))

    # task 2
    print("===== Task 2 - Implementing SVM model from Scratch =====")
    iteration_num = 100
    learning_rate = 0.1
    re_parameter = 20.0
    weights = np.random.randn(top_size)
    intercept = np.random.randn(1)[0]

    def train(input_data, w, c):
        y = input_data[0]
        x = input_data[1]
        if y == 0:
            y = -1
        re = y * (np.dot(w, x) + c)
        cost = 0
        gradient = 0
        gradient_c = 0
        if re < 1:
            cost = 1 - re
            gradient = - y * x
            gradient_c = - y
        return (cost, gradient, gradient_c)


    old_cost = None
    best_cost = None
    best_weights = None
    best_intercept = None
    sample_size = 2000.0
    for i in range(iteration_num):
        sample = sc.parallelize(train_tf_array.takeSample(True, int(sample_size)))
        cost_sum, gradient_sum, gradient_c_sum = sample.map(lambda x: train(x, weights, intercept)).reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1], x[2]+y[2]))
        cost = cost_sum / sample_size + (np.dot(weights, weights) / (2 * sample_size * re_parameter))
        gradient = gradient_sum / sample_size + weights / (sample_size * re_parameter)
        gradient_c = gradient_c_sum / sample_size
        weights = weights - learning_rate * gradient
        intercept = intercept - learning_rate * gradient_c
        print("epoch: {}  cost: {}  learning_rate: {}".format(i + 1, cost, learning_rate))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_weights = weights
            best_intercept = intercept
        if old_cost is None:
            old_cost = cost
            continue
        elif cost < old_cost:
            learning_rate *= 1.05
        else:
            learning_rate *= 0.5
        old_cost = cost

    time_5 = time()

    def predict(input_val, w, c):
        true_label = input_val[0]
        features = input_val[1]
        re = np.dot(w, features) + c
        if re >= 0:
            predict_label = 1
        else:
            predict_label = 0
        return evaluate((true_label, predict_label))


    test_result = test_tf_array.map(lambda x: predict(x, best_weights, best_intercept)).reduceByKey(lambda x, y: x + y).collect()
    time_6 = time()
    TP, FN, FP, TN = 1.0, 1.0, 1.0, 1.0
    for each in test_result:
        if each[0] == "TP" and each[1] > 0:
            TP = float(each[1])
            continue
        if each[0] == "FN" and each[1] > 0:
            FN = float(each[1])
            continue
        if each[0] == "FP" and each[1] > 0:
            FP = float(each[1])
            continue
        if each[0] == "TN" and each[1] > 0:
            TN = float(each[1])
            continue

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
    print("accuracy: {}, precision: {}, recall: {}".format(accuracy, precision, recall))
    f1_score = 2.0 * precision * recall / (precision + recall)
    print("f1_score : {}".format(f1_score))

    print("time cost for data training: {:.2f} minutes".format((time_5 - time_4) / 60))
    print("time cost for data testing: {:.2f} minutes".format((time_6 - time_5) / 60))

    # task 3
    print("===== Task 3 - Weighted Loss Function =====")
    au_num = float(train_tf_array.filter(lambda x: x[0] == 1).count())
    wiki_num = float(train_tf_array.filter(lambda x: x[0] == 0).count())
    weight_au = 1
    weight_wiki = au_num / wiki_num

    print("wiki_num: {}".format(wiki_num))
    print("au_num: {}".format(au_num))
    print("weight of wiki: {}".format(weight_wiki))
    print("weight of au: {}".format(weight_au))

    iteration_num = 100
    learning_rate = 0.1
    re_parameter = 20.0
    weights = np.random.randn(top_size)
    intercept = np.random.randn(1)[0]

    def train2(input_data, w, c):
        y = input_data[0]
        x = input_data[1]
        if y == 0:
            y = -1
        re = y * (np.dot(w, x) + c)
        cost = 0
        gradient = 0
        gradient_c = 0
        if re < 1:
            if y == 1:
                cost = (1 - re) * weight_au
                gradient = (- y * x) * weight_au
                gradient_c = - y * weight_au
            else:
                cost = (1 - re) * weight_wiki
                gradient = (- y * x) * weight_wiki
                gradient_c = - y * weight_wiki
        return cost, gradient, gradient_c

    old_cost = None
    best_cost = None
    best_weights = None
    best_intercept = None
    sample_size = 2000.0
    for i in range(iteration_num):
        sample = sc.parallelize(train_tf_array.takeSample(True, int(sample_size)))
        cost_sum, gradient_sum, gradient_c_sum = sample.map(lambda x: train2(x, weights, intercept)).reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))
        cost = cost_sum / sample_size + (np.dot(weights, weights) / (2 * sample_size * re_parameter))
        gradient = gradient_sum / sample_size + weights / (sample_size * re_parameter)
        gradient_c = gradient_c_sum / sample_size
        weights = weights - learning_rate * gradient
        intercept = intercept - learning_rate * gradient_c
        print("epoch: {}  cost: {}  learning_rate: {}".format(i + 1, cost, learning_rate))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_weights = weights
            best_intercept = intercept
        if old_cost is None:
            old_cost = cost
            continue
        elif cost < old_cost:
            learning_rate *= 1.05
        else:
            learning_rate *= 0.5
        old_cost = cost


    test_result = test_tf_array.map(lambda x: predict(x, best_weights, best_intercept)).reduceByKey(
        lambda x, y: x + y).collect()
    TP, FN, FP, TN = 1.0, 1.0, 1.0, 1.0
    for each in test_result:
        if each[0] == "TP" and each[1] > 0:
            TP = float(each[1])
            continue
        if each[0] == "FN" and each[1] > 0:
            FN = float(each[1])
            continue
        if each[0] == "FP" and each[1] > 0:
            FP = float(each[1])
            continue
        if each[0] == "TN" and each[1] > 0:
            TN = float(each[1])
            continue

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
    print("accuracy: {}, precision: {}, recall: {}".format(accuracy, precision, recall))
    f1_score = 2.0 * precision * recall / (precision + recall)
    print("f1_score : {}".format(f1_score))

