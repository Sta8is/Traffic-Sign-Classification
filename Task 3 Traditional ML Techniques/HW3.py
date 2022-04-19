import cv2 as cv
import numpy as np
import json
import os


# Local feature Extraction using SIFT
def extract_local_features(path):
    img = cv.imread(path)
    _, img_desc = sift.detectAndCompute(img, None)
    return img_desc


# Extract descriptors for all images in training set
def get_train_desc(data_folders):
    print('Extracting features...')
    train_descriptors = np.zeros((0, 128))
    for folder in data_folders:
        class_folder = os.path.join(train_data_path, folder)
        print("Accessing class folder ", class_folder)
        files = os.listdir(class_folder)
        for file in files:
            path = os.path.join(class_folder, file)
            desc = extract_local_features(path)
            if desc is None:
                continue
            train_descriptors = np.concatenate((train_descriptors, desc), axis=0)
    return train_descriptors


# Create vocabulary Using K-means Clustering
def create_vocabulary(train_descriptors, vocabulary_size):
    print('Creating vocabulary using K-Means...')
    K = vocabulary_size
    term_criteria = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    trainer = cv.BOWKMeansTrainer(K, term_criteria, 1, cv.KMEANS_PP_CENTERS)
    vocab = trainer.cluster(train_descriptors.astype(np.float32))
    np.save('Vocabulary_' + str(K) + '.npy', vocab)
    print("Vocabulary_" + str(K) + " created and saved at " + str(os.getcwd()))
    return


# Compute Normalized Histogram
def create_bovw_descriptor(desc, voc):
    bow_desc = np.zeros((1, voc.shape[0]))
    for d in range(desc.shape[0]):
        diff = desc[d, :] - voc
        distances = np.sum(np.square(diff), axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1
    return bow_desc / np.sum(bow_desc)


def bovw_descriptors(vocab, data_path, data_folders):
    class_label = np.zeros((1, 1))
    labels = np.zeros((0, 1))
    # Create Index
    print('Creating index...')
    img_paths = []
    bow_descriptors = np.zeros((0, vocab.shape[0]))
    for folder in data_folders:
        class_folder = os.path.join(data_path, folder)
        files = os.listdir(class_folder)
        for file in files:
            path = os.path.join(class_folder, file)
            desc = extract_local_features(path)
            if desc is None:
                continue
            bow_desc = create_bovw_descriptor(desc, vocab)
            img_paths.append(path)
            bow_descriptors = np.concatenate((bow_descriptors, bow_desc), axis=0)
            class_label[0, 0] = int(folder)
            labels = np.concatenate((labels, class_label), axis=0)
    print("Indexing completed")
    return bow_descriptors, labels


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key


def knn_classifier(k, train_bovw_descriptors, test_bovw_descriptors, training_class_labels, testing_class_labels,
                   dictionary):
    knn_predictions = []
    for i in range(testing_class_labels.shape[0]):
        Sum = np.zeros(len(dictionary), dtype=int)
        differences = test_bovw_descriptors[i] - train_bovw_descriptors
        # distances = np.sum(np.square(differences), axi s=1)             # L2 Distance
        distances = np.sum(np.abs(differences), axis=1)           # L1 Distance
        sorted_indices = np.argsort(distances)
        for j in range(k):
            pos = dictionary[int(training_class_labels[sorted_indices[j]])]
            Sum[pos] += 1
        result = np.argmax(Sum, axis=0)
        knn_predictions.append(get_key(dictionary, result))
    return knn_predictions


def train_svm(path, training_class_labels, training_bovw_desc, dictionary, kernel_type):
    kernel_types = {"LINEAR": cv.ml.SVM_LINEAR, "SIGMOID": cv.ml.SVM_SIGMOID, "RBF": cv.ml.SVM_RBF,
                    "CHI2": cv.ml.SVM_CHI2, "INTER": cv.ml.SVM_INTER}
    os.mkdir(path)
    os.chdir(path)
    for s in range(len(dictionary)):
        class_label = get_key(dictionary, s)
        print('Training SVM_' + str(class_label))
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(kernel_types[kernel_type])
        svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
        labels = np.array([class_label in a for a in training_class_labels], np.int32)
        svm.trainAuto(training_bovw_desc.astype(np.float32), cv.ml.ROW_SAMPLE, labels)
        svm.save(str(class_label))
    print("SVMs saved at ", os.getcwd())
    return


def svm_classifier(path, testing_bovw_desc, dictionary):
    if os.getcwd() != path:
        os.chdir(path)
    SVMs = []
    svm_files = os.listdir(path)
    svm_files = sorted(list(map(int, svm_files)))
    svm_files = list(map(str, svm_files))
    for file in svm_files:
        SVMs.append(cv.ml.SVM_load(file))
    svm_predictions = []
    for i in range(testing_bovw_desc.shape[0]):
        result = []
        descriptor = np.expand_dims(testing_bovw_desc[i], axis=1)
        descriptor = np.transpose(descriptor)
        for svm in SVMs:
            res = svm.predict(descriptor.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
            result.append(res)
        pos = int(np.argmin(result))
        svm_predictions.append(get_key(dictionary, pos))
    return svm_predictions


def evaluate(predictions, class_labels, dictionary):
    class_precision = {}
    count = 0
    for key in dictionary.keys():
        class_precision[key] = [0, 0, 0]
    for index in range(len(class_labels)):
        class_precision[int(class_labels[index])][1] += 1
        if predictions[index] == class_labels[index]:
            class_precision[int(class_labels[index])][0] += 1
            count += 1
    for value in class_precision.values():
        value[2] = str(float(value[0]) / float(value[1]) * 100)[:6] + "%"
    total_precision = [count, len(class_labels), str(float(count) / float(len(class_labels)) * 100)[:8] + "%"]
    return class_precision, total_precision


def save_results(class_precision, total_precision, filename):
    with open(filename + ".json", 'w') as json_file:
        Results = {
            "Total Results":
                {"Correct Predictions: ": total_precision[0],
                 "Total Images:": total_precision[1],
                 "Accuracy: ": total_precision[2]}}
        json.dump(Results, json_file, indent=4)
        for key in class_precision.keys():
            Results = {
                "Class " + str(key) + " Results": {
                    "Correct Class " + str(key) + " Predictions: ": class_precision[key][0],
                    "Total Class Images :": class_precision[key][1],
                    "Class_Accuracy: ": class_precision[key][2]}}
            json.dump(Results, json_file, indent=4)
    return


# --------------------Main--------------------
sift = cv.xfeatures2d_SIFT.create()
# Extract Database
train_data_path = "imagedb"
test_data_path = "imagedb_test"
train_data_folders = os.listdir(train_data_path)
test_data_folders = os.listdir(test_data_path)

# train_desc = get_train_desc(train_data_folders)
# np.save("train_descriptors.npy", train_desc)
# train_desc = np.load("train_descriptors.npy")
# create_vocabulary(train_desc, 50)
vocabulary = np.load('Vocabulary_100.npy')

# train_bovw_desc, train_class_labels = bovw_descriptors(vocabulary, train_data_path, train_data_folders)
# np.save('train_bovw_Desc_'+str(vocabulary.shape[0])+'.npy', train_bovw_desc)
# np.save('train_labels_'+str(vocabulary.shape[0])+'.npy', train_class_labels)

train_bovw_desc = np.load('train_bovw_Desc_100.npy')
train_class_labels = np.load('train_labels_100.npy')

# test_bovw_desc, test_class_labels = bovw_descriptors(vocabulary, test_data_path, test_data_folders)
# np.save('test_bovw_Desc_'+str(vocabulary.shape[0])+'.npy', test_bovw_desc)
# np.save('test_labels_'+str(vocabulary.shape[0])+'.npy', test_class_labels)

test_bovw_desc = np.load('test_bovw_Desc_100.npy')
test_class_labels = np.load('test_labels_100.npy')

dic = {}
for tdf in range(len(test_data_folders)):
    dic[int(test_data_folders[tdf])] = tdf

# K Nearest Neighbors Classifier
# K_nn = 100
# KNN_predictions = knn_classifier(K_nn, train_bovw_desc, test_bovw_desc, train_class_labels, test_class_labels, dic)
# class_precision_knn, total_knn = evaluate(KNN_predictions, test_class_labels, dic)
#
# knn_precision = float(total_knn[0]) / float(test_class_labels.shape[0])
# print("For K=" + str(K_nn) + " Correct Matches K-nn = ", total_knn[0], "Total pictures", test_class_labels.shape[0])
# print(" K-nn Accuracy  {a:.5f}%".format(a=knn_precision * 100))
# save_results(class_precision_knn, total_knn, "Results_KNN_"+str(vocabulary.shape[0])+"_K="+str(K_nn))

# Train SVM
kernel = "INTER"
svm_path = os.path.join(os.getcwd(), "SVM_" + str(vocabulary.shape[0]) + "_kernel_" + kernel)
# train_svm(svm_path, train_class_labels, train_bovw_desc, dic, kernel)

# # SVM Classifier
SVM_predictions = svm_classifier(svm_path, test_bovw_desc, dic)
class_precision_svm, total_svm = evaluate(SVM_predictions, test_class_labels, dic)

svm_precision = float(total_svm[0]) / float(test_class_labels.shape[0])
print(" Correct Matches SVM = ", total_svm[0], "Total pictures", test_class_labels.shape[0])
print(" SVM Accuracy(kernel="+kernel+"):  {a:.5f}%".format(a=svm_precision * 100))
os.chdir('..')
save_results(class_precision_svm, total_svm, "Results_SVM_"+str(vocabulary.shape[0])+"_kernel_"+kernel)

