import sys, os
from time import time
import EAGLET.config as config
from EAGLET.EAGLET import EAGLET
from skmultilearn.dataset import load_from_arff
from arff import BadLayout
import sklearn.metrics as metrics

def main():
    # 1. load configuration file
    try:
        file_arg = sys.argv[1]
        configs = config.load_config(file_arg)
    except IndexError:
        print("Usage: " + os.path.basename(__file__) + " <configFile path>")
        return
    
    ## store variables from configFile
    try:
        # dataset properties:
        train_path = "./Datasets/" + configs['dataset']['train_dataset'] + ".arff"
        test_path = "./Datasets/" + configs['dataset']['train_dataset'] + ".arff"
        label_count = configs['dataset']['label_count']
        sparse = True if configs['dataset']['sparse']=="True" else False
        label_location = configs['dataset']['label_location']

        # algorithm properties
        population_size = configs['algorithm']['population_size']
        max_generations = configs['algorithm']['max_generations']
        crossoverP = configs['algorithm']['crossoverP']
        mutationP = configs['algorithm']['mutationP']
        n_classifiers = configs['algorithm']['n_classifiers']
        labels_in_classifier = configs['algorithm']['labels_in_classifier']
        threshold = configs['algorithm']['threshold']
        beta_number = configs['algorithm']['beta_number']
    except KeyError as exc:
        print(repr(exc))
        return

    ## details should be printed or not
    try:
        details = True if configs['application']['output_details']=="True" else False
    except KeyError as exc:
        print(repr(exc))
        details = False
    print("*******************************************")
    print("Starting algorithm...")
    print("*******************************************")
    # 2. load_dataset    
    ## load from arff
    try:
        X_train, y_train, feature_names_initial, label_names_initial = load_from_arff(train_path, label_count=label_count, load_sparse=sparse, label_location=label_location, return_attribute_definitions=True)
        X_test, y_test, _, _ = load_from_arff(test_path, label_count=label_count, load_sparse=sparse, label_location=label_location, return_attribute_definitions=True)
    except BadLayout as exc:
        print(repr(exc) + ": probably 'sparse' attribute in config is wrong")
        return
    
    # 3. initialize Classifier
    clf = EAGLET(population_size=population_size
        , n_classifiers=n_classifiers
        , labels_in_classifier=labels_in_classifier
        , max_generations=max_generations
        , crossoverP=crossoverP
        , mutationP=mutationP
        , threshold=threshold
        , beta_number=beta_number
        , details=details)

    # 4. fit
    print()
    print("Fitting data...")
    start_time = time()
    clf.fit(X_train, y_train)
    print()
    print("Finished fitting. | execution_time: {} s".format(time()-start_time))

    # 5. predict
    print()
    print("Predicting data...")
    start_time = time()
    y_predict = clf.predict(X_test)
    print()
    print("Finished predicting. | execution_time: {} s".format(time()-start_time))
    # 6. Scoring
    print()
    print("Calculating scores:...")
    f1 = metrics.f1_score(y_test, y_predict, average="samples")
    hamming_loss = metrics.hamming_loss(y_test, y_predict)

    # 7. output results and score
    print()
    print("******************************")
    print("\tF1 Score: {}".format(f1))
    print("\tHamming Loss: {}".format(hamming_loss))
    print("******************************")
    print()
    print("Done!")

if __name__ == "__main__":
    main()
    