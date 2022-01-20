import sys, os
from time import time
import EAGLET.config as config
from EAGLET.EAGLET import EAGLET
from skmultilearn.dataset import load_from_arff, save_to_arff
from arff import BadLayout
import sklearn.metrics as metrics
from datetime import datetime

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
        output_file = configs['dataset']['output_file']

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
    total_start_time = time()
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
    print("Finished fitting. | total_ensemble_fit_time: {:5.3f} s".format(time()-start_time))

    # 5. predict
    print()
    print("Predicting data...")
    start_time = time()
    y_predict = clf.predict(X_test)
    #save file
    save_to_arff(X_test, y_predict, filename="{}-{}.arff".format(output_file, datetime.now().strftime('%Y-%m-%d-%H_%M_%S')))
    print()
    print("Finished predicting. | execution_time: {:5.3f} s".format(time()-start_time))
    # 6. Scoring
    print()
    print("Calculating scores:...")
    hamming_loss = metrics.hamming_loss(y_test, y_predict)
    MaP = metrics.precision_score(y_test, y_predict, average='macro')
    MaR = metrics.recall_score(y_test, y_predict, average='macro')

    # 7. output results and score
    print()
    print("******************************")
    print("\tHamming Loss: {}".format(hamming_loss))
    print("\tMaP: {}".format(MaP))
    print("\tMaR: {}".format(MaR))
    print("******************************")
    print()
    print("Done! | Total execution time: {:5.3f} s".format(time()-total_start_time))

if __name__ == "__main__":
    main()
    