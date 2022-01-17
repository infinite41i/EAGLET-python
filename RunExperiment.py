import sys, os
import EAGLET.config as config
from EAGLET.EAGLET import EAGLET
from skmultilearn.dataset import load_from_arff
from arff import BadLayout

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
        tournament_size = configs['algorithm']['tournament_size']
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
    if(details):
        print("*******************************************")
        print("Starting algorithm...")
        print("*******************************************")
    # 2. load_dataset    
    ## load from arff
    try:
        X_train_inital, y_train_initial, feature_names_initial, label_names_initial = load_from_arff(train_path, label_count=label_count, load_sparse=sparse, label_location=label_location, return_attribute_definitions=True)
        X_test_initial, y_test_initial, _, _ = load_from_arff(test_path, label_count=label_count, load_sparse=sparse, label_location=label_location, return_attribute_definitions=True)
    except BadLayout as exc:
        print(repr(exc) + ": probably 'sparse' attribute in config is wrong")
        return
    
    # 3. initialize Classifier
    clf = EAGLET(population_size=population_size, n_classifiers=n_classifiers, labels_in_classifier=labels_in_classifier
        , tournament_size=tournament_size, max_generations=max_generations, crossoverP=crossoverP
        , mutationP=mutationP, threshold=threshold, beta_number=beta_number, details=details)

    # 4. fit
    clf.fit(X_train_inital, y_train_initial)

    # 5. predict
    clf.predict(X_test_initial)

    # 6. output results and score
    print("Done!")

if __name__ == "__main__":
    main()
    