import sys, os
import EAGLET.config as config
from skmultilearn.dataset import load_from_arff

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

    # 2. load_dataset
    try:
        train_path = "./Datasets/" + configs['dataset']['train_dataset'] + ".arff"
        test_path = "./Datasets/" + configs['dataset']['train_dataset'] + ".arff"
        label_count = configs['dataset']['label_count']
        sparse = configs['dataset']['sparse']
        label_location = configs['dataset']['label_location']
    except KeyError as exc:
        print(repr(exc))
        return
    X_train, y_train, feature_names, label_names = load_from_arff(train_path, label_count=label_count, load_sparse=sparse, label_location=label_location, return_attribute_definitions=True)
    print(X_train)

    # 3. fit

    # 4. predict

    # 5. output results
    

if __name__ == "__main__":
    main()
    