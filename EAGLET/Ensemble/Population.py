import numpy as np
from random import randrange, sample
from EAGLET.utils import sort_labels
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import f1_score
from scipy.sparse.lil import lil_matrix

class Population:
    def __init__(self, pop_size: int, labels_in_individual: int, label_count: int, labels_repeat: list
        , label_frequencies: dict, details=False) -> None:
        self.pop_size = pop_size
        self.label_count = label_count
        self.labels_in_individual = labels_in_individual
        self.labels_repeat = labels_repeat

        #initialize individuals
        self.individuals = np.zeros((pop_size,label_count), np.byte)
        self.population_fitness_table = {}
        self.labels_sorted_by_f = sort_labels(label_frequencies, desc=True)#descending
        self.distribute_labels(label_count ,labels_repeat)

        ## 1.3. delete repeated individuals
        self.remove_duplicate_inds()

        #print individuals
        if details:
            print()
            print("Initial Individuals:")
            print(self.individuals)

    def distribute_labels(self, label_count: int, labels_repeat: list):
        for label in range(label_count):
            selected_inds = set()
            inappropriate_inds = set()

            while len(selected_inds) < labels_repeat[label] and len(inappropriate_inds)+len(selected_inds) < self.pop_size:
                rand_index = randrange(self.pop_size)
                if self.is_ind_suitable(rand_index, label):
                    selected_inds.add(rand_index)
                    self.individuals[rand_index][label] = 1
                else:
                    inappropriate_inds.add(rand_index)

            #fix the individuals when a label cannot be shared in a_l individuals
            while len(selected_inds) < labels_repeat[label]:
                l_inactive_ind = self.get_label_inactive_ind(label)
                l_active_ind = self.get_label_active_ind(label)
                rand_suitable_bit = self.get_random_suitable_bit(l_inactive_ind, l_active_ind, label)
                
                self.individuals[l_inactive_ind][rand_suitable_bit] = 0
                self.individuals[l_inactive_ind][label] = 1
                self.individuals[l_active_ind][rand_suitable_bit] = 1

                selected_inds.add(l_inactive_ind)
    
    def is_ind_suitable(self, ind_index: int, label_index: int) -> bool:
        if(sum(self.individuals[ind_index]) >= self.labels_in_individual):
            return False
        elif(self.individuals[ind_index][label_index] == 1):
            return False
        else:
            return True
    
    def get_label_inactive_ind(self, label: int) -> int:
        rand_index = randrange(self.pop_size)
        while self.individuals[rand_index][label] == 1:
            rand_index = randrange(self.pop_size)
        return rand_index

    def get_label_active_ind(self, label: int) -> int:
        rand_index = randrange(self.pop_size)
        while self.individuals[rand_index][label] == 0:
            rand_index = randrange(self.pop_size)
        return rand_index

    def get_random_suitable_bit(self, ind1: int, ind2: int, label: int) -> int:
        rand1 = randrange(label)
        while not (self.individuals[ind1][rand1] == 1 and self.individuals[ind2][rand1] == 0):
            rand1 = randrange(label)
        return rand1
    
    def remove_duplicate_inds(self):
        unique_inds = []
        duplicate_inds = []
        #iterates on population to find duplicates
        for i in range(self.pop_size):
            indstr = self.ind_to_str(i)
            if indstr in unique_inds:
                duplicate_inds.append(i)
            else:
                unique_inds.append(indstr)
        
        #replace duplicates with random indiviuals
        while len(duplicate_inds) != 0:
            duplicate_ind_index = duplicate_inds[0]
            #first, fill duplicate with zeroes
            self.individuals[duplicate_ind_index].fill(0)

            #fill randomly with 'one's
            new_bits = sample(range(0,self.label_count), self.labels_in_individual)
            for label_index in new_bits:
                self.individuals[duplicate_ind_index][label_index] = 1
            
            #now check if it's duplicate again
            new_indstr = self.ind_to_str(duplicate_ind_index)
            if new_indstr not in unique_inds:
                duplicate_inds.pop(0)
                unique_inds.append(new_indstr)
            
    def ind_to_str(self, ind_num: int) -> str:
        s = ""
        for i in self.individuals[ind_num]:
            if i == 0:
                s += "0"
            elif i == 1:
                s += "1"
        return s

    def get_ind_fitness(self, ind, X_train, y_train):
        ind_str = self.ind_to_str(ind)
        if ind_str in self.population_fitness_table:
            return self.population_fitness_table[ind_str]
        else:
            fitness = self.calc_ind_fitness(ind, X_train, y_train)
            self.population_fitness_table[ind_str] = fitness
            return fitness

    def calc_ind_fitness(self, ind, X_train, y_train):
        # 1. build individual-specific y_train: y_train_ind
        y_train_ind = self.get_ind_y(ind, y_train)
        
        # 2. make a classifier for individual
        ind_clf = LabelPowerset(classifier=DecisionTreeClassifier())
        
        # 3. fit classifier with X_train, y_train_ind
        ind_clf.fit(X_train, y_train_ind)

        # 4. predict on full train data
        y_predict = ind_clf.predict(X_train)

        # 5. calculate F1-score with y_train_ind data and ind_predict and 'smaple' mode
        score = f1_score(y_train_ind, y_predict, average='samples', zero_division=0)
        
        return score

    def get_ind_y(self, ind: int, y_train) -> lil_matrix:
        ind_str = self.ind_to_str(ind)
        start = ind_str.find("1")
        ind_y = lil_matrix(y_train.copy())
        row_count = y_train.shape[0]
        
        for i in range(len(ind_str)):
            if ind_str[i] == "0":
                for row in range(row_count):
                    ind_y[row, i] = 0
        # print(y_train)
        # print("**************")
        # print(ind_y)
        return ind_y
