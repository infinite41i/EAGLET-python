from time import time
import numpy as np
from random import randrange, sample, shuffle
from EAGLET.utils import sort_labels, sort_dict_by_value, decision
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import f1_score
from scipy.sparse.lil import lil_matrix

class Population:
    def __init__(self, pop_size: int, labels_in_individual: int, label_count: int, details = False) -> None:
        self.difinite_pop_size = pop_size
        self.pop_size = pop_size
        self.label_count = label_count
        self.labels_in_individual = labels_in_individual
        self.details = details

        #initialize individuals
        self.individuals = np.zeros((self.pop_size, self.label_count), np.byte)
        self.population_fitness_table = {}
        self.ind_dict = {}

        

    def print_inds(self):
        print()
        print("Initial Individuals:")
        print(self.individuals)

    def distribute_labels(self, labels_repeat: list, label_frequencies: dict):
        """creates initial individuals

        Parameters
        ----------
        labels_repeat : number of times that each label appears in the initial population (a_l)
        label_frequencies: frequency of each label in training data (f_l)
        """
        labels_sorted_by_f = sort_labels(label_frequencies, desc=True)#descending

        for label in labels_sorted_by_f:
            selected_inds = set()
            inappropriate_inds = set()
            
            # select a_l labels
            while len(selected_inds) < labels_repeat[label[0]] and len(inappropriate_inds)+len(selected_inds) < self.pop_size:
                rand_index = randrange(self.pop_size)
                if self.is_ind_suitable(rand_index, label[0]):
                    selected_inds.add(rand_index)
                    self.individuals[rand_index][label[0]] = 1
                else:
                    inappropriate_inds.add(rand_index)

            #fix the individuals when a label cannot be shared in a_l individuals
            while len(selected_inds) < labels_repeat[label[0]]:
                l_inactive_ind = self.get_label_inactive_ind(label[0])
                l_active_ind = self.get_label_active_ind(label[0])
                rand_suitable_bit = self.get_random_suitable_bit(l_inactive_ind, l_active_ind, label[0])
                
                self.individuals[l_inactive_ind][rand_suitable_bit] = 0
                self.individuals[l_inactive_ind][label[0]] = 1
                self.individuals[l_active_ind][rand_suitable_bit] = 1

                selected_inds.add(l_inactive_ind)
    
    def genetic_operations(self, X_train, y, crossoverP, mutationP):
        ## 2.1. calculate individual fitnesses
        if self.details:
            start_time = time()
            print("calculating individual fitnesses...", end="")
        for ind in self.individuals:
            self.get_ind_fitness(ind, X_train, y)
        if self.details:
            print("\rcalculating individual fitnesses | exec_time: {:5.3f} s".format(time()-start_time))
        
        ## 2.2. run a tournament to select two individuals
        selected_ind1, selected_ind2 = self.tournament_selection()
        
        ## 2.3. crossover operation
        if(decision(crossoverP)):
            selected_ind1 , selected_ind2 = self.crossover(selected_ind1, selected_ind2)

        ## 2.4. mutation operation
        if(decision(mutationP)):
            selected_ind1 = self.mutate(selected_ind1)
            selected_ind2 = self.mutate(selected_ind2)
        
        ## 2.5. add new childs to population of generation g
        self.add_ind(selected_ind1)
        self.add_ind(selected_ind2)

        ## 2.6. delete repeated individuals
        self.remove_duplicate_inds()

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
        for ind in self.individuals:
            indstr = self.ind_to_str(ind)
            if indstr in unique_inds:
                duplicate_inds.append(indstr)
            else:
                unique_inds.append(indstr)
        
        #replace duplicates with random indiviuals
        while len(duplicate_inds) != 0:
            duplicate_ind = self.ind_dict[duplicate_inds[0]]
            #first, fill duplicate with zeroes
            duplicate_ind.fill(0)

            #fill randomly with 'one's
            new_bits = sample(range(0,self.label_count), self.labels_in_individual)
            for label_index in new_bits:
                duplicate_ind[label_index] = 1
            
            #now check if it's duplicate again
            new_indstr = self.ind_to_str(duplicate_ind)
            if new_indstr not in unique_inds:
                duplicate_inds.pop(0)
                unique_inds.append(new_indstr)
            
    def ind_to_str(self, ind) -> str:
        s = ""
        for i in ind:
            if i == 0:
                s += "0"
            elif i == 1:
                s += "1"
        
        self.ind_dict[s] = ind
        return s

    def get_ind_by_str(self, indstr: str):
        if indstr not in self.ind_dict:
            for ind in self.individuals:
                self.ind_to_str(ind)
        return self.ind_dict[indstr]
    
    def get_ind_index(self, ind):
        ind_str = self.ind_to_str(ind)
        for i in range(len(self.individuals)):
            if self.ind_to_str(self.individuals[i]) == ind_str:
                return i
        raise IndexError
    
    def get_ind_fitness(self, ind, X_train, y_train) -> float:
        ind_str = self.ind_to_str(ind)
        if ind_str in self.population_fitness_table:
            return self.population_fitness_table[ind_str]
        else:
            fitness = self.calc_ind_fitness(ind, X_train, y_train)
            self.population_fitness_table[ind_str] = fitness
            return fitness

    def get_current_inds_fitness(self, X_train, y) -> dict:
        current_fit_dict = {}
        for ind in self.individuals:
            indstr = self.ind_to_str(ind)
            current_fit_dict[indstr] = self.get_ind_fitness(ind, X_train, y)
        return current_fit_dict

    def calc_ind_fitness(self, ind, X_train, y_train) -> float:
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

    def get_ind_y(self, ind, y_train) -> lil_matrix:
        ind_str = self.ind_to_str(ind)
        ind_y = lil_matrix(y_train.copy())
        row_count = y_train.shape[0]
        
        for i in range(len(ind_str)):
            if ind_str[i] == "0":
                for row in range(row_count):
                    ind_y[row, i] = 0
        return ind_y

    def tournament_selection(self):
        sorted_parents = sort_dict_by_value(self.population_fitness_table, desc=True)
        keys = list(sorted_parents.keys())

        parents = []
        for k in range(len(keys)):
            if len(parents) == 2:
                return self.get_ind_by_str(parents[0]), self.get_ind_by_str(parents[1])
            if keys[k] in self.ind_dict:
                parents.append(keys[k])
    
    def crossover(self, ind1, ind2):
        #find bits where two individuals differ
        ds1 = []
        ds2 = []
        for i in range(self.label_count):
            if ind1[i] == 1 and ind2[i] == 0:
                ds1.append(i)
            elif ind1[i] == 0 and ind2[i] == 1:
                ds2.append(i)
        
        #shuffle ds1 and ds2
        shuffle(ds1)
        shuffle(ds2)

        #divide by midpoint
        ds1mid = len(ds1)//2
        ds2mid = len(ds2)//2
        #crossover
        ds1_prime = ds1[:ds1mid] + ds2[ds2mid:]
        ds2_prime = ds1[ds1mid:] + ds2[:ds2mid]

        new_ind1 = np.copy(ind1)
        new_ind2 = np.copy(ind2)

        for i in range(self.label_count):
            if i in ds1:
                new_ind1[i] = 0
            elif i in ds2:
                new_ind2[i] = 0
        for i in range(self.label_count):
            if i in ds1_prime:
                new_ind1[i] = 1
            elif i in ds2_prime:
                new_ind2[i] = 1
        return new_ind1, new_ind2

    def mutate(self, ind):
        new_ind = np.copy(ind)
        rand_indices = sample(range(self.label_count), 2)
        index1 = rand_indices[0]
        index2 = rand_indices[1]
        new_ind[index1], new_ind[index2] = ind[index2], ind[index1]
        return new_ind
    
    def add_ind(self, ind):
        self.individuals = np.append(self.individuals, [ind], axis=0)
        self.pop_size += 1
    
    def remove_ind(self, ind):
        indstr = self.ind_to_str(ind)
        indnum = self.get_ind_index(ind)
        self.individuals[indnum] = self.individuals[self.pop_size-1]
        self.individuals = np.delete(self.individuals, self.pop_size-1, axis=0)
        self.pop_size -= 1
        self.ind_dict.pop(indstr)