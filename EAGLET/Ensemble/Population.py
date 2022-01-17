import numpy as np
from random import randrange, sample
from EAGLET.utils import sort_labels

class Population:
    def __init__(self, pop_size: int, labels_in_individual: int, label_count: int, labels_repeat: list
        , label_frequencies: dict, details=False) -> None:
        self.pop_size = pop_size
        self.label_count = label_count
        self.labels_in_individual = labels_in_individual
        self.labels_repeat = labels_repeat

        #initialize individuals
        self.individuals = np.zeros((pop_size,label_count), np.byte)
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