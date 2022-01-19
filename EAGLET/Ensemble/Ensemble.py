from time import time
from EAGLET.Ensemble.Population import Population
from EAGLET.utils import sort_dict_by_value
from math import floor
import numpy as np
from random import choices
from skmultilearn.problem_transform import LabelPowerset
from sklearn.tree import DecisionTreeClassifier

class Ensemble:
    classifiers = []
    def __init__(self, population: Population, classifier_count: int, label_count: int
        , labels_in_classifier: int, details = False) -> None:
        self.population = population
        self.classifier_count = classifier_count
        self.label_count = label_count
        self.labels_in_classifier = labels_in_classifier
        self.details = details
        self.e = np.zeros((self.classifier_count, self.label_count), np.byte)

    def generate_ensemble(self, max_generations: int, X, y, crossoverP, mutationP, beta_number):
        # 2. ensemble generatin
        #loop: ('max_generations' times)
        if self.details:
                print()
        for g in range(max_generations):
            if self.details:
                print("Doing Genetic operations for generation {}...".format(g))
            self.population.genetic_operations(X, y, crossoverP, mutationP)
            ## |--> 2.1. calculate individual fitnesses
            ## |--> 2.2. run a tournament to select two individuals
            ## |--> 2.3. crossover operation
            ## |--> 2.4. mutation operation
            ## |--> 2.5. add new childs to population of generation g
            ## |--> 2.6. delete repeated individuals

            ## 2.7. select n individuals and generate ensemble of generation g
            n = self.classifier_count
            n_prime = 0

            if self.details:
                print("selecting {} individuals and generating ensemble of generation {}...".format(n,g))
            #expected number of votes or appearances of each label
            expected_label_votes = floor(self.classifier_count * self.labels_in_classifier / self.label_count)
            
            #expected votes of each label
            if self.details:
                print("Calculating expected votes...")
            ev = []
            for i in range(self.label_count):
                ev.append(expected_label_votes)
            
            #argmax of fitness ind
            if self.details:
                start_time = time()
                print("Finding argmax of fitness ind...", end="")
            current_ind_fitnesses = self.population.get_current_inds_fitness(X, y)
            sorted_fitnesses = list(sort_dict_by_value(current_ind_fitnesses, desc=True).keys())
            fittest_ind_str = sorted_fitnesses[0]
            fittest_ind = self.population.get_ind_by_str(fittest_ind_str)
            if self.details:
                print("\rFinding argmax of fitness ind... | exec_time: {} s".format(time()-start_time))
            
            #add fittest individual to ensemble of current generation
            if self.details:
                start_time = time()
                print("Adding fittest individual to ensemble of current generation...", end="")
            self.e[n_prime] = fittest_ind
            n_prime += 1
            if self.details:
                print("\rAdding fittest individual to ensemble of current generation | exec_time: {} s".format(time()-start_time))

            self.population.remove_ind(fittest_ind)

            #update ev
            self.updateEV(ev, fittest_ind)

            while n_prime < n:
                if self.details:
                    start_time = time()
                    print("Getting argmax of linear combination of beta and fitness for n_prime = {}...".format(n_prime), end="")
                hd = [] #Hamming Distances
                for ind in self.population.individuals:
                    normalized_ev = self.normalize_ev(ev)
                    hd_ind = self.hamming_distance(ind, self.e, normalized_ev, n_prime, self.label_count)
                    hd.append(hd_ind)
            
                #calculate 'beta*hd + (1-beta)*fitness' for each ind and find maximum
                exp_max = 0
                best_ind_index = -1
                best_ind = 0
                i = 0
                for ind in self.population.individuals:
                    ind_fitness = self.population.get_ind_fitness(ind, X, y)
                    ind_hd = hd[i]
                    exp = beta_number*ind_hd + (1-beta_number)*ind_fitness #linear combination of fitness and distance
                    if exp > exp_max:
                        exp_max = exp
                        best_ind_index = i
                        best_ind = ind
                    i += 1
                if best_ind_index == -1:
                    raise Exception
                else:
                    #add best individual to ensemble and remove from population
                    self.e[n_prime] = ind
                    n_prime += 1
                    self.population.remove_ind(ind)
                if self.details:
                    print("\rGetting argmax of linear combination of beta and fitness for n_prime = {} | exec_time: {} s".format(n_prime, time()-start_time))
                
            ## 2.8. population_size - n individuals are selected randomly from population of previous generation to stay in population            
            current_ind_fitnesses = self.population.get_current_inds_fitness(X, y)
            
            #calculate weights for random selection
            # if self.details:
            #     start_time = time()
            #     print("Adding fittest individual to ensemble of current generation...", end="")
            # if self.details:
            #     print("\rAdding fittest individual to ensemble of current generation... | exec_time: {}".format(time()-start_time), end="")
            weights = []
            inds = self.population.individuals
            ind_count = len(inds)
            for i in range(ind_count):
                indstr = self.population.ind_to_str(inds[i])
                weights.append(current_ind_fitnesses[indstr])
            
            #select which individuals to keep...
            selects = choices(range(ind_count), weights, k=self.population.difinite_pop_size-n)
            #and remove others
            for i in range(ind_count):
                if i not in selects:
                    self.population.remove_ind(inds[i])
            
            ## 2.9. n selected individuals are copied to population of generation g+1 (P_g+1)
            for ind in self.e:
                self.population.add_ind(ind)
        #end loop

    def fit_ensemble(self, X, y):
        i = 0
        for ind in self.e:
            self.classifiers.append(LabelPowerset(classifier=DecisionTreeClassifier(), require_dense=[False, True]))
            y_ind = self.population.get_ind_y(ind, y)
            self.classifiers[i].fit(X, y_ind)
            i += 1
    
    def predict_ensemble(self, X_prediction):
        pass

    def normalize_ev(self, ev):
        ev_sum = sum(ev)
        normalized_ev = [float(i)/ev_sum for i in ev]
        return normalized_ev

    def updateEV(self, ev, ind):
        for label in range(len(ind)):
            if ev[label] > 1:
                ev[label] -= 1

    def hamming_distance(self, ind, e, w, n_prime, label_count):
        sigma = 0
        for i in range(n_prime):
            for l in range(label_count):
                if ind[l] != e[i][l]:
                    sigma += w[l]
        return sigma/n_prime