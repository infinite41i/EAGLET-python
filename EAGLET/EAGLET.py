from collections import Counter as freq
from collections import OrderedDict
import math
from scipy.sparse import find
from EAGLET.Ensemble.Population import Population
import EAGLET.utils as utils
class EAGLET:
    __rep_min__ = 1
    def __init__(self, labels_in_classifier = 3, tournament_size = 2, population_size = "default"
        , max_generations = 50, crossoverP = 0.7, mutationP = 0.2, n_classifiers = 47
        , threshold = 0.5, beta_number = 0.75, details=False) -> None:
        self.labels_in_classifier = labels_in_classifier
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossoverP = crossoverP
        self.mutationP = mutationP
        self.n_classifiers = n_classifiers
        self.threshold = threshold
        self.beta_number = beta_number
        self.details = details

    def fit(self, X, y):
        """Fits classifier to training data

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        self
            fitted instance of self
        """
        # 1. initialize individuals
        ## 1.1. do calculations for frequency of each label in the initial population
        if(self.details):
            print("Doing calculations for frequency of each label in the initial population...")
            print()

        #Store some values
        self.label_count = y.shape[1] #q

        label_frequencies = self.get_label_frequenciers(y) #f
        f_sum = self.get_label_frequency_sum(label_frequencies) #sigma f_l

        label_repeat_in_pop = [] #pop = population
        active_bits_count = self.labels_in_classifier*self.population_size #k*popSize
        remaining_bit_count = active_bits_count - self.label_count*self.__rep_min__ #r
        
        #distribute bit count among labels
        total_ind_count = self.population_size #individuals count upper bound
        for i in range(self.label_count):
            distribute = self.__rep_min__ + round(label_frequencies[i]/f_sum*remaining_bit_count)# a_min + ||f_l/sigmaF*r||
            label_repeat_in_pop.append(max(total_ind_count, distribute))
        
        #increment or decrement label appearances until sum(a_l)==k*popSize
        if(sum(label_repeat_in_pop)<active_bits_count):
            labels_sorted_by_f = utils.sort_labels(label_frequencies) #ascending
            i = 0
            while sum(label_repeat_in_pop) < active_bits_count:
                label_index = labels_sorted_by_f[i][0]
                label_repeat_in_pop[label_index] += 1
                i = (i+1)%self.label_count

        elif(sum(label_repeat_in_pop)>active_bits_count):
            labels_sorted_by_f = utils.sort_labels(label_frequencies, desc=True)#descending
            i = 0
            while sum(label_repeat_in_pop) > active_bits_count:
                label_index = labels_sorted_by_f[i][0]
                label_repeat_in_pop[label_index] -= 1
                i = (i+1)%self.label_count
        
        if(self.details):
            print("Number of times that each label appears in the initial population:")
            print(label_repeat_in_pop)

        self.population = Population(self.population_size, self.labels_in_classifier, self.label_count, label_repeat_in_pop
            , label_frequencies, self.details)
        #With the above commands, these works are done in sequence to generate the ensemble
        ## 1.2. create initial individuals and fix if needed
        ## 1.3. delete repeated individuals

        # 2. loop: ('max_generations' times)
        ## 2.1. calculate individual fitnesses
        ## 2.2. run a tournament to select two individuals
        ## 2.3. crossover operation
        ## 2.4. mutation operation
        ## 2.5. add new childs to population of generation g
        ## 2.6. delete repeated individuals
        ## 2.7. select n individuals and generate ensemble of generation g
        ## 2.8. n selected individuals are copied to population of generation g+1 (P_g+1)
        ## 2.9. population_size - n individuals are selected randomly from population of previous generation
        # end loop
        # 3. fit each MLC in the ensemble

    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """
        # 1. predict with each classifier using fit method
        # 2. vote
        pass

    def get_label_frequenciers(self, y):
        return freq(find(y)[1])

    def get_label_frequency_sum(self, frequencies: dict) -> int:
        return sum(frequencies.values())
