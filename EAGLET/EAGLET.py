class EAGLET:
    def __init__(self, tournament_size = 2, population_size = 94, max_generations = 50, crossoverP = 0.7,
     mutationP = 0.2, n_classifiers = 47, labels_in_classifier = 3, threshold = 0.5, beta_number = 0.75) -> None:
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossoverP = crossoverP
        self.mutationP = mutationP
        self.n_classifiers = n_classifiers
        self.labels_in_classifier = labels_in_classifier
        self.threshold = threshold
        self.beta_number = beta_number

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
        ## 1.1. do calculations
        ## 1.2. create initial individuals and fix if needed
        ## 1.3. delete repeated individuals

        # 2. 
        pass

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
        # 1. predict each classifier using fit method
        # 2. vote
        pass