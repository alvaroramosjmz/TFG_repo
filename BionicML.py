from numpy import zeros, sqrt, sum, where
#from sklearn.neighbors import KNeighborsClassifier
import time
import sys

dic_args = {
"alg": 1,
"agents": 2,
"iterations": 3,
"processes": 4,
"desired_features": 5,
}
measure_mode = False
if(sys.argv[1] == "-m"):
    measure_mode = True
    dic_args = {
    "alg": 2,
    "agents": 3,
    "iterations": 4,
    "processes": 5,
    "desired_features": 6,
    }

ALPHA = 0.95 # parameters used in the cost function
BETA = 1 - ALPHA

#======================== common functions ===============================

#credit:
#https://towardsdatascience.com/create-your-own-k-nearest-neighbors-algorithm-in-python-eb7093fc6339

def most_common(lst):
    """
    Example: lst = [2, 1, 2, 3, 2]
    Returns the most common element in the list : 2
    """
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return sqrt(sum((point - data)**2, axis=1))

class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train    
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])        
        return list(map(most_common, neighbors)) 
    
    # I think it do not uses this function (we use cost_func instead)
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


def count_features(agent):
    """
    Returns the number of 1s of a given agent
    
    Parameters:
    agent -- The agent for which the number of features is
    counted

    Return:
    the number of features of the agent
    """

    ind = np.where(agent == 1)[0]

    return len(ind)


def cost_func(agent, alpha, beta, num_samples_train, num_samples_test, train_x, train_y, test_x, test_y, num_char, max_features):     
    """
    Cost functions of the algorithms.

    The cost is calculated as the error of the agent
    divided by the average error for a agent with
    the same number of features

    Parameters:
    agent -- The agent evaluated (binary vector[3600])
    alpha -- to adjust the importance of the error (0.95)
    beta -- to adjust the importance of the number of features (0.05)
    num_samples_train -- number of samples used for training (177)
    num_samples_test -- number of samples used for testing (177)
    train_x -- training data [177x3600]
    train_y -- training labels [177]
    test_x -- testing data [177x3600]
    test_y -- testing labels [177]
    num_char -- number of classes in the dataset (3)
    max_features -- maximum number of features available (3600)

    Returns:
    The cost of the agent
    """ 

    inputY = train_y
    testY = test_y
    
    # indices of the features selected by the agent
    ind = where(agent == 1)[0]
    if (len(ind) == 0):
        return 1
    # we make matrices with only the features selected by the agent for the training and testing data
    inputX = zeros((num_samples_train,len(ind)), dtype=float)
    testX = zeros((num_samples_test,len(ind)), dtype=float)
    aux = [i for i in range(len(ind))]
    inputX[:,aux] = train_x[:,ind]
    testX[:,aux] = test_x[:,ind]
    
    # a KNN classifier is trained with the training data
    neigh = KNeighborsClassifier(k=100)
    neigh.fit(inputX,inputY)

    # calculate the accuracy of the agent with the testing data
    num_success = 0
    prediction = neigh.predict(testX)

    for i in range(num_samples_test):

        if (prediction[i] == testY[i]):
            num_success += 1

    acc = num_success/num_samples_test


    # an adjustment is performed to the accuracy value for de kappa index
    # to take into account the samples guessed right 'by luck'
    value = acc - 1/(num_char -1)*(1 - acc)
    error = 1 - value

    return alpha * error + beta * len(ind)/max_features


if (sys.argv[dic_args["alg"]].upper() == "GA"):

    from random import random
    def fitness_func(ga_instance, solution, solution_idx):
        """
        Fitness function of the algorithm
        
        Parameters:
        ga_instance -- instance of the algorithm
        solution -- agent to be evaluated
        solution_idx -- index of the solution in the current population

        Returns:
        The fitness value. It is equal to minus the cost value. This
        is defined like that due to the maximization of the value
        that pygad looks for.
        """

        return -cost_func(solution,ga_instance.alpha, ga_instance.beta, ga_instance.num_samples_train, ga_instance.num_samples_test, ga_instance.train_x,
                            ga_instance.train_y, ga_instance.test_x, ga_instance.test_y, ga_instance.num_char, ga_instance.max_features)

    def mutation_func(offspring, ga_instance):
        """
        This function provides the mutation in the GA

        Parameters:
        offspring -- set of chromosomes of the population. In
        this case, the chromosomes are the population.
        ga_instance -- instance of GA

        Returns:
        offspring -- set of chromosomes of the population
        already mutated
        """

        for chromosome_idx in range(offspring.shape[0]):
            for gen_idx in range(offspring.shape[1]):
                if (offspring[chromosome_idx, gen_idx] == 0):
                    if (random() < MUTATION_PROBABILITY_0):
                        offspring[chromosome_idx, gen_idx] = 1
                else:
                    if (random() < MUTATION_PROBABILITY_1):
                        offspring[chromosome_idx, gen_idx] = 0

        return offspring

    def on_gen(ga_instance):
        """
        Function that provides information everytime a new population
        is evaluated
        """

        if (not measure_mode):
            if(ga_instance.generations_completed == 1):
                print(f"Iteration {1} "+
                        f"| Best value = {-ga_instance.best_solutions_fitness[0]}")

            print(f"Iteration {ga_instance.generations_completed +1} "+
                    f"| Best value = {-ga_instance.best_solution()[1]}")

    def on_start(ga_instance):
        """
        Funciton used to initialise the necessary variables to
        achieve parallelization
        """

        ga_instance.alpha = ALPHA
        ga_instance.beta = BETA
        ga_instance.num_samples_train = NUM_SAMPLES_TRAIN
        ga_instance.num_samples_test = NUM_SAMPLES_TEST
        ga_instance.train_x = TRAIN_X
        ga_instance.train_y = TRAIN_Y
        ga_instance.test_x = TEST_X
        ga_instance.test_y = TEST_Y
        ga_instance.num_char = NUM_CHAR
        ga_instance.max_features = MAX_FEATURES


if (sys.argv[dic_args["alg"]].upper() == "ACO"):

    import numpy as np

    def ant_path_maker(alpha, beta, num_samples_train, num_samples_test, train_x, train_y, test_x, test_y, num_char, max_features, pheromone, lut, desired_n_features):
        """
        Function run in parallel. It contains the part of the ACO
        algorithm where the path is created.

        It evaluates the path at each new addition of a feature. It is
        done only in between desired_n_features/2 and desired_n_features
        number of features the path in construction contains.

        Parameters:

        desired_n_features -- Number that represents the maximum number of
        features that the constructed path must have.

        Returns:

        best_path_ant -- Best path found
        best_path_score_ant -- Score of the best path found

        """
        
        visited = np.array([False]*max_features)  # list that indicates which features are selected already               
        first_feature = np.random.randint(max_features)
        visited[first_feature] = True
        unvisited = np.where(np.logical_not(visited))[0]    # list of the positions of features not selected
        next_feature = np.random.choice(unvisited)
        path = [first_feature]
        best_path_ant = [first_feature]     # best path of the current ant
        best_path_score_ant = 2             # score of the best path of the current ant

        selected_features = 1
        run = True
        while selected_features < desired_n_features and run:    # in each iteration, a new feature is added to the path
            unvisited = np.where(np.logical_not(visited))[0]
            probabilities = np.zeros(len(unvisited))

            for i, unvisited_feature in enumerate(unvisited):
                probabilities[i] = pheromone[unvisited_feature]**alpha * heuristic(unvisited_feature, lut)**beta
            sum = np.sum(probabilities)

            if(sum > 0):
                probabilities /= np.sum(probabilities)

                next_feature = np.random.choice(unvisited, p=probabilities)

                path.append(next_feature)
                selected_features += 1
                visited[next_feature] = True

                path_score = cost_func(visited, ALPHA, BETA, num_samples_train, num_samples_test, train_x, train_y, test_x, test_y, num_char, max_features)

                if(path_score < best_path_score_ant):
                    best_path_score_ant = path_score
                    best_path_ant = path.copy()
            else:
                run = False
        return best_path_ant, best_path_score_ant

    def heuristic(feature, lut_):
        """Heuristic of the algorithm"""
        return lut_[feature]

if __name__ == "__main__":

    #here are imported all the libraries necessary for the main process
    import random as rand           
    import numpy as np
    import csv
    import pygad
    import math
    from matplotlib import pyplot as plt
    import time
    from multiprocessing.pool import Pool
    from codecarbon import EmissionsTracker


    def print_syntax():
        """Shows the correct syntax to execute the programm"""

        print("Correct syntax: BionicML.py [-m] <GA/PSO/ACO/CS/WOA> <number"
            " of agents> <number of iterations> <number of processes> [<desired number of features>]")

    def manage_error(msg):
        """
        Function that is executed in the case of an error
        
        Parameters:
        msg -- message containing information of the error
        """
        print(msg)
        print_syntax()
        sys.exit(2)


    DESIRED_N_FEATURES = 100 # number of features around of which algorithms will search


    if((len(sys.argv) == 7 and measure_mode) or (len(sys.argv) == 6 and not(measure_mode))):
        try:
            DESIRED_N_FEATURES = int(sys.argv[dic_args["desired_features"]])
        except ValueError:
            manage_error("Error. Number of features must be an integer")

    try:
        num_proc = int(sys.argv[dic_args["processes"]]) # number of processes the algorithm will execute
    except ValueError:
        manage_error("Error. Number of processes must be an integer")

    except IndexError:
        manage_error("Error. Number of processes not provided")            

    try:
        num_it = int(sys.argv[dic_args["iterations"]]) # number of iterations the algorithm will perform
    except ValueError:
        manage_error("Error. Number of iterations must be an integer")

    except IndexError:
        manage_error("Error. Number of iterations not provided")

    try:
        num_ind = int(sys.argv[dic_args["agents"]]) # number of agents the algorithm will work with
    except ValueError:
        manage_error("Error. Number of agents must be an integer")

    except IndexError:
        manage_error("Error. Number of agents not provided")

    if (DESIRED_N_FEATURES <= 0 or num_it <= 0 or num_ind <= 0):
        manage_error("Error. All integers parameters must be greater than 0")

    if(sys.argv[dic_args["alg"]].upper() not in ["GA", "PSO", "ACO", "WOA", "GWO"]):
        manage_error("Error. Incorrect algorithm name")

    # learning data
    NUM_CHAR = 3 # number of different classes
    MAX_FEATURES = 3600 # total number of features

    NUM_SAMPLES_TRAIN = 178 # number of samples used for training
    NUM_SAMPLES_TEST = 178 # number of samples used for testing

    TRAIN_X = np.empty((NUM_SAMPLES_TRAIN,MAX_FEATURES))
    TRAIN_Y = np.empty(NUM_SAMPLES_TRAIN)
    TEST_X = np.empty((NUM_SAMPLES_TEST,MAX_FEATURES))
    TEST_Y = np.empty(NUM_SAMPLES_TEST)

    try:
        with open('Essex/104_training_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)

            i = 0
            for row in reader:
                for j in range(MAX_FEATURES):
                    TRAIN_X[i,j] = row[j]
                i+=1
    except ValueError:
        manage_error("Error reading training data")

    try:
        with open('Essex/104_training_class.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)

            i = 0
            for row in reader:
                TRAIN_Y[i] = row[0]
                i+=1
    except ValueError:
        manage_error("Error reading training classes")

    try:
        with open('Essex/104_testing_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)

            i = 0
            for row in reader:
                for j in range(MAX_FEATURES):
                    TEST_X[i,j] = row[j]
                i+=1
    except ValueError:
        manage_error("Error reading testing data")

    try:
        with open('Essex/104_testing_class.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)

            i = 0
            for row in reader:
                TEST_Y[i] = row[0]
                i+=1
    except ValueError:
        manage_error("Error reading testing classes")

    print("Algorithm: ", sys.argv[dic_args["alg"]])
    print("Number of agents: ", num_ind)
    print("Number of iterations: ", num_it)
    print("Number of processes: ", num_proc)

    def construct_agent(num_features):
        """
        Creates a random agent with a given number of 1s

        A filter is used in order to discard the unwanted features
        
        Parameters:
        num_features -- the number of features that the agent
        will contain

        Returns:
        agent -- the agent created
        """
        agent = np.zeros(MAX_FEATURES,int)
        remaining = np.array([i for i in range(MAX_FEATURES)])
        
        for i in range(num_features):
            new_feature = rand.choice(remaining)
            agent[new_feature] = 1
            remaining = np.where(np.logical_not(agent))[0]
        return agent
    
    def delete_features(agent):
        maximum = DESIRED_N_FEATURES
        selected_features = np.where(agent == 1)[0]
        num1 = len(selected_features)
        if (num1 <= maximum):
            return agent

        ind = np.random.choice(selected_features, size=num1-maximum, replace=False)            
        agent[ind] = 0

        return agent

    def cost_to_acc(cost, agent):
        """
        Calculates the accuracy of a agent given its cost

        Patameters:
        cost -- The cost of the agent. Calculated previously
        by cost_func

        agent -- The agent

        Returns:
        The accuracy of the agent
        """
        selected_features = count_features(agent)

        # the inverse functions used to calculate the cost are executed
        # in order to obtain the accuracy of the agent
        error = (cost - BETA*selected_features/MAX_FEATURES)/ALPHA
        value = 1 - error
        return ((NUM_CHAR -1)*value +1)/NUM_CHAR 

    def write_time(time, filename):
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([time])

    def write_accuracy(accuracy, filename):
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([accuracy])
    
    def write_solution(solution, filename):
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(solution)

    def write_output(solution, accuracy):
        with open("output.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["accuracy",accuracy])
            writer.writerow(["solution",*solution])

#//////////////////////////////////////////////////////////////////////////
#
#                              GA  
#
#//////////////////////////////////////////////////////////////////////////

    if (sys.argv[dic_args["alg"]].upper() == "GA"):

        MUTATION_PROBABILITY = 0.01 # average mutation popularity

        # adjusted mutation probability.
        # the mutation probability of 0s and 1s are adjusted in order to keep
        # the number of 1s around the number DESIRED_N_FEATURES
        MUTATION_PROBABILITY_0 = (MUTATION_PROBABILITY/2)/((MAX_FEATURES-DESIRED_N_FEATURES)/MAX_FEATURES)
        MUTATION_PROBABILITY_1 = (MUTATION_PROBABILITY/2)/(DESIRED_N_FEATURES/MAX_FEATURES)

        def fitness_to_acc(fitness, solution):
            """
            It calculates the accuracy of a given solution
            
            Parameters:
            fitness -- fitness value of the solution
            solution -- agent to be evaluated

            Returns:
            the accuracy of the solution
            """
            return cost_to_acc(-fitness,solution)

        num_parents_mating = num_ind//4
        if(num_parents_mating == 0):
            num_parents_mating = 1

        sol_per_pop = num_ind
        num_genes = MAX_FEATURES
        gene_space = [0,1]
        initial_population = [construct_agent(DESIRED_N_FEATURES) for i in range(sol_per_pop)]
        ga_instance = pygad.GA(num_generations=num_it-1,
                            num_parents_mating=num_parents_mating,
                            sol_per_pop=sol_per_pop,
                            num_genes=num_genes,
                            initial_population = initial_population,
                            fitness_func=fitness_func,
                            parent_selection_type='sss',
                            crossover_type="uniform",
                            gene_space = gene_space,
                            gene_type = int,
                            on_generation = on_gen,
                            on_start= on_start,
                            mutation_type=mutation_func,
                            parallel_processing=["process", num_proc],
                            mutation_probability=0.01,
                            keep_elitism=0
                            )
    
        


#//////////////////////////////////////////////////////////////////////////
#
#                              PSO  
#
#//////////////////////////////////////////////////////////////////////////

    if (sys.argv[dic_args["alg"]].upper() == "PSO"):

        # in this algorithm, contrary to GA, these variables
        # are used to modify the sigmoid function as well as
        # the maximum and minimum veloicities of the particles
        PROB = DESIRED_N_FEATURES/MAX_FEATURES
        MUTATION_PROBABILITY = 0.01
        MUTATION_PROBABILITY_0 = (MUTATION_PROBABILITY/2)/((MAX_FEATURES-DESIRED_N_FEATURES)/MAX_FEATURES)
        MUTATION_PROBABILITY_1 = (MUTATION_PROBABILITY/2)/(DESIRED_N_FEATURES/MAX_FEATURES)

        def S(t):
            """
            Sigmoid function.

            """
            return 1/(1 + math.exp(-t))

        #   Algorithm based on Nathan A. Rooy implementation of
        #   Simple Particle Swarm Optimization (PSO) with Python


        class Particle:
            def __init__(self,x0):
                self.position_i=np.zeros(num_dimensions)          # particle position
                self.velocity_i=np.zeros(num_dimensions)          # particle velocity
                self.pos_best_i=np.zeros(num_dimensions)         # best position individual
                self.err_best_i=-1          # best error individual
                self.err_i=-1               # error individual


                for i in range(0,num_dimensions):
                    self.velocity_i[i] = -math.log(1/PROB -1) 
                    self.position_i[i] = x0[i]

            # evaluate current fitness
            def evaluate(self,cost):
                self.err_i=cost
                # check to see if the current position is an individual best
                if self.err_i < self.err_best_i or self.err_best_i==-1:
                    self.pos_best_i=self.position_i.copy()
                    self.err_best_i=self.err_i               

            # update new particle velocity
            def update_velocity(self,pos_best_g):
                w=1       # constant inertia weight (how much to weigh the previous velocity)
                c1=1        # cognative constant
                c2=1        # social constant

                # v_min =  - math.log(1/MUTATION_PROBABILITY_0 -1) -5.187
                # v_max =  - math.log(1/(1 - MUTATION_PROBABILITY_1) -1) 2.944

                v_min = -6
                v_max = 2
                
                for i in range(0,num_dimensions):
                    r1=rand.random()
                    r2=rand.random()

                    vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
                    vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
                    self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social
                    
                    if(self.velocity_i[i] > v_max):
                        self.velocity_i[i] = v_max
                    else:
                        if(self.velocity_i[i] < v_min):
                            self.velocity_i[i] = v_min

            # update the particle position based off new velocity updates
            def update_position(self):
                for i in range(0,num_dimensions):

                    if (rand.random() < S(self.velocity_i[i])):
                        self.position_i[i] = 1
                    else:
                        self.position_i[i] = 0

                    
                        
        class PSO():
            def __init__(self,costFunc,n_features,num_particles,num_it):
                global num_dimensions

                num_dimensions=MAX_FEATURES
                err_best_g=-1                   # best error for group

                self.pos_best_g=np.zeros(num_dimensions)                   # best position for group
                self.err_best_iterations = [float("inf") for i in range(num_it)] # best global best found with each iteration

                agents_per_proc = num_particles//num_proc
                # establish the swarm
                swarm=[]
                for i in range(0,num_particles):
                    swarm.append(Particle(construct_agent(n_features)))

                # begin optimization loop
                i=0
                while i < num_it:                   

                    args = [] # arguments used passed to the cost function
                          
                    for j in range(num_particles):
                        swarm[j].position_i = delete_features(swarm[j].position_i)
                        args.append([swarm[j].position_i, ALPHA, BETA, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, 
                                    TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, NUM_CHAR, MAX_FEATURES])  

                    # cost of each particle is calculated in parallel
                    with Pool(processes=num_proc) as pool:
                        costs = pool.starmap(cost_func,args,chunksize=num_particles//num_proc + int(num_particles%num_proc != 0))

                    # cycle through particles in swarm and evaluate fitness
                    for j in range(len(swarm)):
                        swarm[j].evaluate(costs[j])
                        # print("particle ", j, " | cost ", swarm[j].err_i," | n features ", count_features(swarm[j].position_i))

                    for j in range(0,num_particles):
                        # determine if current particle is the best (globally)
                        if swarm[j].err_i < err_best_g or err_best_g == -1:
                            self.pos_best_g=swarm[j].position_i.copy()
                            err_best_g=float(swarm[j].err_i)

                    self.err_best_iterations[i] = err_best_g
                    # cycle through swarm and update velocities and position
                    for j in range(0,num_particles):
                        swarm[j].update_velocity(self.pos_best_g)
                        swarm[j].update_position()
                    
                    if (not measure_mode):
                        print(f"Iteration {i+1} | Best value = {self.err_best_iterations[i]}")
                    i+=1


#//////////////////////////////////////////////////////////////////////////
#
#                              ACO 
#
#//////////////////////////////////////////////////////////////////////////
        
    if (sys.argv[dic_args["alg"]].upper() == "ACO"):

        def path_to_solution(path):
            """
            Function that converts a path in an agent

            Paths are list that contains the positions of the selected features,
            whereas solutions are lists of 1s and 0s, that indicate which
            features are selected
            """
            solution = np.zeros(MAX_FEATURES,int)
            for p in path:
                solution[p] = 1
            return solution


        # algorithm based on https://induraj2020.medium.com/implementation-of-ant-colony-optimization-using-python-solve-traveling-salesman-problem-9c14d3114475

        def ant_colony_optimization(desired_n_features, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
            pheromone = np.ones(MAX_FEATURES)
            best_path = None        # global best path found
            best_path_score = 2     # score of the global best path
            pheromone_max = Q/(1-evaporation_rate)/2    # maximum quantity of pheromone that a feature can have
            pheromone_min = 0.01            # minimum quantity of pheromone that a feature can have
            best_score_so_far = np.zeros(n_iterations)  # list that contains the best score found with each iteration
            
            for iteration in range(n_iterations):
                best_path_iteration = None      # best path at the current iteration
                best_path_score_iteration = 2   # score of the best path at the current iteration

                args = []
                for j in range(n_ants):
                    args.append([alpha, beta, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, 
                                TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, NUM_CHAR, MAX_FEATURES, pheromone, LUT, desired_n_features]) 

                with Pool(processes=num_proc) as pool:
                    ant_path = pool.starmap(ant_path_maker,args,chunksize=n_ants//num_proc + int(n_ants%num_proc != 0))

                for i in range(n_ants):
                    
                    # print("Iteration ", iteration, "| Ant ", i, "| Best ant score ", 
                    #     ant_path[i][1], "| Features ", len(ant_path[i][0]))


                    if(ant_path[i][1] < best_path_score_iteration):     # if the constructed path is better than any of the paths
                        best_path_score_iteration = ant_path[i][1]      # of the current iteration, a new best path of the iteration
                        best_path_iteration = ant_path[i][0].copy()           # is selected


                if (best_path_score_iteration < best_path_score):          # if the constructed path is better than the best path
                    best_path = best_path_iteration.copy()                 # found until now, a new global best path is selected
                    best_path_score = best_path_score_iteration


                best_score_so_far[iteration] = best_path_score
                if (not measure_mode):
                    print(f"Iteration {iteration+1} | Best value = {best_path_score}")


                pheromone *= evaporation_rate

                # A MAX-MIN system with a hybrid approach using both the
                # best global score and the best score of the iteration
                # is implemented here 

                for i in range(len(best_path)):
                    pheromone[best_path[i]] += Q/(best_path_score*best_path_score)/2
                        
                for i in range(len(best_path_iteration)):
                    pheromone[best_path_iteration[i]] += Q/(best_path_score_iteration*best_path_score_iteration)/2


                for i in range(len(pheromone)):
                    if (pheromone[i] < pheromone_min):
                        pheromone[i] = pheromone_min
                    elif (pheromone[i] > pheromone_max):
                        pheromone[i] = pheromone_max

            return path_to_solution(best_path), best_score_so_far
        

        # Read of a look up table that will serve as
        # the heuristic of the algorithm

        LUT = np.zeros(MAX_FEATURES)
        with open('LUT.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                for n in row:
                    LUT[i] = n
                    i+=1


#//////////////////////////////////////////////////////////////////////////
#
#                              WOA
#
#//////////////////////////////////////////////////////////////////////////

    if (sys.argv[dic_args["alg"]].upper() == "WOA"):
        
        def init_position(lb, ub, N, dim):
            X = np.zeros([N, dim], dtype='float')
            for i in range(N):
                for d in range(dim):
                    X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand.random()        
            
            return X

        def index_to_solution(sol_in):
            """
            Function that converts a path in an agent

            Paths are list that contains the positions of the selected features,
            whereas solutions are lists of 1s and 0s, that indicate which
            features are selected
            """
            solution = np.zeros(MAX_FEATURES,int)
            for p in sol_in:
                solution[p] = 1
            return solution

        def binary_conversion(X, thres, N, dim):
            Xbin = np.zeros([N, dim], dtype='int')
            for i in range(N):
                for d in range(dim):
                    if X[i,d] > thres:
                        Xbin[i,d] = 1
                    else:
                        Xbin[i,d] = 0

            return Xbin


        def boundary(x, lb, ub):
            if x < lb:
                x = lb
            if x > ub:
                x = ub
            
            return x

        def delete_features_x(x, xbin, thres, dim):

            for i in range(dim):
                if(x[i] > thres and xbin[i] == 0):
                    x[i] = thres*rand.random()

            return x

        def woa(opts):
            # Parameters
            ub    = 1
            lb    = 0
            thres = 1 - DESIRED_N_FEATURES/MAX_FEATURES
            b     = 1       # constant
            
            N        = opts['N']
            max_iter = opts['T']
            if 'b' in opts:
                b    = opts['b']

            # Dimension
            dim = MAX_FEATURES
            if np.size(lb) == 1:
                ub = ub * np.ones([1, dim], dtype='float')
                lb = lb * np.ones([1, dim], dtype='float')
                
            # Initialize position 
            X    = init_position(lb, ub, N, dim)
            
            # Binary conversion
            Xbin = binary_conversion(X, thres, N, dim)
            
            # Fitness at first iteration
            fit  = np.zeros([N, 1], dtype='float')
            Xgb  = np.zeros([1, dim], dtype='float')
            fitG = float('inf')
            
            args = []
            for i in range(N):
                Xbin[i,:] = delete_features(Xbin[i,:])
                args.append([Xbin[i,:], ALPHA, BETA, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, 
                            TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, NUM_CHAR, MAX_FEATURES]) 
           
            with Pool(processes=num_proc) as pool:
                costs = pool.starmap(cost_func,args,chunksize=N//num_proc + int(N%num_proc != 0))

            for i in range(N):
                fit[i,0] = costs[i]
                if fit[i,0] < fitG:
                    Xgb[0,:] = X[i,:].copy()
                    Xgb[0,:] = delete_features_x(Xgb[0,:], Xbin[i,:], thres, dim)
                    fitG     = fit[i,0]
                                
            # Pre
            curve = np.zeros([1, max_iter], dtype='float') 
            t     = 0
            
            curve[0,t] = fitG.copy()
            if (not measure_mode):
                print(f"Iteration {t+1} | Best value = {curve[0,t]}")
            t += 1

            while t < max_iter:
                # Define a, linearly decreases from 2 to 0 
                a = 2 - t * (2 / max_iter)
                
                for i in range(N):
                    # Parameter A (2.3)
                    A = 2 * a * rand.random() - a
                    # Paramater C (2.4)
                    C = 2 * rand.random()
                    # Parameter p, random number in [0,1]
                    p = rand.random()
                    # Parameter l, random number in [-1,1]
                    l = -1 + 2 * rand.random()  
                    # Whale position update (2.6)
                    if p  < 0.5:
                        # {1} Encircling prey
                        if abs(A) < 1:
                            for d in range(dim):
                                # Compute D (2.1)
                                Dx     = abs(C * Xgb[0,d] - X[i,d])
                                # Position update (2.2)
                                X[i,d] = Xgb[0,d] - A * Dx
                                # Boundary
                                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
                        
                        # {2} Search for prey
                        elif abs(A) >= 1:
                            for d in range(dim):
                                # Select a random whale
                                k      = np.random.randint(low = 0, high = N)
                                # Compute D (2.7)
                                Dx     = abs(C * X[k,d] - X[i,d])
                                # Position update (2.8)
                                X[i,d] = X[k,d] - A * Dx
                                # Boundary
                                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
                    
                    # {3} Bubble-net attacking 
                    elif p >= 0.5:
                        for d in range(dim):
                            # Distance of whale to prey
                            dist   = abs(Xgb[0,d] - X[i,d])
                            # Position update (2.5)
                            X[i,d] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[0,d] 
                            # Boundary
                            X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
                
                # Binary conversion
                Xbin = binary_conversion(X, thres, N, dim)

                args = []
                for i in range(N):
                    Xbin[i,:] = delete_features(Xbin[i,:])
                    args.append([Xbin[i,:], ALPHA, BETA, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, 
                                TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, NUM_CHAR, MAX_FEATURES]) 
                
                with Pool(processes=num_proc) as pool:
                    costs = pool.starmap(cost_func,args,chunksize=N//num_proc + int(N%num_proc != 0))

                for i in range(N):
                    fit[i,0] = costs[i]
                    if fit[i,0] < fitG:
                        Xgb[0,:] = X[i,:].copy()
                        Xgb[0,:] = delete_features_x(Xgb[0,:], Xbin[i,:], thres, dim)
                        fitG     = fit[i,0]
                
                # Store result
                curve[0,t] = fitG.copy()

                if (not measure_mode):
                    print(f"Iteration {t+1} | Best value = {curve[0,t]}")
                t += 1            

                    
            # Best feature subset
            Gbin       = binary_conversion(Xgb, thres, 1, dim) 
            Gbin       = Gbin.reshape(dim)    
            pos        = np.asarray(range(0, dim))    
            sel_index  = pos[Gbin == 1]
            num_feat   = len(sel_index)
            # Create dictionary
            woa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
            
            return woa_data 


#//////////////////////////////////////////////////////////////////////////
#
#                              GWO
#
#//////////////////////////////////////////////////////////////////////////

    if (sys.argv[dic_args["alg"]].upper() == "GWO"):

        def index_to_solution(sol_in):
            """
            Function that converts a path in an agent

            Paths are list that contains the positions of the selected features,
            whereas solutions are lists of 1s and 0s, that indicate which
            features are selected
            """
            solution = np.zeros(MAX_FEATURES,int)
            for p in sol_in:
                solution[p] = 1
            return solution

        def init_position(lb, ub, N, dim):
            X = np.zeros([N, dim], dtype='float')
            for i in range(N):
                for d in range(dim):
                    X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand.random()        
            
            return X


        def binary_conversion(X, thres, N, dim):
            Xbin = np.zeros([N, dim], dtype='int')
            for i in range(N):
                for d in range(dim):
                    if X[i,d] > thres:
                        Xbin[i,d] = 1
                    else:
                        Xbin[i,d] = 0
            
            return Xbin


        def boundary(x, lb, ub):
            if x < lb:
                x = lb
            if x > ub:
                x = ub
            
            return x

        def delete_features_x(x, xbin, thres, dim):

            for i in range(dim):
                if(x[i] > thres and xbin[i] == 0):
                    x[i] = thres*rand.random()

            return x
            

        def gwo(opts):

            # Parameters
            ub    = 1
            lb    = 0
            thres = 1 - DESIRED_N_FEATURES/MAX_FEATURES
            
            N        = opts['N']
            max_iter = opts['T']
            
            # Dimension
            dim = MAX_FEATURES
            if np.size(lb) == 1:
                ub = ub * np.ones([1, dim], dtype='float')
                lb = lb * np.ones([1, dim], dtype='float')
                
            # Initialize position 
            X      = init_position(lb, ub, N, dim)
            
            # Binary conversion
            Xbin   = binary_conversion(X, thres, N, dim)
            
            # Fitness at first iteration
            fit    = np.zeros([N, 1], dtype='float')
            Xalpha = np.zeros([1, dim], dtype='float')
            Xbeta  = np.zeros([1, dim], dtype='float')
            Xdelta = np.zeros([1, dim], dtype='float')
            Falpha = float('inf')
            Fbeta  = float('inf')
            Fdelta = float('inf')
            
            args = []
            for i in range(N):
                Xbin[i,:] = delete_features(Xbin[i,:])
                args.append([Xbin[i,:], ALPHA, BETA, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, 
                            TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, NUM_CHAR, MAX_FEATURES])

            with Pool(processes=num_proc) as pool:
                costs = pool.starmap(cost_func,args,chunksize=N//num_proc + int(N%num_proc != 0))  


            for i in range(N):
                fit[i,0] = costs[i]
                if fit[i,0] < Falpha:
                    Xalpha[0,:] = X[i,:].copy()
                    Xalpha[0,:] = delete_features_x(Xalpha[0,:],Xbin[i,:],thres,dim)
                    Falpha      = fit[i,0]
                    
                if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                    Xbeta[0,:]  = X[i,:].copy()
                    Xbeta[0,:]  = delete_features_x(Xbeta[0,:],Xbin[i,:],thres,dim)
                    Fbeta       = fit[i,0]
                    
                if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                    Xdelta[0,:] = X[i,:].copy()
                    Xdelta[0,:] = delete_features_x(Xdelta[0,:],Xbin[i,:],thres,dim)
                    Fdelta      = fit[i,0]


            # Pre
            curve = np.zeros([1, max_iter], dtype='float') 
            t     = 0
            
            curve[0,t] = Falpha.copy()
            if (not measure_mode):
                print(f"Iteration {t+1} | Best value = {curve[0,t]}")

            t += 1
            
            while t < max_iter:  
                # Coefficient decreases linearly from 2 to 0 
                a = 2 - t * (2 / max_iter) 
                
                for i in range(N):
                    for d in range(dim):
                        # Parameter C (3.4)
                        C1     = 2 * rand.random()
                        C2     = 2 * rand.random()
                        C3     = 2 * rand.random()
                        # Compute Dalpha, Dbeta & Ddelta (3.5)
                        Dalpha = abs(C1 * Xalpha[0,d] - X[i,d]) 
                        Dbeta  = abs(C2 * Xbeta[0,d] - X[i,d])
                        Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                        # Parameter A (3.3)
                        A1     = 2 * a * rand.random() - a
                        A2     = 2 * a * rand.random() - a
                        A3     = 2 * a * rand.random() - a
                        # Compute X1, X2 & X3 (3.6) 
                        X1     = Xalpha[0,d] - A1 * Dalpha
                        X2     = Xbeta[0,d] - A2 * Dbeta
                        X3     = Xdelta[0,d] - A3 * Ddelta
                        # Update wolf (3.7)
                        X[i,d] = (X1 + X2 + X3) / 3                
                        # Boundary
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
                
                # Binary conversion
                Xbin  = binary_conversion(X, thres, N, dim)
                
                # Fitness
                args = []
                
                for i in range(N):
                    Xbin[i,:] = delete_features(Xbin[i,:])
                    args.append([Xbin[i,:], ALPHA, BETA, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, 
                                TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, NUM_CHAR, MAX_FEATURES]) 

                with Pool(processes=num_proc) as pool:
                    costs = pool.starmap(cost_func,args,chunksize=N//num_proc + int(N%num_proc != 0))  


                for i in range(N):
                    fit[i,0] = costs[i]
                    if fit[i,0] < Falpha:
                        Xalpha[0,:] = X[i,:].copy()
                        Xalpha[0,:] = delete_features_x(Xalpha[0,:],Xbin[i,:],thres,dim)
                        Falpha      = fit[i,0]
                        
                    if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                        Xbeta[0,:]  = X[i,:].copy()
                        Xbeta[0,:]  = delete_features_x(Xbeta[0,:],Xbin[i,:],thres,dim)
                        Fbeta       = fit[i,0]
                        
                    if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                        Xdelta[0,:] = X[i,:].copy()
                        Xdelta[0,:] = delete_features_x(Xdelta[0,:],Xbin[i,:],thres,dim)
                        Fdelta      = fit[i,0]

                
                curve[0,t] = Falpha.copy()

                if (not measure_mode):
                    print(f"Iteration {t+1} | Best value = {curve[0,t]}")
                t += 1
            
                        
            # Best feature subset
            Gbin       = binary_conversion(Xalpha, thres, 1, dim) 
            Gbin       = Gbin.reshape(dim)
            pos        = np.asarray(range(0, dim))    
            sel_index  = pos[Gbin == 1]
            num_feat   = len(sel_index)
            # Create dictionary
            gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
            
            return gwo_data  


   # ============================= main =======================================

    output_dir = "Measurements/" + sys.argv[dic_args["alg"]].upper()

    if(measure_mode):
        tracker = EmissionsTracker(measure_power_secs=999, output_dir = output_dir)
        tracker.start()
        t1 = time.time()

    try:
        if (sys.argv[dic_args["alg"]].upper() == "GA"):
            ga_instance.run()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            best_solutions_fitness = -np.array(ga_instance.best_solutions_fitness)
        elif (sys.argv[dic_args["alg"]].upper() == "PSO"):
            pso = PSO(cost_func,DESIRED_N_FEATURES,num_particles=num_ind,num_it=num_it)
            best_solutions_fitness = pso.err_best_iterations
            solution = pso.pos_best_g
        elif (sys.argv[dic_args["alg"]].upper() == "ACO"):
            solution, best_solutions_fitness = ant_colony_optimization(desired_n_features=DESIRED_N_FEATURES,n_ants=num_ind, n_iterations=num_it, alpha=1, beta=1, evaporation_rate=0.5, Q=2)
        elif (sys.argv[dic_args["alg"]].upper() == "WOA"):
            woa_data = woa({'N': num_ind, 'T': num_it})
            solution = index_to_solution(woa_data['sf'])
            best_solutions_fitness = woa_data['c'][0,:]
        elif (sys.argv[dic_args["alg"]].upper() == "GWO"):
            gwo_data = gwo({'N': num_ind, 'T': num_it})
            solution = index_to_solution(gwo_data['sf'])
            best_solutions_fitness = gwo_data['c'][0,:]
    finally:
        
        if(measure_mode):
            t2 = time.time()
            tracker.stop()

        # Returning the details of the best solution.
        solution_fitness = best_solutions_fitness[-1]
        accuracy = cost_to_acc(solution_fitness, solution)
        write_output(solution, accuracy)

        if(measure_mode):
            write_time(t2-t1,output_dir + "/times.csv")
            write_accuracy(accuracy,output_dir + "/accuracy.csv")
            write_solution(solution, output_dir + "/solutions.csv")
        else:
            print(f"Number of features of the best solution : {count_features(solution)}")
            print(f"Fitness value of the best solution = {solution_fitness}")
            print(f"Accuracy of the best solution = {accuracy}")

            iterations_x = np.arange(1, num_it+1)
            plt.plot(iterations_x, best_solutions_fitness)
            plt.title(sys.argv[dic_args["alg"]].upper() + " - Iteration vs. Cost")
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.show()
