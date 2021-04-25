import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

class Rate(object):
    """object to hold individual rate information.

    Attributes:
    name: identifier for the rate (string)
    k: rate magnitude (float)
    compartment_from: compartment of origin (string)
    compartment_to: destination compartment (string or None)
    ref: reference information (string)
    """

    def __init__(self, name, k, cfrom, cto, ref=''):
        """
        Parameters:
        name: identifier for the rate (string)
        k: rate magnitude (float)
        cfrom: compartment of origin (string)
        cto: destination compartment (string or None)
        ref: reference information (string) [optional]
        """
        self.name = name
        self.k = k
        self.compartment_from = cfrom
        self.compartment_to = cto
        self.ref = ref

class Emission(object):
    """object to hold individual input information.

    Attributes:
    name: identifier for the input (string)
    raw_E: input rate as given (float or arraylike)
    raw_t: input time axis given (None or arraylike)
    compartment_to: input destination compartment (string)
    ref: optional reference for source info (string)

    Methods:
    get_E(t): get input rate at given time
    """
    
    def __init__(self, name, E, t, cto, ref=''):
        """
        Parameters:
        name: identifier for the input (string)
        E: input rate (float or array)
        t: input times (None or array)
        cto: destination compartment name (string)
        ref: reference for emission (string) [optional]
        """
        self.name = name
        self.raw_E = E
        self.raw_t = t
        self.compartment_to = cto
        self.ref = ref

        if t is None:
            self.E_of_t = lambda x:self.raw_E
        else:
            pad_t = np.concatenate(([-4e9], self.raw_t, [4e9]))
            pad_E = np.concatenate((self.raw_E[0:1], self.raw_E,
                                    self.raw_E[-1:]))
            self.E_of_t = interp1d(pad_t, pad_E)

    def get_E(self, t=0):
        """Get input from this source at given time."""
        return self.E_of_t(t)
    
class Model(object):
    """Box model implementation.

    Attributes:
    compartments: names of compartments (list of strings)
    compartment_indices: look-up for compartment index (dict)
    N: number of compartments (int)
    rates: collection of Rate objects (dict)
    emissions: collection of Emissions objects (dict)
    matrix: transfer matrix representation of rates
    emis_to_use: which emissions to use, by name ('all' or list of strings)

    Methods:
    add_rate: add new Rate to collection
    add_emission: add new Emission to collection
    build_matrix: translate Rate collection to matrix form
    get_E: look up total emissions at a given time
    run: solve for compartment masses for given times
    """
    
    def __init__(self, compartments):
        """Initialize model based on listed compartments."""
        
        self.compartments = compartments
        self.compartment_indices = {c:i for i,c in enumerate(compartments)}
        self.N = len(compartments)
        self.rates = {}
        self.emissions = {}
        self.matrix = None
        self.emis_to_use = 'all'
        
    def add_rate(self, rate):
        """Add a rate to the model's collection.

        Parameters:
        rate: individual rate info as Rate object
        """
        
        self.rates[rate.name] = rate

    def add_emission(self, emission):
        """Add an input to the model's collection.

        Parameters:
        emission: individual input info as Emission object
        """
        
        self.emissions[emission.name] = emission

    def build_matrix(self, ):
        """Translate collection of rates to matrix form."""
        
        self.matrix = np.zeros((self.N, self.N))
        for name, rate in self.rates.items():
            i = self.compartment_indices[rate.compartment_from]
            # check if flow stays in system:
            if rate.compartment_to is not None: 
                j = self.compartment_indices[rate.compartment_to]
                self.matrix[j,i] += rate.k
            self.matrix[i,i] -= rate.k

    def choose_emis_to_use(self, emis_list):
        """Use only the emissions given by name in emis_list.

        Parameters:
        emis_list: list of (string) emis names
        """
        if emis_list != 'all':
            for emisname in emis_list:
                assert (emisname in self.emissions.keys()),\
                f"can't use {emisname} because it's not in emissions!"

        self.emis_to_use = emis_list
            
    def get_E(self, t=0):
        """Get total inputs at given time.

        Parameters:
        t: time (float) [optional]
        
        Returns: array of total emissions
        """
        
        if self.emis_to_use == 'all':
            emis_list = self.emissions.keys()
        else:
            emis_list = self.emis_to_use

        E = np.zeros(self.N)
        for emisname in emis_list:
            emission = self.emissions[emisname]
            i = self.compartment_indices[emission.compartment_to]
            E[i] += emission.get_E(t)

        return E
            
    def run(self, tstart, tend, dt=0.01, initial_conditions=None):
        """Calculate masses in all compartments through time.

        Parameters:
        tstart: start time (float)
        tend: end time (float)
        dt: time step (float) [optional]
        initial_conditions: initial mass in each box 
                         (None or array) [optional]
        
        Returns: mass, time
        mass: mass in each compartment through time 
                 (2D array of size (Ntime, Ncompartmets))
        time: time axis of solution (1D array of length Ntime)
        """
        
        nsteps = int( (tend - tstart) / dt ) + 1
        time = np.linspace(tstart, tend, nsteps)
        M = np.zeros((nsteps, self.N))
        if initial_conditions is None:
            M0 = np.zeros(self.N)
        else:
            M0 = initial_conditions
        M[0,:] = M0
        for i,t in enumerate(time[:-1]):
            dMdt = np.dot(self.matrix, M[i,:]) + self.get_E(t)
            dM = dMdt * dt
            M[i+1,:] = M[i, :] + dM

        return M, time
    
    def get_steady_state(self, emissions_vector):
        """Calculate the steady state reservoirs associated with given emissions.

        model: Model object
        emissions_vector: vector of emissions to each compartment

        returns: steady state reservoirs
        """
        emissions_vector = np.array(emissions_vector)
        return np.linalg.solve(self.matrix,-emissions_vector)

    def eigen(self):
        """Get eigenvalues, eigenvectors, timescales and residence times."""
        eigenvalues, eigenvectors, timescales, residence_times =  decompose(self.matrix)
        return eigenvalues, eigenvectors, timescales, residence_times
    
def create_model(compartments, rates):
    """Turn list of compartment names and list of rates into model.
    
    compartments: list of compartment names
    rates: list of Rate objects
    
    returns: created Model object
    """
    model = Model(compartments)
    
    for rate in rates:
        model.add_rate(rate)
    model.build_matrix() # model creates matrix
    
    return model

def add_emissions(model, emissions):
    """Add list of emissions to model.
    
    model: Model object
    emissions: list of Emission objects
    
    returns: updated Model object
    """
    
    for emission in emissions:
        model.add_emission(emission)
    
    return model

def decompose(matrix):
    """ Decompose Matrix into eigenvalues/vectors. 
        
    Returns:
        eigenvalues (array): unordered array of eigenvalues
        eigenvectors (array): normalized right eigenvectors 
                ( eigenvectors[:,i] corresponds to eigenvalues[i] ) 
        timescales (array): eigenvalues to time units 
        residence_times (array): residence time for each compartment
    """
    eigenvalues,eigenvectors = np.linalg.eig(matrix)
    timescales = -1./np.float64(np.real(eigenvalues))
    residence_times = -1/np.float64(np.diagonal(matrix))
    return eigenvalues, eigenvectors, timescales, residence_times

def perturbation_analysis(model, time_horizon, log=True,
                          compartment_index=0, numtimes=1000,
                          mintime=1e-3):
    """ Do a perturbation analysis over given time_horizon.
        
    Arguments:
        model (Model): model to do analysis on
        time_horizon (Scalar): number of years afterwhich to end
        log (Boolean): use a log time axis? (default=True)
        compartment_index (Int): compartment of original perturbation
                    (default=0)
        numtimes (Int): number of time points to calculate
                    (default=1000)
        mintime (float): number of years after perturbation for first
                time point (default=1e-3)

    Returns:
        perturbation_output (array): compartment fractions of original perturbation.
                                (numcompartments x numtimes)
        perturbation_times (array): time points (numtimes x 1)
    """

    perturbation_output = np.zeros((model.N,numtimes),dtype=complex)
    if log:
        times = np.logspace(np.log10(mintime), np.log10(time_horizon),
                            numtimes)
    else:
        times = np.linspace(mintime, time_horizon,
                            numtimes)
    
    initcond = np.zeros(model.N)
    initcond[compartment_index] = 1. # initial conditions
    eigenvalues, eigenvectors, blah, blah = model.eigen()
    inverse_eigenvectors = np.linalg.inv(eigenvectors)
    eiginitcond = np.dot(inverse_eigenvectors, initcond)
    for i,t in enumerate(times):
        in_eig_space = np.dot(  np.diag( np.exp(eigenvalues*t) ),
                                eiginitcond  )
        perturbation_output[:,i] = np.dot(eigenvectors, 
                                          in_eig_space)
    return np.real(perturbation_output), times    

def eigen_analysis_plot(eigenvalues, eigenvectors, compartment_names=None):
    """Plot eigenvalues and vectors for current model setup.
    
    :param no_losses: (Boolean)

    """

    vecs = eigenvectors
    vals = eigenvalues

    plt.figure(figsize=(len(vals)*2,len(vals)))
    inds = range(len(vals))
    for i in inds:
        vec = 0.5*vecs[:,i]
        plt.plot(i+np.real(vec),inds,'o-',color='k')
        plt.plot(i+np.imag(vec),inds,'o-',color='g')
        plt.axvline(i,color='gray',linestyle='--')
        plt.axhline(i,color='gray',linestyle='--')
    printvals = []
    for x in -1/vals:
        if x.imag==0.:
            printvals.append('%.2f'%x.real)
        else:
            printvals.append('%.2f%+.2i'%(x.real,x.imag))
    plt.xticks(inds,printvals,fontsize=15)
    if compartment_names is None:
        compartment_names = range(len(vals))
    plt.yticks(inds,compartment_names,fontsize=15)
    return

def perturbation_plot(pert, tpert, compartment_names, rev=False, colors=None):
    if rev:
        inc = 1
    else:
        inc = -1
    if colors is not None:
        plt.stackplot(tpert,pert[::inc,:],
                      labels=compartment_names[::inc],edgecolor='k',colors=colors)
    else:
        plt.stackplot(tpert,pert[::inc,:],
                      labels=compartment_names[::inc],edgecolor='k')

            
    plt.semilogx()
    plt.legend(loc='upper right')
    plt.xlabel('Time (years)',fontsize=15)
    plt.ylabel('Fraction of perturbation',fontsize=15)
