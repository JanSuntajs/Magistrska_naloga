#!/usr/bin/env python

import numpy as np
import numpy.ma as ma
import os 
import multiprocessing
import pandas as pd 
import itertools as it 
from joblib import Parallel, delayed
import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt 
from matplotlib import colors 
from scipy import stats
from scipy import special
from scipy.misc import comb
import scandir
# from nfft import nfft_adjoint     

    

 
plt.rc('text', usetex = False)
# plt.rc('font',  family = 'sans-serif')
plt.rc('text', usetex = True)
plt.rc('font',family='serif',serif=['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
fontsize=[17,20,24]

def prepare_ax(ax, plot_mbl_ergodic=False, legend=True, fontsize=fontsize,grid=True):
    
    ax.tick_params(axis='x', labelsize=fontsize[1],pad=5,direction='out')
    if legend:
        ax.legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5)
    ax.tick_params(axis='x', labelsize=fontsize[1])
    ax.tick_params(axis='y', labelsize=fontsize[1])
    if grid:
        ax.grid(which='both')
   

    if plot_mbl_ergodic:
        ax.axhline(y=mbl, ls='--', color='green')
        ax.axhline(y=ergodic, ls='--', color='red')


def save_data_file(file,savename='', data_type='', desc=''):

    data_folder='./Data/'+data_type+'/'+desc
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    np.save(data_folder+'/'+savename,file)

def load_data_file(loadname='', data_type='',desc='', load=True):

    filename='./Data/'+data_type+'/'+desc+'/'+loadname
    if load:
        loaded_file=np.load(filename)

        return loaded_file
    else:
        return filename



def prepare_plt(savename='' ,plot_type='', desc='', top=0.89, save=True, show=True):

    plt.tight_layout()
    plt.subplots_adjust(top=top)
    if save:
        graphs_folder='./Graphs/'+plot_type+'/'+desc
        if not os.path.isdir(graphs_folder):
            os.makedirs(graphs_folder)

        # plt.savefig(graphs_folder+'/' +'double_plot'+'{}{}_{}_{}.pdf'.format(nametag,file.syspar['size'], file.syspar['ne'],file.syspar['nu']))
        plt.savefig(graphs_folder+'/'+savename)
    if show:

        plt.show()

def prepare_axarr(nrows=1, ncols=2, sharex=True, sharey=True,fontsize=[18,21,25],figsize=(8,7)):

    figsize=(ncols*figsize[0], nrows*figsize[1])

    fig, axarr=plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)

    return fig, axarr, fontsize


num_cores=multiprocessing.cpu_count()
if num_cores==None:
    num_cores=1

#                                   DESCRIPTION
#
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
'''
This code implements helper routines used to extract and plot numerical data in MBL calculations. 
The folder structure for storing numerical data is as follows: 

Results/desc*/Data/Eigensystems/mod_name*/sys_par*/

With:

desc* - job description
mod_name* - joined string of module names
sys_par* -system parameters - dimensionality, system size, number of charge carriers, whether a basis exists, number of up spins

The information about the propagation type, module parameters values and initial propagation state are 
encoded in the filenames in in the sys_par* subdir. 

The user should be able to provide the system parameters of interest and the routines
should enable relatively straightforward extraction of data. 

'''


#preparation
storage_folder='/scratch/jan/MBLexact/'
data_folder=storage_folder+'Data/'

#some definitions
pars={'JOF': '_ff_0',        #offdiagonal J Heisenberg coupling 
      'W': '_dg_0',          #spin disorder field parameter
      'T': '_ih_2',          #T- kinetic hopping
      'J': '_dg_0',          #J - Heisenberg coupling (diagonal)
      'H': '_dg_1',          #hole disorder field parameter
      'HSYM': '_dg_2',       #symmetry breaking hole term
      'WSYM': '_dg_3',        #symmetry breaking spin term
      'H_STAGG':'_dg_0'}     #staggered B field

print(pars)
val_cases={'hops':['T', 'J'],           #hopping module has T and diagonal heisenberg coupling
            'flip':['JOF', 'W'],        #spin flips module has the offdiagonal Heisenberg coupling (SpSm) and spin disorder
             'hole':['H'],              #hole disorder module has the diagonal hole disorder term
             'hole_sym':['HSYM'],       #hole symmetry breaking term 
             'spin_sym':['WSYM'],       # spin symmetry breaking term 
             'h_stagg': ['H_STAGG']}    #staggered field term in the efield module 



reformat_dict={'W':2., 'J':4., 'JOF':2., 'WSYM':2., 'H_STAGG':2., 'T':1., 'HSYM':1., 'H':1.} #how to reformat parameter values

sweep_dict={'W':0, 'H':1}
sweep_types=['W','H']
sweep_names=['Spin','Hole']

def mod_str(modules):
    '''
    Creates a joined string of modules given the names of Hamiltonian's modules as an input.
    Different versions of python do not seem to give the same result!
    '''
    # print(sorted(set( [module.strip("!?") for module in modules] )))
    mod_names="".join(sorted(set( [module.strip("!?") for module in modules] ) ))
    return mod_names







desc=''
# modstr=''
modules=['sybr_s_ex','mbl_hop_ex', 'rd_field_ex', 'rd_holes_ex', 'sybr_h_ex',]
modstr=mod_str(modules)

syspar={}
modpar={}

ergodic=0.5307
mbl=0.386
#HELPER DATA/FILE EXTRACTION ROUTINES
#-------------------------------------------------------------------------------



def format_module_string(case, modpar):

    """
    A function that properly formats the modpar string.
    Case: an entry from the val_cases dict
    """
    names=[]
    mod_template='{:+.5f}d0{}'

    for param in val_cases[case]:
        i=0
        try:
            iter_type=pars[param]
            param_value=modpar[param]
            param_value*= 1./float(reformat_dict[param])

            #offdiagonal term written first, then the diagonal
            if '_dg_' in iter_type:
                i=len(names)
            
            names.insert(i,mod_template.format(param_value ,iter_type))

        except KeyError:
            print('format_module_string info: Key {} in modpar not present.'.format(param))
        #rescale param_value



    return '_'.join(names)


def rescale_modpar_params(modpar):

    modpar_=modpar.copy()
    for param in modpar: 
        try:
            modpar_[param]*=1./float(reformat_dict[param])
        except KeyError:
            print('rescale_modpar_params info: Key {} not present in reformat_dict.'.format(param))

    return modpar_



def get_results_folder(desc):
    """
    Returns the path to the results folder
    An example of a function return:
    '/scratch/jan/MBLexact/Results/full_diag_doping_time_test/Data/'
    full_diag_dopint_time_test => desc
    """
    return storage_folder+'Results/'+desc+'/Data/'

def get_eigsys_folder(desc, modules, sys_par):
    """
    Returns the path to the folder with eigenvalues data.eigenvalues
    An example: 

    '/scratch/jan/MBLexact/Results/full_diag_doping_time_test/Data/Eigensystems/rd_field_exmbl_hop_ex/D1SS0011_Ne005F_NU003'
    
    full_diag_doping_time_test => desc
    rd_field_exmbl_hop_ex => modules
    D1SS0011_Ne005F_NU003 => sys_par

    """
    parent_folder=get_results_folder(desc) 

    return parent_folder+'Eigensystems/'+modules + '/' +sys_par

def sys_str(system):

    '''
    A helper routine that prints the sys_par name as formatted in the file output. 
    '''

    if system['base']==False:
        system['base']='F'
    elif system['base']==True:
        system['base']='T'

    if system['size']<system['ne']:
        raise ValueError('Number of charge carriers cannot be greater than the system size.')
    elif system['size']<system['nu']:
        raise ValueError('Number of up-spins cannot be greater than the system size.')
    elif system['size']<(system['ne']+system['nu']):
        raise ValueError('The number of up-spins and charge carriers must not be greater than the system size.')
    return 'D{}SS{:0>4}_Ne{:0>3}{}_NU{:0>3}'.format(system['dim'], system['size'], system['ne'], system['base'], system['nu'])

def get_syspar(syspar_file):

    syspar_file=syspar_file.strip('.dat').strip('.txt')

    syspar={}
    low=syspar_file.find('D')+1
    up=syspar_file.find('SS')
    syspar['dim']=int(syspar_file[low:up])

    low=up+2
    up=syspar_file.find('_Ne')
    syspar['size']=int(syspar_file[low:up])

    low=up+3
    base='F'
    if 'T' in syspar_file:
        base='T'
    syspar['base']=base
    up=syspar_file.find(base)

    syspar['ne']=int(syspar_file[low:up])
    syspar['nu']=int(syspar_file[-3:])

    return syspar
    

def get_values_folder(desc, mod_str, system):
# def get_values_folder(desc, mod_str, system):
    """"
    Gets the path of the folder with the actual simulation values. 

    INPUT: 

    desc - job description
    mod_str - modules in the job, already in the form returned by the mod_str result 
    ss - system size 
    ne - number of holes/electrons
    nu - number of up spins

    
    dim(optional) - system dimension, defaults to 1
    base(optional) - whether the system has basis or not, optional

    """
    
    #system settings/syspar 
    # system={'base':base, 'size': ss, 'ne': ne, 'NU':nu, 'dim':dim}
    # system=kwargs


    syspar_str=sys_str(system) #get the system parameters string

    values_folder_path=get_eigsys_folder(desc, mod_str, syspar_str)
    # print(os.path.isdir(values_folder_path))


    return values_folder_path

#------------------------------------------------------------------------------

def get_number(str):
    
    return np.float(str)

def get_modpar_values(file):

    """"
    Extract module parameter values from a filename

    """
    vals={}
    file, misc_vals=file.split('_tof_')
    names=file.split('_Mod_')
    quantity, mods=names[0], names[1:]

    # pars={'JOF': '_ff_','W': '_dg_','T': '_ih_','J': '_dg_','H': '_dg_'}

    # par_cases={'hops':[pars['T'], pars['J']], 'flip':[pars['JOF'], pars['W']], 'hole':[pars['H']]}
    # val_cases={'hops':['T', 'J'], 'flip':['JOF', 'W'], 'hole':['H']}


    #modify this in the future; it can probably be rewritten in a more elegant way 
    for mod in mods: 
        if (('_ff_' in mod) and ('_dg_' in mod)):
            case='flip' #spin flips module

        elif (('_ih_' in mod) and ('_dg_' in mod)):
            case='hops' #hopping module

        elif (('_dg_' in mod) and not (('_ih_' or '_ff_') in mod)):
            if ('_dg_0' in mod):
                case='h_stagg'
            if ('_dg_1' in mod):
                case='hole' #random hole disorder module 
            elif ('_dg_2' in mod):
                case='hole_sym'
            elif ('_dg_3' in mod):
                case='spin_sym'


        #finds the number in the modpar part of the filename string
        ind0=0
        for val_case in val_cases[case]:
            ind=str.index(mod,pars[val_case])
            numstr=mod[ind0: ind]
            vals[val_case]=np.float(numstr.replace('d','e'))
    
            ind0=ind+5

    #formatting where needed so that the proper parameter values are obtained 
    # reformat_dict={'W':2, 'J':4, 'JOF':2, 'WSYM':2, 'H_STAGG':2}



    #reformats (rescales) the extracted parameter value
    for key in reformat_dict:
        try:
            vals[key]=float(reformat_dict[key])*vals[key]
        except KeyError:
            print('get_modpar_values info: Key {} in vals not yet initialized'.format(key))
        
    return vals



def check_folder(folder, cases, all_cases=True):
    """
    Checks if a given folder is to be selected from the list of subfolders in a given directory.

    INPUT: 

    folder - a string, foldername
    cases - a dictionary of module cases
    """
    def check_true(modstring):

        """
        We need to treat the quantity substring separately.


        """
        check_quantity=( (modstring.strip() in folder)) and (folder.startswith(modstring.strip()))
        check_modstr= ('Mod_'+modstring.strip()+'_Mod' in folder) or ('Mod_'+modstring.strip()+'_tof' in folder) 
        # print(check_modstr)
        return check_modstr or check_quantity

    cases_fun={True: all, False: any}

    include=cases_fun[all_cases](check_true(case) for case in cases.values())
    
    
    return include


def check_file(file, cases, all_cases=True):
    """
    Checks if a given file is to be selected from the list of files in a given directory.

    INPUT: 

    file - a string, filename
    cases - a dictionary of module cases
    """

    # print('Check_file info. Filename:')
    # print(file)
    # print('Check_file info. Cases list: \n')
    # print([case for case in cases.values()])


    def check_true(modstring):

        """
        We need to treat the quantity substring separately.


        """
        check_quantity=( (modstring.strip() in file)) and (file.startswith(modstring.strip()))
        check_modstr= ('Mod_'+modstring.strip()+'_Mod' in file) or ('Mod_'+modstring.strip()+'_tof' in file) 
        return check_quantity or check_modstr



    cases_fun={True: all, False: any}

    include=cases_fun[all_cases](check_true(case) for case in cases.values())

    return include





# def get_results_file(quantity, T=-1., J=1., JOF=1.,W=0., H=0., values_folder=storage_folder):
def get_results_file(quantity, modpar, values_folder=storage_folder, all_cases=True, new_ordering=False):
    """
    Gets the results file given the module parameters - produces 
    a file list of all files that correspond to a given 
    file string. 

    all_cases -> used for backwards compatibility; cases refer to different hamiltonian modules used 
    
    new_ordering - defaults to false; old ordering lists all files for all module parameters
    in the same directory. This can be rather slow for large datasets; new ordering lists data into 
    separate folders for different module parameters values
    """

    # J=modpar['J']
    # JOF=modpar['JOF']
    # W=modpar['W']
    # T=modpar['T']
    # H=modpar['H']
    # HSYM=modpar['HSYM']
    # WSYM=modpar['WSYM']


    # #format appropriately 
    # J*=0.25
    # JOF*=0.5
    # W*=0.5
    # WSYM*=0.5

    #get modpar strings
    #flip module
    cases={}

    cases['quantity']=quantity
    for key in val_cases:
        cases[key]=format_module_string(key, modpar)

    # cases['flip']='{:+.5f}d0_ff_0_{:+.5f}d0_dg_0'.format(JOF, W)
    # cases['hops']='{:+.5f}d0_ih_2_{:+.5f}d0_dg_0'.format(T, J)
    # if all_cases: #so that also previous runs when only two modules were used are compatible
    #     cases['hole']='{:+.5f}d0_dg_1'.format(H)
    #     cases['hole_sym']='{:+.5f}d0_dg_2'.format(HSYM)
    #     cases['spin_sym']='{:+.5f}d0_dg_3'.format(WSYM)
    print('get_results_file info. cases values: {}'.format(cases))

    cwd=os.getcwd()

    os.chdir(values_folder)

    print(cases.values())
    # filelist=[file for file in os.listdir(os.getcwd()) if check_file(file, cases,all_cases)] 
    iter_generator=scandir.scandir(os.getcwd())
    files=[entry.name for entry in iter_generator]
    # print(files)

    if not new_ordering:  
        
        #the old style, when all files were dropped in the same folder
        print('get_results_file info. number of all files in the folder:', len(files))
        # print('subfolders:')
        # print(files)
        filelist=[file for file in files if check_file(file, cases, all_cases)]
        print('get_results_file info. numberr of corresponding files in the folder:', len(filelist))
        print(os.getcwd())
        os.chdir(cwd)
        return filelist, cases
        

    else:      
        #the new style with subfolders ->subfolder is the modpar name 
        #quantity not needed in this case 
        del cases['quantity']
        print(cases)
        # print('new ordering valid. Listing module parameters folders in the directory.')
        # print('get_results_file info. number of all subfolders in the directory:', len(files))
        #finds the appropriate folder
        # print(files)
        # folder=[file for file in files if check_folder(file, cases, all_cases)]
        folder=next(file for file in files if check_file(file, cases, all_cases))
        print(folder)


        print(os.getcwd())
        os.chdir(cwd)
        #which quantity in the subfolder are we looking for
        return folder, cases

    

#-------------------------------------------------------------------------------

#load files -> used in dataSet class

# def load_data(desc, modules, size=1, ne=0, nu=0,dim=1, base=false, T=-1., J=1., JOF=1., W=0., H=0.):
def load_data(quantity, desc, modules, syspar, modpar, all_cases=True, importrange=1,print_filenames=False, filetype='bin', num_cores=num_cores, new_ordering=False):
    #implement with system, modpar
    #filetype: 'bin' or 'txt' for binary and txt files
    # system={'size': size, 'ne':ne, 'nu':nu, 'dim':dim, 'base':base}

    #initialize to default values
    data=pd.DataFrame([])
    filestring=''

    #find the folder where values are located - this is up to the syspar level; 
    #everything is the same w.r.t. the new_ordering
    folder=get_values_folder(desc, modules, syspar)
    print('Folder is:', folder)

    subfolder=''
    if new_ordering:
        subfolder, cases=get_results_file(quantity, modpar, folder, all_cases, new_ordering)

    folder+='/'+subfolder
    print('Folder is:', folder)
    #how many results files exist:
    #here the difference between orderings appears.
    filelist, cases=get_results_file(quantity, modpar, folder, all_cases, new_ordering=False)
    filelist=sorted(filelist)

    if print_filenames:
        print('Load_data_info: filenames \n')
        print(filelist)

    nfiles=len(filelist)
    print('Load_data info. Number of the corresponding files in \n {}: \n {}'.format(folder,nfiles))

    cwd=os.getcwd()
    os.chdir(folder)

    if not filelist:
        print('Load_data info: No files with {} name present!'.format(cases))
        # os.chdir(cwd)
        # return data, filestring


    else:

        #np.fromfile is used since fortran saves files in binary format
        #for now, quantities that are not entanglement entropy are only one column of data
        inputs=filelist[:int(importrange*nfiles)]
        #the string describing the file following the quantity
        filestring=filelist[0][:-11].replace(quantity,'')

        if filetype=='bin':
            sep=''
        elif filetype=='txt':
            sep=' ' 


        if quantity!='Ent_entropy':
            # data=np.array([np.fromfile(file).T for file in inputs])

            data=Parallel(n_jobs=num_cores)(delayed(np.fromfile)(file,sep=sep) for file in inputs)
            data=pd.DataFrame(np.array(data))
            # data=[pd.DataFrame(np.fromfile(file)) for file in filelist[:int(importrange*nfiles)]]
        else:
            #entanglement entropy data are stored in format of energy first and then corresponding
            #subsize entropies follow; shape is inferred from the filename parameters

            #infer shape
            ncols=int(np.ceil(syspar['size']/2.))+1
            #ncols, -1 takes care of automatic reshape - the routine finds out what the remaining dimension should be
            # data=pd.concat([pd.DataFrame(np.fromfile(file).reshape((ncols,-1))) for file in inputs ])

            data=Parallel(n_jobs=num_cores)(delayed(pd.DataFrame)(np.fromfile(file,sep=sep).reshape((-1,ncols))) for file in inputs)
            # print('datazero',data[0])

            # data=pd.DataFrame.from_dict(dict(enumerate(data)), orient='index')
            data=pd.concat(dict(enumerate(data)),join='outer',axis=1)
            # data=pd.DataFrame.join(data)
            
            print('Load_data info. Data list length: {}'.format(len(data)))

        # data=pd.concat(data)
        
        # data=pd.concat([pd.read_csv(file).T for file in filelist[:int(importrange*nfiles)]])
    os.chdir(cwd)
        # print('filestring_type {}'.format(type(filestring)))
    
    return data, filestring

    # return data.values


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------


#Helper routines
def calc_gaps(energies):

    """
    A function that calculates the ratios of the adjacent energy gaps in the spectrum.

    INPUT:

    energies - spectrum for a particular realization of disorder of for a particular hamiltonian

    OUTPUT:

    ratios - gap ratios of adjacent energy levels
    avg_ratio - average gap ratio for a given energy spectrum, calculated by taking the mean 
    of ratios array. 
    """

    num_energies=len(energies)
    energies=np.array(energies)[int(0.25*num_energies):int(0.75*num_energies)]
    gaps=energies[1:]-energies[:-1]

    ratios=np.ones(len(gaps)-1, dtype=np.float)
    for i, gap in enumerate(gaps[:-1]):
        ratios[i]=min(gaps[i], gaps[i+1])/max(gaps[i], gaps[i+1])

    #average r in the centre of the spectrum
    # avg_ratio=np.mean(ratios[int(0.25*num_energies):int(0.75*num_energies)])
    avg_ratio=np.mean(ratios)
    return ratios,avg_ratio 

def gap_avg(engylist):

    # nsamples,nener=engylist.shape
    
    """
    A function that calculates the average gap ratio for a given ensemble
    of Hamiltonians/spectra.  

    INPUT: 

    engylist - a list of spectra for different Hamiltonians (for instance due
    to different realizations of disorder.)


    OUTPUT: 

    gap_mean - the function first calculates the average gap ratio avg_ratio
    for each member of the ensemble and then averages those quantities. 
    """

    ratiolist=np.array([calc_gaps(energy)[1] for energy in engylist])
    # print(ratiolist)
    # ratiolist=np.mean(ratiolist,axis=1)

    gap_mean=np.mean(ratiolist)

    return gap_mean

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#UNFOLDING FUNCTION
def unfold_function(energies, n=15, bounds=(0.25,0.75)):

    """
    A function that performs spectral unfolding so that the mean level spacing in the spectrum equals 
    one after unfolding. The routine is needed in spectral statistics calculations and in 
    SFF calculations. By unfolding the spectrum we get rid of the scaling effects in the spectra, 
    since the energy density of states does no longer play a role in our considerations. 

    What is done: a cumulative distribution of energies is fitted by the polynomial 
    of degree n ->p_n(energy); we then store the values of the polynomial at our energies ->
    energies_fit = p_n(energies)


    PARAMETERS: 

    energies - energies in the spectrum for a given Hamiltonian 

    n = 15 - the degree of the fitting polynomial

    bounds - which parts of the spectrum to consider. By default the middle half is considered.
    
    OUTPUT: 

    energies_fit - the unfolded energies.
    """
    nener=len(energies) #number of all energies
    low=bounds[0]
    up=bounds[1]

    #take only the middle part of the spectrum 
    energies_new=energies[int(low*nener):int(up*nener)]
    nener=len(energies_new)
    #cumulative distribution
    cum_dist=np.arange(1, nener+1,1)

    #fit the polynomial 
    z=np.polyfit(energies_new, cum_dist, n)
    #get the polynomial form so that the energies can be plugged in 
    p=np.poly1d(z)

    #plug the energies in, obtain the functional form for the energies
    energies_fit=p(energies_new)
    # print(energies_fit[:10])
    return energies_fit


#create a histogram of various gap averages obtained over different disorder realizations
def gap_avg_hist(engylist, hist_bins=100, hist_bounds=(0,1)):
    """

    NOTE: this one plots a histogram for the case where we observe the statistics of 
    the ratios of the consecutive level spacings. The case with unfolding is a 
    different matter alltogether!


    A routine that retunrs spectraly averaged histogram of the gap ratios
    This is mostly needed in functions used for plotting gap-ratios distributions.

    INPUT: 

    engylist - spectra for different disorder realizations

    hist_bins - number of histogram bins

    hist_bounds - how to bound the histogram.the

    OUTPUT: 

    vals - histogram values

    edges - histogram edges 
    """

    nsamples, nener=engylist.shape
    # nener=int(nener*(bounds[1]-bounds[0]))
    vals=np.zeros((nsamples, hist_bins),dtype=np.float)

    for i, engy in enumerate(engylist):
        ratio=calc_gaps(engy)[0]
        hist,edges=np.histogram(ratio, bins=hist_bins, range=hist_bounds,density=True)
        vals[i]=hist 

    vals=np.mean(vals, axis=0)
    
    return vals, edges

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#SPECTRAL UNFOLDING AND AVERAGING OVER DIFFERENT DISORDER REALIZATIONS
#perform polynomial spectral unfolding
def spectral_unfolding(energies, n=15, bounds=(0.25, 0.75), hist_bins=100, hist_bounds=(0,4) ):
    """
    Performs spectral unfolding for an eigenspectrum and returns a histogram of data

    INPUT: 

    energies - energies in the polynomial eigenspectrum
    n - fitting polynomial degree
    bounds (optional) - which part of spectrum to consider
    hist_bins - histogram bins


    """

    # nener=len(energies)
    # low=bounds[0]
    # up=bounds[1]

    # #take only the middle part of the spectrum 
    # energies_new=energies[int(low*nener):int(up*nener)]
    # nener=len(energies_new)
    # #cumulative distribution
    # cum_dist=np.arange(1, nener+1,1)

    # #fit the polynomial 
    # z=np.polyfit(energies_new, cum_dist, n)
    # #get the polynomial form so that the energies can be plugged in 
    # p=np.poly1d(z)

    #fitted energies
    energies_fit=unfold_function(energies, n, bounds)

    #get the differences between the adjacent energy levels
    s=np.diff(energies_fit)

    #make a histogram -> how many energies differ by a certain amount
    hist,edges=np.histogram(s, bins=hist_bins, range=hist_bounds, density=True)

    #hist - histogram values, edges - histogram edges 
    return hist, edges

    #perform a histogram of the data

#average over different unfolded spectra for various realizations of disorder
def spectral_unfolding_average(engylist,  n=15, bounds=(0.25, 0.75), hist_bins=100, hist_bounds=(0,4) ):

    """
    Performs the ensemble average of spectrally unfolded values - averaging is over 
    different disorder realizations in our case.

    The spectral_unfolding() function is used in order to make calculations on each separate sample
    

    We take constant bin widths in our examples and only average over the histogram values
    """

    nsamples, nener=engylist.shape #first entry in engylist.shape =>number of energies, nsamples
    #initialize an array for histogram values 
    nener=int(nener*(bounds[1]-bounds[0]))

    vals=np.zeros((nsamples, hist_bins),dtype=np.float)
    for i, engy in enumerate(engylist):

        hist,edges=spectral_unfolding(engy, n, bounds, hist_bins, hist_bounds )
        vals[i]=hist  
        

    vals=np.mean(vals, axis=0)

    return vals, edges 

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------



#wigner surmise
def wigner_surmise(x, cumulative=False):

    """
    Wigner surmise for the Wigner-Dyson statistics
    in the ergodic case where level crossings are forbidden and
    thus level repulsion is observed - no levels with zero spacing

    This is relevant for the case when we observe the ratios of the 
    consecutive energy levels, not for the case with unfolding!
    """
    if cumulative:

        return 1.-np.exp(-(np.pi*x**2)/(4))

    else:
        return (np.pi/2)*x*np.exp(-x**2*np.pi/4.)

#poisson statistics
def poisson(x, cumulative=False):

    """
    Poisson function for the Poisson statistics in 
    in the MBL or integrable case. In the latter, conserved quantities
    lead to degeneracies whihc cause level crossing and level attraction.
    Accidental degeneracies cause a similar effect in the MBL phase.

    This is relevant for the case when we observe the ratios of the 
    consecutive energy levels, not for the case with unfolding!
    """
    if cumulative:
        return 1-np.exp(-x)

    else:
        return np.exp(-x)

#check for spectral degeneracy
def check_deg(energies, eps=1e-013):
    """
    A routine that checks spectral degeneracy
    

    INPUT: 
    energies - energy spectrum 

    eps - accuracy treshold for determining whether 
    energies are the same or not

    OUTPUT:  
    returns an array eng_arr:

    first row - energies
    second row - consecutive energy levels
    third row - degeneracies
    """

    eng_arr=np.zeros((3,len(energies)), dtype=np.float)
    eng_arr[0,:]=energies

    j=0
    eng_arr[1,0]=j

    k=1
    for i in range(1, len(energies)):

        diff=energies[i]-energies[i-1]
        
        eng_arr[2,i-k:i]=k
        if np.abs(diff) <= eps:

            k+=1

        else:
            if i!=len(energies)-1:
                k=1
            j+=1

        # eng_arr[2,i-k:i]=k
        eng_arr[1,i]=j

    # for i, elt in enumerate(eng_arr[1,:]):
    i=len(energies)-1
    eng_arr[2,i-k+1:i+1]=k

    return eng_arr
            


    """
    0 1 2 2 2 3 4 4 4 5 6 7 7 
    """

#returns levels and their corresponding degeneracies
def get_level_deg(eng_arr, eps=1e-013):
    """
    #returns levels and their corresponding degeneracies
    # eng_arr=check_deg(energies, eps)
    INPUT: 

    eng_arr - output of the check_deg function

    eps - accuracy treshold for determining whether 
    energies are the same or not 
    """

    lvls=[eng_arr[1,0]]
    degs=[eng_arr[2,0]]


    for i, lvl in enumerate(eng_arr[1,1:]):

        if lvl != eng_arr[1,i]:
            lvls.append(lvl)
            degs.append(eng_arr[2,i+1])


    return np.array([lvls, degs])



#get number of states in the system
def get_num_states(size, nh, nu):

    return comb(size, nh, exact=True)*comb(size-nh, nu, exact=True)


def running_mean(x, N):

    """
    Input: 
    x - signal
    N - window width
    """

    cumsum=np.cumsum(np.insert(x,0,0))
    
    return (cumsum[N:]-cumsum[:-N])/float(N)


def K_GOE(taulist):
    """
    GOE spectral form factor
    """
    #there are two limiting cases, one for tau<1, the other for tau>1

    tau_small=taulist[np.where(taulist<=1)]
    tau_large=taulist[np.where(taulist>1)]

    K_small=2*tau_small - tau_small*np.log(1+2*tau_small)
    K_large=2 - tau_large*np.log((2*tau_large+1)/(2*tau_large-1))
    return np.append(K_small, K_large)
#---------------------------------------------------------------------------------
#CLASS FOR EASIER STORAGE OF DATA AND THEIR ASSOCIATED MODEL AND SYSTEM PARAMETERS
class dataSet(object):

    """
    A special class for eigenvals data
    """

    def __init__(self,desc,modules, quantity,syspar, modpar, all_cases=True, importrange=1):
        super(dataSet, self).__init__()
        self.desc=desc
        self.modules=modules
        self.quantity=quantity
        self.syspar=syspar
        self.modpar=modpar
        
        self.system=sys_str(self.syspar)
        self.eigsys_folder=get_eigsys_folder(desc, modules, self.system)
        # self.data, self.filestring=load_data(self.quantity, self.desc, self.modules, self.syspar, self.modpar,all_cases, importrange)
        
        #NOTE!!!: self.data is a pd.DataFrame()
        self.data=pd.DataFrame()
        self.filestring=''
        self.all_cases=all_cases
        self.importrange=importrange

        # self.nsamples=self.data.shape[0]

    def load_dataSet(self, filetype='bin', new_ordering=False):

        """
        Filestring is the substring that remains after the quantity has been taken out from the 
        filename.

        filetype: 'bin' or 'txt' for binary and text files
        """
        self.data, self.filestring=load_data(self.quantity, self.desc, self.modules, self.syspar, self.modpar,self.all_cases, self.importrange,filetype=filetype, new_ordering=new_ordering)

    def mean_data(self):
        """ 
        Properly groups data together and takes the mean along groups
        The averaged data are still a pandas DataFrame.
        """
        df=self.data
        if self.quantity=='Ent_entropy':
            return df.groupby(df.index).mean()
        else:
            return df.mean()
    # def group_engies(self):
    #     """
    #     Groups energies together and renormalizes them to the [0,1] interval;
    #     needed in ent_entropy calculations
    #     """
    #     df=self.data
    #     if self.quantity=='Ent_entropy':

    #         #groups data together
    #         df=df.groupby(df.index)


    #     else:
    #         pass


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

##PLOTTER ROUTINES
## H or W sweep for average r

def set_sybr_values(staggered, modpar, sybr):

    if not staggered:
        print('Symmetry breaking terms present! Their values will now be set.')
        modpar['HSYM']=sybr
        modpar['WSYM']=sybr

    else:
        print('Symmetry breaking terms not present! Setting H_STAGG value instead.')
        modpar['H_STAGG']=sybr


    return modpar

def r_mean_sweep_graph(wlist, sybr_list, sweep='W', desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, filetype='bin'):



    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    
    sweep_string=sweep_types[sweep_dict[sweep]]

    staggered=False

    if all( param in modpar for param in ['WSYM', 'HSYM']):
        staggered=False
        label='$H_\mathrm{{sym}}=W_\mathrm{{sym}}={:g}$'
        nametag='_disorder_sym_break_'

    if 'H_STAGG' in modpar:
        staggered=True
        label='$H_\mathrm{{\\perp}}={:g}$'
        nametag='_h_staggered_'


    fig,ax=plt.subplots(1,1)
    print(modstr)
    ax.set_ylabel('$\langle r\\rangle$')
    for sybr in sybr_list:
        r_mean=[]
        wplot=[]


        modpar=set_sybr_values(staggered, modpar,sybr)

        for W in wlist:

            modpar[sweep_string]=W
            file=dataSet(desc, modstr, 'Eigvals', syspar, modpar)
            file.load_dataSet(filetype=filetype)
            # print(file.nsamples)
            if file.data.values is not None:

                print(file.data.values)
                r=gap_avg(file.data.values)
                r_mean.append(r)
                wplot.append(W)
                print(r_mean)
                print(wplot)

        ax.plot(wplot, r_mean, marker='o', label=label.format(sybr))

    prepare_ax(ax, True)


    ax.set_xlabel('${0}$, ${1}=0$'.format(sweep_string, sweep_types[1-sweep_dict[sweep]]))
    ax.set_title('$L={}$, $N_h={}$, $N_u={}$'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu']))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    
    savename=sweep_string+'{}{}_{}_{}.pdf'.format(nametag,file.syspar['size'], file.syspar['ne'],file.syspar['nu'])
    prepare_plt(savename=savename, desc=desc, plot_type='Mean_r_stats')

def r_mean_sweep_graph_double(sizelist, wlist, sybr_list,rules, desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, save=True, slo=True ):


    fig, axarr, fontsize=prepare_axarr(1,2, fontsize=[17,20,24])
    # fontsize=[17,20,24]

    titlestring='\n $\mathrm{{t}}={:.2f},\\ '.format(modpar['T'])

    if modpar['J']==modpar['JOF']:
        titlestring+=' \mathrm{{J}}={:.2f},\\ $'.format(modpar['J'])
    else:
        titlestring+='\mathrm{{J}}={:.2f}, \mathrm{{J}}_\mathrm{{\\perp}}={:.2f},\\ $'.format(modpar['J'], modpar['JOF'])


    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    if slo:
        sweep_names=['Spinski nered','Vrzelni nered']
    else:
        sweep_names=['Spin disorder','Hole disorder']

    labels={'size':

        '$\mathrm{{L}}={}, \mathrm{{N_h}}={}, \mathrm{{N_u}}={}$'

         , 
            'sybr':{

            False: '$\mathrm{{H}}_\mathrm{{sym}}=\mathrm{{W}}_\mathrm{{sym}}={:.2f}$',
            True: '$\mathrm{{H}}_\mathrm{{\\perp}}={:.2f}$'

            }}
    staggered=False
    label_sizes=False


    if all( param in modpar for param in ['WSYM', 'HSYM']):
        staggered=False
        # label='$H_\mathrm{{sym}}=W_\mathrm{{sym}}={:g}$'
        nametag='_disorder_sym_break_'
    if 'H_STAGG' in modpar:
        staggered=True
        # label='$H_\mathrm{{\\perp}}={:g}$'
        nametag='_h_staggered_'    

    if len(sybr_list)==1:
        label=labels['size']
        label_sizes=True
        titlestring+=labels['sybr'][staggered].format(sybr_list[0])

        
    elif len(sizelist)==1:
        label=labels['sybr'][staggered]
        titlestring+=labels['size'].format(size, syspar['ne'], syspar['nu'])


    for size in sizelist:

        syspar['size']=size

        for key in rules:
            syspar.update(rules[key](syspar[key], syspar))  
            print(syspar)         

        for sybr in sybr_list:
            modpar=set_sybr_values(staggered, modpar,sybr)
        

            for i, ax in enumerate(axarr.flatten()):
                r_mean=[]
                wplot=[]

                for W in wlist:
                    # modpar['W']=W
                    modpar[sweep_types[i]]=W
                    modpar[sweep_types[1-i]]=0

                    file=dataSet(desc, modstr, 'Eigvals', syspar, modpar)
                    file.load_dataSet()
                # print(file.nsamples)
                    if file.data.values is not None:
                        r=gap_avg(file.data.values)
                        r_mean.append(r)
                        wplot.append(W)




                if label_sizes:
                    # print(label)
                    label_=label.format(file.syspar['size'], file.syspar['ne'], file.syspar['nu'])
                else:
                    label_=label.format(sybr)               


                r_mask=np.isfinite(r_mean)
 
                ax.plot(np.array(wplot)[r_mask], np.array(r_mean)[r_mask], marker='o', label=label_)

    for i,ax in enumerate(axarr.flatten()):
        if i==0:
            ax.set_ylabel('$\langle \\tilde{r}\\rangle$',fontsize=fontsize[1])
        ax.set_xlabel('$\mathrm{{{0}}}$'.format(sweep_types[i]),fontsize=fontsize[1])            
        ax.set_title('{}'.format(sweep_names[i]), fontsize=fontsize[-1])
        ax.xaxis.set_major_locator(MultipleLocator(2))              
        prepare_ax(ax, plot_mbl_ergodic=True) 
    if slo:
        fig.suptitle('Analiza povpre\\v{c}nega razmika med nivoji $\\langle \\tilde{r} \\rangle$'+titlestring,fontsize=fontsize[-1])

    else:
        fig.suptitle('Mean level spacing analysis'+titlestring,fontsize=fontsize[-1])

    if slo:
        language='_slo'
    else:
        language=''
    savename='double_plot'+'{}{}_{}_{}{}.pdf'.format(nametag,file.syspar['size'], file.syspar['ne'],file.syspar['nu'],language)
    prepare_plt(savename, desc=desc, plot_type='Mean_r_stats',save=save, top=0.84)



def r_mean_sweep_scaling_graph(wlist, sizelist,sybr=0.5, sweep='H', desc=desc, modstr=modstr, syspar=syspar, modpar=modpar,importrange=1 ):

    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']

    sweep_string=sweep_types[sweep_dict[sweep]]


    fig,ax=plt.subplots(1,1)
    print(modstr)
    ax.set_ylabel('$\langle r\\rangle$')
    for size in sizelist:
        r_mean=[]
        wplot=[]
        syspar['size']=size
        syspar['nu']=int((syspar['size']-syspar['ne'])/2)
        modpar['HSYM']=sybr
        modpar['WSYM']=sybr

        for W in wlist:

            modpar[sweep]=W
            # modpar[sweep_types[1-sweep_dict[sweep]]]=0. 
            file=dataSet(desc, modstr, 'Eigvals', syspar, modpar, importrange=importrange)
            # print(file.nsamples)
            if file.data is not None:
                r=gap_avg(file.data)
                r_mean.append(r)
                wplot.append(W)

        ax.plot(wplot, r_mean, marker='o', label='$L=%g$'%size)

    ax.axhline(y=ergodic, ls='--', color='green')
    # ax.axhline(y=0.44, ls='--', color='green')
    # ax.axhline(y=ergodic, ls='--', color='red')
    # plt.xlabel('${0}$, ${1}=0$, $H_\mathrm{{sym}}=W_\mathrm{{sym}}={2}$'.format(sweep_string, sweep_types[1-sweep_dict[sweep]],sybr))
    ax.set_xlabel('${0}$, ${1}=0$, $H_{{sym}}=W_{{sym}}={2}$'.format(sweep_string, sweep_types[1-sweep_dict[sweep]], file.modpar['WSYM']))
    ax.set_title('Scaling analysis, $N_h={}$, $S^z=0$'.format(file.syspar['ne']))
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    graphs_folder='./graphs_'+desc
    if not os.path.isdir(graphs_folder):
        os.makedirs(graphs_folder)

    plt.savefig(graphs_folder+'/' +sweep_string+'_scaling_analysis_sym_break_Nh{}.pdf'.format(file.syspar['ne']))
    plt.show()


def r_density_plot(wlist=[0,2,4,6], desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, importrange=1, load=True, save=True, new_ordering=True):

    filesave='{}_{}_{}_{}.npy'.format(desc,syspar['size'], syspar['ne'], syspar['nu'] )

    load_name=load_data_file(filesave,'dens',desc=desc,load=False)


    num=len(wlist)
    rvals=np.zeros((num,num),dtype=np.float)

    w_min=h_min=wlist[0];w_max=h_max=wlist[-1]; dy=wlist[1]-wlist[0]


    w_vals,h_vals=np.meshgrid(np.linspace(w_min,w_max+dy,num+1)-dy/2,np.linspace(h_min,h_max+dy,num+1)-dy/2)

    print(w_vals)
    
    fig, axarr, fontsize=prepare_axarr(1,1, fontsize=[17,20,24])


    #if the file is not to be loaded or if it does not exist:
    if ((not load) or not os.path.isfile(load_name)):
        for i,W in enumerate(wlist):
            modpar['W']=W
            for j,H in enumerate(wlist):

                modpar['H']=H
                file=dataSet(desc, modstr, 'Eigvals', syspar, modpar)

                file.load_dataSet(new_ordering=new_ordering)
                if file.data.values is not None:
                    r=gap_avg(file.data.values)
                    rvals[i,j]=r
    else:
        
        rvals=load_data_file(filesave,'dens', desc=desc)

    #save the data
    save_data_file(rvals,filesave,'dens',desc=desc)
    # print(rvals)
    axarr.set_xticks(np.arange(w_min, w_max+dy, 1))
    axarr.set_yticks(np.arange(h_min, h_max+dy, 1))

    axarr.axis([w_vals.min(), w_vals.max(), h_vals.min(), h_vals.max()])
    im=axarr.pcolormesh(w_vals, h_vals, rvals.T, linewidth=0, rasterized=True)
    cbar=plt.colorbar(im,fraction=0.08,orientation='horizontal')
    cbar.ax.invert_xaxis()
    cbar.ax.tick_params(labelsize=fontsize[-1],direction='out')
    cbar.ax.set_xlabel('$\\langle \\tilde{r} \\rangle$',fontsize=fontsize[1])
    prepare_ax(axarr, fontsize=fontsize, legend=False, grid=False)
    axarr.set_xlabel('$W$', fontsize=fontsize[1])
    axarr.set_ylabel('$H$', fontsize=fontsize[1])
    axarr.set_title('$\\langle \\tilde{{r}} \\rangle$, $L={}$, $N_h={}$, $N_u={}$'.format(syspar['size'], syspar['ne'], syspar['nu']), fontsize=fontsize[-1])
    savename='r_density_{}_{}_{}.pdf'.format(syspar['size'],syspar['ne'],syspar['nu'])

    prepare_plt(plot_type='Mean_r_stats',save=True, show=True, savename=savename, desc=desc)







##---------------------------------------------------------------------------------
## SYM BREAK TEST

def sym_break_test_graph(Wlist,sizelist,sweep='both', desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True,slo=True):

    """ 
    Sweep options: 'both', 'H', 'W'
    """
    # Wlist=[ 1.,2., 3. ,4., 5., 6., 7., 8.  ]
    # Wlist=np.logspace(-3,1,25)[:-1]
    # Wlist=[0.001]
    titlestring='$L={}$'.format(syspar['size'])
    if len(sizelist)>1:
        if slo:
            titlestring='skalirna analiza'
        else:
            titlestring='scaling analysis'
    fig=plt.figure()

    
    plt.ylabel('$\langle \\tilde{r}\\rangle$')
    
    modpar['WSYM']=0
    modpar['HSYM']=0
    if sweep=='both':
        if slo:
            typestring='Spinski in vrzelni'
        else:
            typestring='Spin and hole'
        labelstring='$W_\mathrm{{sym}}=H_\mathrm{{sym}}$'
        savestring='hole_spin_'
    elif sweep=='H':
        if slo:
            typestring='Vrzelni'
        else:
            typestring='Hole'
        labelstring='$H_\mathrm{{sym}}$'
        savestring='hole_'
    elif sweep=='W':
        if slo:
            typestring='Spinski'
        else:
            typestring='Spin'
        labelstring='$W_\mathrm{{sym}}$'
        savestring='spin_'

    plt.xlabel(labelstring)

    for size in sizelist:
        r_mean=[]
        Wplot=[]
        syspar['size']=size
        syspar['nu']=int((syspar['size']-syspar['ne'])/2)

        for W in Wlist:

            if sweep=='both':
                modpar['WSYM']=W
                modpar['HSYM']=W
            elif sweep=='H':
                modpar['HSYM']=W
            elif sweep=='W':
                modpar['WSYM']=W 

            file=dataSet(desc, modstr, 'Eigvals', syspar, modpar, all_cases=True)
            file.load_dataSet(filetype='txt')

            if file.data is not None:

                gap=calc_gaps(file.data.values[0])
                nsamples=len(gap)
                r_mean.append(np.mean(gap[int(0.25*nsamples):int(0.75*nsamples)]))
                Wplot.append(W)
            
        plt.semilogx(Wplot, r_mean, marker='o', label='L={:g}'.format(syspar['size']))

    if slo:
        title='{} tJ zlom simetrije, {}, $S^z=0$, $N_h={}$'
    else:
        title='{} tJ symmetry breaking {}, $S^z=0$, $N_h={}$'
    plt.title(title.format(typestring, titlestring, file.syspar['ne']))
    if slo:
        holename='{vrzel }'
        nodis='{Brez nereda}'
    else:
        holename='{hole }'
        nodis='{No disorder}'
    plt.text(1,0.2, '$\hat{{h}}_\mathrm{{spin}}=W\hat{{S}}^\\mathrm{{z}}_1$ \n $\hat{{h}}_\\mathrm {}= H\hat{{n}}_\mathrm{{L}}$ \n \\textbf{}'.format(holename, nodis))
    plt.axhline(y=ergodic, color='red', ls='--')
    plt.axhline(y=mbl, color='green',ls='--')
    if len(sizelist)>1:
        plt.legend()
    # plt.semilogx(Wplot, r_mean, marker='o')
    graphs_folder='./graphs_'+desc
    if not os.path.isdir(graphs_folder):
        os.makedirs(graphs_folder)

    if slo:
        language='_slo'
    else:
        language=''

    savename='{}disorder_sym_break_{}_{}_{}{}.pdf'.format(savestring,file.syspar['size'], file.syspar['ne'],file.syspar['nu'],language)
    prepare_plt(desc=desc,plot_type='Sym_break',savename=savename, save=True)
    # plt.savefig(graphs_folder+'/{}disorder_sym_break_{}_{}_{}{}.pdf'.format(savestring,file.syspar['size'], file.syspar['ne'],file.syspar['nu'],language))
    # plt.show()

#unfolding plots

#demonstration of the unfolding procedure

def plot_unfolding_schematics(importrange=1,nvals=[2,3,5]):

    """
    A schematic to show how the unfolding procedure works
    """

    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    # print(modstr)
    size=5
    nu=1
    ne=3
    sybr=0.5
    syspar={'size':size, 'nu':nu, 'ne':ne, 'dim':1, 'base':'F'}
    w=8
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':sybr, 'WSYM':sybr,'W':0, 'H':w }
    # rules_one_hole_doping={'ne': lambda x,y: {'ne': 1},
    #      'nu': lambda x,y: {'nu': (y['size']-y['ne'])/2},
    #       } #rules for one third doping and Sz=0 case

    desc='plot_unfolding_schematics_sym_break'

    #load data

    file=dataSet(desc, modstr, 'Eigvals', syspar, modpar, importrange=importrange)
    
    file.load_dataSet()

    spectrum=file.data.values[0]
    nener=len(spectrum)

    spectrum=spectrum[int(0.25*nener):int(0.75*nener)]
    cumulative=np.arange(1,len(spectrum)+1,1)



    #plug the energies in, obtain the functional form for the energies
    
    xvals=np.linspace(np.min(spectrum), np.max(spectrum),100)

    fig, axarr, fontsize=prepare_axarr(1,1, fontsize=[18,21,23])
    ax=axarr
    # plt.xticks(spectrum)
    # plt.yticks(spectrum_fit)
    ax.scatter(spectrum, cumulative, color='red', label='spekter')

    for n in nvals:
        z=np.polyfit(spectrum, cumulative, n)
        p=np.poly1d(z)
        spectrum_fit=p(xvals)

        ax.plot(xvals, spectrum_fit, label='$n={:g}$'.format(n))
    # ax.yaxis.set_major_locator(MultipleLocator(2))
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    prepare_ax(ax, fontsize=fontsize)
    ax.yaxis.grid(True, which='major', linestyle='-')
    ax.yaxis.grid(True, which='minor', linestyle='--')
    ax.set_xlabel('$E$', fontsize=fontsize[1])
    ax.set_ylabel('$\\tilde{N}(E)$', fontsize=fontsize[1])
    ax.set_title('Razgrnjenje spektra s polinomi razli\\v{c}nih stopenj $n$', fontsize=fontsize[-1])
    # ax.legend()
    prepare_plt(plot_type='Schematics',savename='unfolding_schematics.pdf', save=True, desc=desc)





#the actual plots of unfolded data histograms

def plot_spectral_unfolding(Wlist=[0,8.], n=15, bounds=(0.25,0.75), hist_bins=20, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True):
    

    fontsize=[18,21,25]


    fig, axarr=plt.subplots(2,2, sharey=True, sharex=True, figsize=(16,12))
    fig.suptitle('Level statistics of the tJ model unfolded spectra, $L={}, N_h={}, N_u={}$,  $W_\mathrm{{sym}}=H_\mathrm{{sym}}={}$'.format(syspar['size'], syspar['ne'],syspar['nu'], modpar['WSYM']), fontsize=fontsize[-1])

    axarr[0][0].set_title('Spin disorder $W$, $H=0$', fontsize=fontsize[-1])
    axarr[0][1].set_title('Hole disorder $H$, $W=0$', fontsize=fontsize[-1])
 
    for i,W in enumerate(Wlist):
        sweep_types=[W,0]
        sweep_strings=['W','H']
        x=np.linspace(hist_bounds[0],hist_bounds[1],1000)
        for j in [0,1]:
            modpar['W']=sweep_types[j]
            modpar['H']=sweep_types[1-j]

            file=dataSet(desc, modstr, 'Eigvals', syspar, modpar)
            hist, edges=spectral_unfolding_average(file.data, n, bounds, hist_bins, hist_bounds  )
        
            axarr[i,j].bar(edges[:-1], hist,fill=False,width=np.diff(edges), align='edge',alpha=1,label='${}={:g}$'.format(sweep_strings[j],W))
            if i*j!=1:
                axarr[i,j].plot(x, wigner_surmise(x), color='red', ls='-')
                axarr[i,j].plot(x, poisson(x), color='green', ls='-')
            else:
                axarr[i,j].plot(x, wigner_surmise(x), color='red', ls='-', label='Wigner-Dyson')
                axarr[i,j].plot(x, poisson(x), color='green', ls='-', label='Poisson')


    #prepare the graphs
    for i, ax in enumerate(axarr.flatten()):
       ax.grid(which='both')
       ax.tick_params(axis='x', labelsize=fontsize[1])
       ax.tick_params(axis='y', labelsize=fontsize[1])
       ax.set_xlim(hist_bounds[0], hist_bounds[1])
       if i>1:

        ax.set_xlabel('$s$', fontsize=fontsize[1])
       # ax.xaxis.set_minor_locator(MultipleLocator(0.5))
       # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
       ax.tick_params(axis='x', labelsize=fontsize[1],pad=5,direction='out')
       ax.legend(loc='best',prop={'size':fontsize[1]},fontsize=fontsize[1],framealpha=0.5)

    for ax in axarr[:,0]:
        ax.set_ylabel('$p(s)$', fontsize=fontsize[-1])


    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    graphs_folder='./graphs_'+desc
    if not os.path.isdir(graphs_folder):
        os.makedirs(graphs_folder)


    # filename=desc+'_'+mod_name+'_'+'srule_engy_compare.pdf'
    # plt.show()
    #ax1 - W sweep
    #ax2 - H sweep 
    plt.savefig(graphs_folder+'/unfolded_spectra_stats_{}_{}_{}.pdf'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu']))
    plt.show()


def plot_unfolding_three_demo(Wlist=[0.001,8,16], n=15, bounds=(0.25,0.75), hist_bins=20, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True, importrange=1):

    

    fig, axarr,fontsize=prepare_axarr(1,3, fontsize=[21,26,35])
    fig.suptitle('Model t-J, primer ergodi\\v{{c}}ne, vmesne in MBL nivojske statistike, $L={}, N_h={}, N_u={}$'.format(syspar['size'], syspar['ne'],syspar['nu']), fontsize=fontsize[-1])
    
    axarr[0].set_title('Ergodi\\v{{c}}ni primer, $W={:g}$'.format(Wlist[0]), fontsize=fontsize[-1])
    axarr[1].set_title('Vmesni primer, $W=%g$'%Wlist[1], fontsize=fontsize[-1])
    axarr[2].set_title('MBL, $W=%g$'%Wlist[2], fontsize=fontsize[-1])

    for i,W in enumerate(Wlist):

        modpar['W']=W
        x=np.linspace(hist_bounds[0],hist_bounds[1],1000)

        file=dataSet(desc, modstr, 'Eigvals', syspar, modpar, importrange=importrange)
        file.load_dataSet()
        # print(file.data.values)
        hist, edges=spectral_unfolding_average(file.data.values, n, bounds, hist_bins, hist_bounds  )
    
        axarr[i].bar(edges[:-1], hist,fill=False,width=np.diff(edges), align='edge',alpha=1, label='izra\\v{c}un')
        axarr[i].plot(x, wigner_surmise(x), color='red', ls='-',label='Wigner-Dyson')
        axarr[i].plot(x, poisson(x), color='green', ls='-', label='Poisson')

    for i, ax in enumerate(axarr.flatten()):
        prepare_ax(ax,legend=False)
        ax.set_xlim(hist_bounds[0], hist_bounds[1])
        ax.set_xlabel('$s$', fontsize=fontsize[1])
        if i==0:
            ax.set_ylabel('$p(s)$', fontsize=fontsize[-1])
        if i==2:
            ax.legend(loc='best',prop={'size':fontsize[1]},fontsize=fontsize[1],framealpha=0.5)


    axarr[0].text(2,0.4,'$p_\mathrm{{WD}}(s)=\\frac{\pi}{2}s\exp\left(-\\frac{\pi}{4} s^2 \\right)$',color='red',fontsize=fontsize[-2])
    axarr[2].text(2,0.4,'$p_\mathrm{{P}}(s)=\exp\left(-s \\right)$',color='green',fontsize=fontsize[-2])

    prepare_plt(save=True, plot_type='Level_statistics',desc=desc, savename='unfolding_demo_three_slo.pdf', top=0.82)

def plot_unfolding_three_error(Wlist=[0.001,8,16], n=15, bounds=(0.25,0.75), hist_bins=20, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True):

    fig, axarr,fontsize=prepare_axarr(1,3, fontsize=[22,28,33], sharey=False)
    fig.suptitle('Odstopanje ergodi\\v{{c}}ne, vmesne in MBL nivojske statistike od napovedi, $L={}, N_h={}, N_u={}$'.format(syspar['size'], syspar['ne'],syspar['nu']), fontsize=fontsize[-1])
    
    axarr[0].set_title('Ergodi\\v{{c}}ni primer, $W={:g}$'.format(Wlist[0]), fontsize=fontsize[-1])
    axarr[1].set_title('Vmesni primer, $W=%g$'%Wlist[1], fontsize=fontsize[-1])
    axarr[2].set_title('MBL, $W=%g$'%Wlist[2], fontsize=fontsize[-1])

    for i,W in enumerate(Wlist):

        modpar['W']=W
        x=np.linspace(hist_bounds[0],hist_bounds[1],1000)

        file=dataSet(desc, modstr, 'Eigvals', syspar, modpar)
        file.load_dataSet()
        # print(file.data.values)
        hist, edges=spectral_unfolding_average(file.data.values, n, bounds, hist_bins, hist_bounds  )
        print(edges)
        x=(edges[:-1]+edges[1:])/2
        # axarr[i].bar(edges[:-1], hist,fill=False,width=np.diff(edges), align='edge',alpha=1, label='izra\\v{c}un')
        axarr[i].plot(x, np.log10(np.abs(wigner_surmise(x)-hist)), color='red', ls='-',label='Wigner-Dyson')
        axarr[i].plot(x, np.log10(np.abs(poisson(x)-hist)), color='green', ls='-', label='Poisson')

    for i, ax in enumerate(axarr.flatten()):
        prepare_ax(ax,legend=False)
        ax.set_xlim(hist_bounds[0], hist_bounds[1])
        ax.set_xlabel('$s$', fontsize=fontsize[1])
        if i==0:
            ax.set_ylabel('$\\log_{10}|p(s)_\\mathrm{num.}-p(s)_\\mathrm{teor.}|$', fontsize=fontsize[1])
        if i==2:
            ax.legend(loc='best',prop={'size':fontsize[1]},fontsize=fontsize[1],framealpha=0.5)


    # axarr[0].text(2,0.4,'$p_\mathrm{{WD}}(s)=\\frac{\pi}{2}s\exp\left(-\\frac{\pi}{4} s^2 \\right)$',color='red',fontsize=fontsize[-2])
    # axarr[2].text(2,0.4,'$p_\mathrm{{P}}(s)=\exp\left(-s \\right)$',color='green',fontsize=fontsize[-2])

    prepare_plt(save=True, plot_type='Level_statistics',desc=desc, savename='unfolding_demo_three_error_slo.pdf', top=0.83)

def plot_unfolding_kolm_smir(Wlist=[0.001,8,16], n=10, bounds=(0.25,0.75), hist_bins=40, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True):

    fig, axarr,fontsize=prepare_axarr(1,3, fontsize=[22,28,33], sharey=False,sharex=False)
    fig.suptitle('Odstopanje ergodi\\v{{c}}ne, vmesne in MBL nivojske statistike od napovedi, $L={}, N_h={}, N_u={}$'.format(syspar['size'], syspar['ne'],syspar['nu']), fontsize=fontsize[-1])
    
    axarr[0].set_title('Ergodi\\v{{c}}ni primer, $W={:g}$'.format(Wlist[0]), fontsize=fontsize[-1])
    axarr[1].set_title('Vmesni primer, $W=%g$'%Wlist[1], fontsize=fontsize[-1])
    axarr[2].set_title('MBL, $W=%g$'%Wlist[2], fontsize=fontsize[-1])

    for i,W in enumerate(Wlist):

        modpar['W']=W
        x=np.linspace(0,5,1000)

        file=dataSet(desc, modstr, 'Eigvals', syspar, modpar)
        file.load_dataSet()
        datalen=len(file.data.values)
        test_vals=np.zeros((2,datalen))
        # print(file.data.values)
        for j,value in enumerate(file.data.values):
            energies_fit=unfold_function(value, n=n)
            diffs=np.diff(energies_fit)
            
                # wig_dys=wigner_surmise(x,True)
                # poiss=poisson(x, True)
            #cumulative distribution of hist

            #perform tests
            
            test_vals[0,j]=stats.kstest(diffs, lambda x:wigner_surmise(x,True))[0]
            test_vals[1,j]=stats.kstest(diffs, 'expon')[0]
            

        # axarr[i].plot(x,1-special.kolmogorov(x),lw=2,color='red',label='napoved')
        axarr[i].hist(np.sqrt(datalen)*test_vals[0],bins='auto',cumulative=True,normed=1,label='simulacija')
        # axarr[i].bar(edges[:-1], hist,fill=False,width=np.diff(edges), align='edge',alpha=1, label='izra\\v{c}un')
        # axarr[i].plot(x, np.log10(np.abs(wigner_surmise(x)-hist)), color='red', ls='-',label='Wigner-Dyson')
        # axarr[i].plot(x, np.log10(np.abs(poisson(x)-hist)), color='green', ls='-', label='Poisson')

    for i, ax in enumerate(axarr.flatten()):
        prepare_ax(ax,legend=False)
        # ax.set_xlim(hist_bounds[0], hist_bounds[1])
        ax.set_xlabel('$s$', fontsize=fontsize[1])
        if i==0:
            ax.set_ylabel('$\\log_{10}|p(s)_\\mathrm{num.}-p(s)_\\mathrm{teor.}|$', fontsize=fontsize[1])
        if i==2:
            ax.legend(loc='best',prop={'size':fontsize[1]},fontsize=fontsize[1],framealpha=0.5)

    plt.show()

def plot_unfolding_demo(Wlist=[0.001,16], n=15, bounds=(0.25,0.75), hist_bins=20, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True):
    fontsize=[18,21,25]

    # print(syspar)
    # print(modpar)
    fig, axarr=plt.subplots(1,2, sharey=True, sharex=True, figsize=(16,7))


    fig.suptitle('tJ model, examplary ergodic and MBL level statistics, $L={}, N_h={}, N_u={}$'.format(syspar['size'], syspar['ne'],syspar['nu'], modpar['WSYM']), fontsize=fontsize[-1])

    axarr[0].set_title('Ergodic, $W=%g$'%Wlist[0], fontsize=fontsize[-1])
    axarr[1].set_title('MBL, $W=%g$'%Wlist[1], fontsize=fontsize[-1])
    
    # modpar['H']=0

    for i,W in enumerate(Wlist):

        modpar['W']=W
        x=np.linspace(hist_bounds[0],hist_bounds[1],1000)

        file=dataSet(desc, modstr, 'Eigvals', syspar, modpar)
        file.load_dataSet()
        hist, edges=spectral_unfolding_average(file.data, n, bounds, hist_bins, hist_bounds  )
    
        axarr[i].bar(edges[:-1], hist,fill=False,width=np.diff(edges), align='edge',alpha=1, label='simulation')
        if i==0:
            axarr[i].plot(x, wigner_surmise(x), color='red', ls='-',label='Wigner-Dyson')
            # axarr[i].plot(x, poisson(x), color='green', ls='-')
        else:
            # axarr[i,j].plot(x, wigner_surmise(x), color='red', ls='-', label='Wigner-Dyson')
            axarr[i].plot(x, poisson(x), color='green', ls='-', label='Poisson')


    #prepare the graphs
    for i, ax in enumerate(axarr.flatten()):
       ax.grid(which='both')
       ax.tick_params(axis='x', labelsize=fontsize[1])
       ax.tick_params(axis='y', labelsize=fontsize[1])
       ax.set_xlim(hist_bounds[0], hist_bounds[1])
       ax.set_xlabel('$s$', fontsize=fontsize[1])
       # ax.xaxis.set_minor_locator(MultipleLocator(0.5))
       # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
       ax.tick_params(axis='x', labelsize=fontsize[1],pad=5,direction='out')
       ax.legend(loc='best',prop={'size':fontsize[1]},fontsize=fontsize[1],framealpha=0.5)
       if i==0:
        ax.set_ylabel('$p(s)$', fontsize=fontsize[-1])

    axarr[0].text(2,0.4,'$p_\mathrm{{WD}}(s)=\\frac{\pi}{2}s\exp\left(-\\frac{\pi}{4} s^2 \\right)$',color='red',fontsize=fontsize[-1])
    axarr[1].text(2,0.4,'$p_\mathrm{{P}}(s)=\exp\left(-s \\right)$',color='green',fontsize=fontsize[-1])
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    graphs_folder='./graphs_'+desc
    if not os.path.isdir(graphs_folder):
        os.makedirs(graphs_folder)


    # filename=desc+'_'+mod_name+'_'+'srule_engy_compare.pdf'
    # plt.show()
    #ax1 - W sweep
    #ax2 - H sweep 
    plt.savefig(graphs_folder+'/unfolded_demo_{}_{}_{}.pdf'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu']))
    plt.show() 


#level statistics
def plot_ratio_statistics(Wlist=[0,9.], hist_bins=20, hist_bounds=(0,1), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True):

    fontsize=[18,21,25]


    fig, axarr=plt.subplots(2,2, sharey=True, sharex=True, figsize=(16,12))
    fig.suptitle('Statistics of the tJ model gap ratios, $L={}, N_h={}, N_u={}$,  $W_\mathrm{{sym}}=H_\mathrm{{sym}}={}$'.format(syspar['size'], syspar['ne'],syspar['nu'], modpar['WSYM']), fontsize=fontsize[-1])

    axarr[0][0].set_title('Spin disorder $W$, $H=0$', fontsize=fontsize[-1])
    axarr[0][1].set_title('Hole disorder $H$, $W=0$', fontsize=fontsize[-1])
 
    for i,W in enumerate(Wlist):
        sweep_types=[W,0]
        sweep_strings=['W','H']
        x=np.linspace(hist_bounds[0],hist_bounds[1],1000)
        for j in [0,1]:
            modpar['W']=sweep_types[j]
            modpar['H']=sweep_types[1-j]

            file=dataSet(desc, modstr, 'Eigvals', syspar, modpar)
            hist, edges=gap_avg_hist(file.data, hist_bins, hist_bounds  )
        
            # axarr[i,j].bar(edges[:-1], hist,fill=False,width=np.diff(edges), align='edge',alpha=1,label='${}={:g}$'.format(sweep_strings[j],W))
            axarr[i,j].step(edges[:-1], hist,label='${}={:g}$'.format(sweep_strings[j],W))
            x=np.linspace(0,1,1000)
            axarr[i,j].plot(x,2/(1+x)**2, color='red')
            # if i*j!=1:
            #     axarr[i,j].plot(x, wigner_surmise(x), color='red', ls='-')
            #     axarr[i,j].plot(x, poisson(x), color='green', ls='-')
            # else:
            #     axarr[i,j].plot(x, wigner_surmise(x), color='red', ls='-', label='Wigner-Dyson')
            #     axarr[i,j].plot(x, poisson(x), color='green', ls='-', label='Poisson')


    #prepare the graphs
    for i, ax in enumerate(axarr.flatten()):
       # ax.set_ylim(0.001,0.004)
       ax.grid(which='both')
       ax.tick_params(axis='x', labelsize=fontsize[1])
       ax.tick_params(axis='y', labelsize=fontsize[1])
       ax.set_xlim(hist_bounds[0], hist_bounds[1])
       if i>1:

        ax.set_xlabel('$r$', fontsize=fontsize[1])
       # ax.xaxis.set_minor_locator(MultipleLocator(0.5))
       # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
       ax.tick_params(axis='x', labelsize=fontsize[1],pad=5,direction='out')
       ax.legend(loc='best',prop={'size':fontsize[1]},fontsize=fontsize[1],framealpha=0.5)

    for ax in axarr[:,0]:
        ax.set_ylabel('$p(r)$', fontsize=fontsize[-1])


    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    graphs_folder='./graphs_'+desc
    if not os.path.isdir(graphs_folder):
        os.makedirs(graphs_folder)


    # filename=desc+'_'+mod_name+'_'+'srule_engy_compare.pdf'
    # plt.show()
    #ax1 - W sweep
    #ax2 - H sweep 
    plt.savefig(graphs_folder+'/r_spectra_stats_{}_{}_{}.pdf'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu']))
    plt.show()

def plot_degeneracy(Wlist, sybr_list, sweep, desc, modstr, syspar, modpar,all_cases=True,save=False, importrange=1, new_ordering=True):

    titlestring='$L={}$'.format(syspar['size'])
    # if len(sizelist)>1:
    #     titlestring='scaling analysis'
    fig,ax=plt.subplots(1,1)


    
    ax.set_ylabel('$\log_{10}|\\bar{\mathrm{Deg.}}-1|$')
    
    modpar['WSYM']=0
    modpar['HSYM']=0
    if sweep=='both':
        typestring='spin and hole'
        labelstring='$W_\mathrm{{sym}}=H_\mathrm{{sym}}$'
        savestring='hole_spin_'
    elif sweep=='H':
        typestring='hole'
        labelstring='$H_\mathrm{{sym}}$'
        savestring='hole_'
    elif sweep=='W':
        typestring='spin'
        labelstring='$W_\mathrm{{sym}}$'
        savestring='spin_'

    # plt.xlabel(labelstring)
    avgdeg_list=[]
    for sybr in sybr_list:
        # r_mean=[]
        # Wplot=[]
        # syspar['size']=size
        # syspar['nu']=int((syspar['size']-syspar['ne'])/2)

        # for W in Wlist:

        if sweep=='both':
            modpar['WSYM']=sybr
            modpar['HSYM']=sybr
        elif sweep=='H':
            modpar['HSYM']=sybr
        elif sweep=='W':
            modpar['WSYM']=sybr 

        file=dataSet(desc, modstr, 'Eigvals', syspar, modpar, all_cases=True,importrange=importrange)
        file.load_dataSet(new_ordering=new_ordering)
        deg_arr=check_deg(file.data[0])
        # lvls, degs=deg_arr[1:,:]
        lvls,degs=get_level_deg(deg_arr)
        # xvals=np.arange(0,len(lvls),1)
        avgdeg_list.append(np.mean(degs))

    avgdeg_list=np.array(avgdeg_list)
    print(avgdeg_list-1)
    print(np.log10(np.abs(avgdeg_list-1)))
    # ax.scatter(sybr_list,np.log10(np.abs(avgdeg_list-1.)), label='{}={:.4f}'.format(labelstring,sybr))
    # ax.plot(sybr_list,np.log10(np.abs(avgdeg_list-1.)))
    ax.scatter(sybr_list, np.log10(np.abs(avgdeg_list-1)))
    ax.set_title('Avg. deg. for {} disorder sweep - tJ model, {}, $S^z=0$, $N_h={}$'.format(typestring, titlestring, file.syspar['ne']))
    # plt.text(1,0.2, '$\hat{h}_\mathrm{spin}=W\hat{S}^\mathrm{z}_1$ \n $\hat{h}_\mathrm{hole}= H\hat{n}_\mathrm{L}$ \n \\textbf{No disorder}')
    # plt.axhline(y=ergodic, color='red', ls='--')
    # plt.axhline(y=mbl, color='green',ls='--')
    # if len(syb_list)>1:
    #     plt.legend()
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    ax.legend()
    ax.set_xlabel(labelstring)
    # plt.semilogx(Wplot, r_mean, marker='o')
    graphs_folder='./graphs_'+desc
    if save:
        if not os.path.isdir(graphs_folder):
            os.makedirs(graphs_folder)


        plt.savefig(graphs_folder+'/{}degeneracy_check_{}_{}_{}.pdf'.format(savestring,file.syspar['size'], file.syspar['ne'],file.syspar['nu']))
    plt.show()


#spectral form factor, connected and unconnected, double plot 
def plot_sff(wlist,sweep,   sybr_list, desc, modstr, syspar, modpar, xlim=(10**(-3),3), ylim1=(10**(-3),3), ylim2=(10**(-3),3), new_ordering=False):
    fontsize=[17,28,32]

    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    sweep_names=['Spin','Hole']
    subfolder=''

    folder=get_values_folder(desc, modstr, syspar)
    print(folder)
    cwd=os.getcwd()
    
    # sweep_string=sweep_types[sweep_dict[sweep]]
    sweep_type=sweep_types[sweep_dict[sweep]]

    fig, axarr, fontsize=prepare_axarr(1, 2, fontsize=fontsize, sharey=False, sharex=False)

    # print(modstr)
    
    for sybr in sybr_list:


        modpar['HSYM']=sybr
        modpar['WSYM']=sybr

        for W in wlist:
            i=sweep_dict[sweep]
            modpar[sweep_types[i]]=W
            modpar[sweep_types[1-i]]=0

            if new_ordering:
                subfolder=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0]
                
                print('subfolder',subfolder)
                filename=get_results_file('SFF', modpar, folder+'/'+subfolder, all_cases=True, new_ordering=False)[0][0]
            else:
                filename=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0][0]


            print(filename)
            os.chdir(folder+'/'+subfolder)

            sff_data=np.load(filename)
            file=dataSet(desc, modstr, 'SFF', syspar, modpar)

            # N_data=np.mean(sff_data[1][-30:])
            N_data=get_num_states(syspar['size'], syspar['ne'], syspar['nu'])*0.5
            axarr[0].loglog(sff_data[0],sff_data[1]/N_data, label='{}={:g}'.format(sweep_type ,W))

            axarr[1].loglog(sff_data[0],sff_data[2]/N_data, label='{}={:g}'.format(sweep_type ,W))
        axarr[0].loglog(sff_data[0],K_GOE(sff_data[0]),label='$K_\\mathrm{GOE}(\\tau)$', ls='--', color='red' )
        axarr[1].loglog(sff_data[0],K_GOE(sff_data[0]),label='$K_\\mathrm{GOE}(\\tau)$', ls='--', color='red' )
        print('theory',np.max(2*sff_data[0]-sff_data[0]*np.log(1+2*sff_data[0])) )

        os.chdir(cwd)

    for ax in axarr.flatten():


        prepare_ax(ax, fontsize=fontsize, legend=False, grid=True)
        ax.axvline(x=1, color='black', ls='--', label='$\\tau_\mathrm{{H}}$')
        
        
        ax.set_xlabel('$\\tau$',fontsize=fontsize[1])
        ax.set_xlim((xlim[0],xlim[1]))
    axarr[1].legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5, ncol=2)
    axarr[0].set_ylim((ylim1[0],ylim1[1]))
    axarr[1].set_ylim((ylim2[0],ylim2[1]))      
    #   ax.grid()
    axarr[0].set_ylabel('$K(\\tau)$', fontsize=fontsize[1])
    axarr[0].set_title('Spektralni oblikovni faktor', fontsize=fontsize[1])
    axarr[1].set_ylabel('$K_c(\\tau)$', fontsize=fontsize[1])
    axarr[1].set_title('Povezani spektralni oblikovni faktor', fontsize=fontsize[1])
    fig.suptitle('$L={}$, $N_h={}$, $N_u={}$'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu']),fontsize=fontsize[-1])
    
    savename=sweep+'_sweep_sff_disorder'+'_{}_{}_{}.pdf'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu'])
    prepare_plt(savename=savename,plot_type='SFF', desc=desc, top=0.87, save=True, show=True)




def plot_sff_four(wlist,sweep,sybr_list,desc,  modstr, syspar, modpar, xlim1=(5*10**(-3),1.5),xlim2=(5*10**(-3),1.5),xlim3=(5*10**(-3),1.5),xlim4=(5*10**(-3),1.5),n_window=200, new_ordering=False):
    

    fontsize=[17,31,36]
    xlim=[xlim1,xlim2,xlim3,xlim4]
    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    sweep_names=['Spin','Hole']
    subfolder=''

    print('desc', desc)
    print('modstr', modstr)
    print('syspar',syspar)

    folder=get_values_folder(desc, modstr, syspar)
    print(folder)
    cwd=os.getcwd()
    
    # sweep_string=sweep_types[sweep_dict[sweep]]
    sweep_type=sweep_types[sweep_dict[sweep]]

    fig, axarr, fontsize=prepare_axarr(2, 2, fontsize=fontsize, sharey=True, sharex=True)

    axes=axarr.flatten()
    # print(modstr)
    
    for sybr in sybr_list:


        modpar['HSYM']=sybr
        modpar['WSYM']=sybr

        for j,W in enumerate(wlist):
            i=sweep_dict[sweep]
            modpar[sweep_types[i]]=W
            modpar[sweep_types[1-i]]=0

            if new_ordering:
                subfolder=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0]
                
                print('subfolder',subfolder)
                filename=get_results_file('SFF', modpar, folder+'/'+subfolder, all_cases=True, new_ordering=False)[0][0]
            else:
                filename=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0][0]


            print(filename)
            os.chdir(folder+'/'+subfolder)

            sff_data=np.load(filename)
            file=dataSet(desc, modstr, 'SFF', syspar, modpar)
            # N_data=np.mean(sff_data[1][-30:])
            N_data=get_num_states(syspar['size'], syspar['ne'], syspar['nu'])*0.5
            
            sff_plot=running_mean(sff_data[2],n_window)
            sff_tau=sff_data[0][int(n_window/2): -int(n_window/2 -1)]
            # axarr[0].loglog(sff_data[0],sff_data[1]/N_data, label='{}={:g}'.format(sweep_type ,W))
            theory=K_GOE(sff_tau)
            axes[j].loglog(sff_tau,np.abs(sff_plot/N_data - theory)/theory, label='{}={:g}'.format(sweep_type ,W))

        # axarr[0].loglog(sff_data[0],2*sff_data[0]-sff_data[0]*np.log(1+2*sff_data[0]),label='$2\\tilde{\\tau}-\\tilde{\\tau}\\log(1+2\\tilde{\\tau})$', ls='--', color='red' )
            # axes[j].loglog(sff_data[0],2*sff_data[0]-sff_data[0]*np.log(1+2*sff_data[0]),label='$2\\tilde{\\tau}-\\tilde{\\tau}\\log(1+2\\tilde{\\tau})$', ls='--', color='red' )
        os.chdir(cwd)

    for j,ax in enumerate(axarr.flatten()):


        prepare_ax(ax, fontsize=fontsize, legend=False, grid=True)
        ax.axvline(x=1, color='black', ls='--', label='$\\tau_\mathrm{{H}}$')
        
        if ((j==2) or (j==3)):
            ax.set_xlabel('$\\tau$',fontsize=fontsize[1])
        ax.set_xlim((xlim[j][0],xlim[j][1]))
    # axarr[1].legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5, ncol=2)
    # axarr[0].set_ylim((ylim1[0],ylim1[1]))
    # axarr[1].set_ylim((ylim2[0],ylim2[1]))      
    #   ax.grid()
    # axarr[0].set_ylabel('$g(\\tilde{\\tau})$', fontsize=fontsize[1])
        ax.set_title('${}={:.2f}$'.format(sweep,wlist[j]), fontsize=fontsize[1])
        if ((j==0) or (j==2)):
            ax.set_ylabel('$\\Delta_\\mathrm{rel} K_c(\\tau)$', fontsize=fontsize[1])
    # axarr[1].set_title('Povezani spektralni oblikovni faktor', fontsize=fontsize[1])
    fig.suptitle('Odstopanje $K_c(\\tau)$ od teoreti\\v{{c}}ne napovedi, $L={}$, $N_h={}$, $N_u={}$'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu']),fontsize=fontsize[-1])
    
    savename=sweep+'_sweep_sff_disorder'+'_{}_{}_{}_four_.pdf'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu'])
    prepare_plt(savename=savename,plot_type='SFF', desc=desc, top=0.9, save=True, show=True)


def plot_sff_four_new(wlist,hlist,sybr_list,desc,  modstr, syspar, modpar, xlim1=(5*10**(-3),1.5),xlim2=(5*10**(-3),1.5),xlim3=(5*10**(-3),1.5),xlim4=(5*10**(-3),1.5),n_window=4, new_ordering=False):
    
    disorder_list=np.array([wlist, hlist])
    fontsize=[24,31,36]
    xlim=[xlim1,xlim2,xlim3,xlim4]
    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    sweep_names=['Spin','Hole']
    subfolder=''

    print('desc', desc)
    print('modstr', modstr)
    print('syspar',syspar)

    folder=get_values_folder(desc, modstr, syspar)
    print(folder)
    cwd=os.getcwd()
    
    # sweep_string=sweep_types[sweep_dict[sweep]]
    # sweep_type=sweep_types[sweep_dict[sweep]]

    fig, axarr, fontsize=prepare_axarr(2, 2, fontsize=fontsize, sharey=True, sharex=True)

    axes=axarr.flatten()
    # print(modstr)
    
    for sybr in sybr_list:


        modpar['HSYM']=sybr
        modpar['WSYM']=sybr

        for i in range(2):

            sweep_type=sweep_types[i]
            # print('sweep_type:', sweep_type)
            for j,W in enumerate(disorder_list.T):
                i=sweep_dict[sweep_type]
                modpar[sweep_types[i]]=W[i]
                modpar[sweep_types[1-i]]=0

                if new_ordering:
                    subfolder=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0]
                    
                    print('subfolder',subfolder)
                    filename=get_results_file('SFF', modpar, folder+'/'+subfolder, all_cases=True, new_ordering=False)[0][0]
                else:
                    filename=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0][0]


                print(filename)
                os.chdir(folder+'/'+subfolder)

                sff_data=np.load(filename)
                file=dataSet(desc, modstr, 'SFF', syspar, modpar)
                # N_data=np.mean(sff_data[1][-30:])
                N_data=get_num_states(syspar['size'], syspar['ne'], syspar['nu'])*0.5
                sff_plot=sff_data[2]
                sff_tau=sff_data[0]
                # sff_plot=running_mean(sff_data[2],n_window)
                # sff_tau=sff_data[0][int(n_window/2): -int(n_window/2 -1)]
                # axarr[0].loglog(sff_data[0],sff_data[1]/N_data, label='{}={:g}'.format(sweep_type ,W))
                theory=K_GOE(sff_tau)
                axes[j].loglog(sff_tau,np.abs(sff_plot/N_data - theory)/theory, label='{}={:.2f}'.format(sweep_type ,W[i]))

            # axarr[0].loglog(sff_data[0],2*sff_data[0]-sff_data[0]*np.log(1+2*sff_data[0]),label='$2\\tilde{\\tau}-\\tilde{\\tau}\\log(1+2\\tilde{\\tau})$', ls='--', color='red' )
                # axes[j].loglog(sff_data[0],2*sff_data[0]-sff_data[0]*np.log(1+2*sff_data[0]),label='$2\\tilde{\\tau}-\\tilde{\\tau}\\log(1+2\\tilde{\\tau})$', ls='--', color='red' )
            os.chdir(cwd)

    for j,ax in enumerate(axarr.flatten()):


        prepare_ax(ax, fontsize=fontsize, legend=True, grid=True)
        ax.axvline(x=1, color='black', ls='--', label='$\\tau_\mathrm{{H}}$')
        ax.set_ylim(10**(-6), 100)
        if ((j==2) or (j==3)):
            ax.set_xlabel('$\\tau$',fontsize=fontsize[1])
        ax.set_xlim((xlim[j][0],xlim[j][1]))

        # ax.set_title('${}={:.2f}$'.format(sweep,wlist[j]), fontsize=fontsize[1])
        if ((j==0) or (j==2)):
            ax.set_ylabel('$\\Delta_\\mathrm{rel} K_c(\\tau)$', fontsize=fontsize[1])
    # axarr[1].set_title('Povezani spektralni oblikovni faktor', fontsize=fontsize[1])
    fig.suptitle('Odstopanje $K_c(\\tau)$ od teoreti\\v{{c}}ne napovedi, $L={}$, $N_h={}$, $N_u={}$'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu']),fontsize=fontsize[-1])
    
    savename='double_sweep_sff_disorder'+'_{}_{}_{}_four_.pdf'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu'])
    prepare_plt(savename=savename,plot_type='SFF', desc=desc, top=0.9, save=True, show=True)

def plot_sff_schematic(wlist=[1,6], xlim=(10**(-5),1.5*2*np.pi), ylim1=(10**(-3),3*10**3), ylim2=(10**(-3),3), new_ordering=True):
    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    fontsize=[22,30,32]
    syspar={'size':14, 'nu':7, 'ne':0, 'dim':1, 'base':'F'}
    
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    sweep='W'
    desc='no_doping_XXX_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    sybr_list=[0.5]
    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    sweep_names=['Spin','Hole']
    subfolder=''

    folder=get_values_folder(desc, modstr, syspar)
    print(folder)
    cwd=os.getcwd()
    
    # sweep_string=sweep_types[sweep_dict[sweep]]
    sweep_type=sweep_types[sweep_dict[sweep]]

    fig, axarr, fontsize=prepare_axarr(1, 2, fontsize=fontsize, sharey=False, sharex=False)

    # print(modstr)
    
    for sybr in sybr_list:


        modpar['HSYM']=sybr
        modpar['WSYM']=sybr

        for W in wlist:
            i=sweep_dict[sweep]
            modpar[sweep_types[i]]=W
            modpar[sweep_types[1-i]]=0

            if new_ordering:
                subfolder=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0]
                
                print('subfolder',subfolder)
                filename=get_results_file('SFF', modpar, folder+'/'+subfolder, all_cases=True, new_ordering=False)[0][0]
            else:
                filename=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0][0]


            print(filename)
            os.chdir(folder+'/'+subfolder)

            sff_data=np.load(filename)
            file=dataSet(desc, modstr, 'SFF', syspar, modpar)
            # sff_data*=2*np.pi
            # N_data=np.mean(sff_data[1][-30:])
            N_data=get_num_states(syspar['size'], syspar['ne'], syspar['nu'])*0.5
            axarr[0].loglog(sff_data[0],sff_data[1]/N_data, label='{}={:g}'.format(sweep_type ,W))
            # print(sff_data[1][0])
            # print('n_data:',N_data)
            # print('connected_one', sff_data[1][0]/N_data)
            axarr[1].loglog(sff_data[0],sff_data[2]/N_data, label='{}={:g}'.format(sweep_type ,W))
        axarr[0].loglog(sff_data[0],K_GOE(sff_data[0]),label='$K_\\mathrm{GOE}(\\tau)$', ls='--', color='red' )
        axarr[1].loglog(sff_data[0],K_GOE(sff_data[0]),label='$K_\\mathrm{GOE}(\\tau)$', ls='--', color='red' )
        os.chdir(cwd)

    for ax in axarr.flatten():


        prepare_ax(ax, fontsize=fontsize, legend=False, grid=True)
        ax.axvline(x=1, color='black', ls='--', label='$\\tau_\mathrm{{H}}$')
        
        
        ax.set_xlabel('$\\tau$',fontsize=fontsize[1])
        ax.set_xlim((xlim[0],xlim[1]))
    # axarr[0].set_xlim((10**(-3),3))
    # axarr[1].set_xlim((10**(-3),3))
    axarr[1].legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5, ncol=1)
    axarr[0].set_ylim((ylim1[0],ylim1[1]))
    axarr[1].set_ylim((ylim2[0],ylim2[1]))      
    #   ax.grid()
    axarr[0].set_ylabel('$K(\\tau)$', fontsize=fontsize[1])
    axarr[0].set_title('Spektralni oblikovni faktor', fontsize=fontsize[1])
    axarr[1].set_ylabel('$K_c(\\tau)$', fontsize=fontsize[1])
    axarr[1].set_title('Povezani spektralni oblikovni faktor', fontsize=fontsize[1])
    fig.suptitle('SFF v ergodi\\v{{c}}nem in MBL re\\v{{z}}imu, $L={}$, $N_h={}$, $N_u={}$'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu']),fontsize=fontsize[-1])
    
    savename='scheme_sff_disorder'+'_{}_{}_{}.pdf'.format(file.syspar['size'], file.syspar['ne'],file.syspar['nu'])
    prepare_plt(savename=savename,plot_type='SFF', desc=desc, top=0.87, save=True, show=True)   

#     plt.show()
def plot_sff_scaling(W, sizelist, sweep,sybr, desc, modstr, syspar, modpar,xlim=(0.005,3),ylim1=(0.001,3), ylim2=(0.001,2)):

    fontsize=[17,20,24]

    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    sweep_names=['Spin','Hole']
    
    # sweep_string=sweep_types[sweep_dict[sweep]]
    modpar['HSYM']=sybr
    modpar['WSYM']=sybr

    i=sweep_dict[sweep]
    modpar[sweep_types[i]]=W
    modpar[sweep_types[1-i]]=0

    folder=get_values_folder(desc, modstr, syspar)
    cwd=os.getcwd()

    fig, axarr=plt.subplots(1,2,figsize=(16,7))
    # print(modstr)

    for size in sizelist:
        syspar['size']=size
        syspar['nu']=int((syspar['size']-syspar['ne'])/2)
        folder=get_values_folder(desc, modstr, syspar)
        filename=get_results_file('SFF', modpar, folder, all_cases=True, new_ordering=new_ordering)[0][0]
        print(folder)
        print(filename)
        
        os.chdir(folder)
        # os.chdir(folder)

        sff_data=np.load(filename)
        file=dataSet(desc, modstr, 'SFF', syspar, modpar)
        # print(file.nsamples)
        # if file.data is not None:
            # file=dataSet(desc,modstr, 'Eigvals',syspar, modpar, all_cases=True, importrange=1)

        # sff, sff_connect, taulist = calc_sff_average(file.data, taulist=taulist)
        N_data=np.mean(sff_data[1][-30:])
        axarr[0].axvline(x=1, ls='--', color='red')
        axarr[1].axvline(x=1, ls='--', color='red')
        axarr[0].loglog(sff_data[0], sff_data[1]/N_data, label='L=%g'%size)
        axarr[1].loglog(sff_data[0], sff_data[2]/N_data, label='L=%g'%size)

    axarr[0].loglog(sff_data[0], 2*sff_data[0], label='$\\tau/\\tau_\mathrm{{H}}$')
    axarr[1].loglog(sff_data[0], 2*sff_data[0], label='$\\tau/\\tau_\mathrm{{H}}$')
    os.chdir(cwd)

    for ax in axarr.flatten():
        ax.tick_params(axis='x', labelsize=fontsize[1],pad=5,direction='out')
        # ax.axhline(y=mbl, ls='--', color='green')
        # ax.axhline(y=ergodic, ls='--', color='red')
        ax.legend(loc='best',prop={'size':fontsize[1]},fontsize=fontsize[1],framealpha=0.5, ncol=2)
        
        ax.set_xlabel('$\\tau$',fontsize=fontsize[1])
        ax.set_xlim((xlim[0],xlim[1]))

        ax.tick_params(axis='x', labelsize=fontsize[1])
        # ax.set_title('{} disorder'.format(sweep_names[i]), fontsize=fontsize[-1])
        ax.tick_params(axis='y', labelsize=fontsize[1])
        # ax.xaxis.set_major_locator(MultipleLocator(2))
    axarr[0].set_ylim((ylim1[0],ylim1[1]))
    axarr[1].set_ylim((ylim2[0],ylim2[1]))      
    axarr[0].grid()
    axarr[0].set_ylabel('$g(\\tau)$', fontsize=fontsize[1])
    axarr[0].set_title('Spektralni oblikovni faktor', fontsize=fontsize[1])
    axarr[1].grid()
    axarr[1].set_ylabel('$g_c(\\tau)$', fontsize=fontsize[1])
    axarr[1].set_title('Povezani spektralni oblikovni faktor', fontsize=fontsize[1])
    fig.suptitle(' $W={:.3f}$, $H={:.3f}$, $N_h={}$'.format(file.modpar['W'], file.modpar['H'],file.syspar['ne']),fontsize=fontsize[-1])
    plt.tight_layout()
    plt.subplots_adjust(top=0.89)
    graphs_folder='./graphs_'+desc+'/sff'
    if not os.path.isdir(graphs_folder):
        os.makedirs(graphs_folder)

    plt.savefig(graphs_folder+'/' +'sff_scaling'+'W_{:.5f}_H_{:.5f}_ne{}.pdf'.format(file.modpar['W'], file.modpar['H'],file.syspar['ne']))
    

    plt.show()


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#entanglement entropy plots
#default rules dict:
rules={'ne': lambda x,y: {'ne': 0 },
             'nu': lambda x,y:{'nu': int((y['size']-1)/2)},
              }

def plot_ent_entropy(sizelist,  desc, modstr, syspar, modpar,rules=rules, nrows=1, ncols=1, titlestring='',save='False', savename=''):


    """
    INPUT: 

    rules - a dictionary of rules on how to assign values to system parameters. How to write rules,
    an example code snippet:

    # sizelist=[5,7,9,11,13]
    # for size in sizelist:
    #     seznam={'ne':5, 'size':size, 'nu':3}
    #     print('before change',seznam)

    #     rules={'ne': lambda x,y: {'ne': 0 },
    #            'nu': lambda x,y:{'nu': int((y['size']-1)/2)},

    #     }

    #     for key in rules:

    #         seznam.update(rules[key](seznam[key], seznam))

    #         print('during change', seznam)

        

    """
    quantity='Ent_entropy'
    fig, axarr, fontsize=prepare_axarr(nrows, ncols, fontsize=[18,21,23])

    fig.suptitle(titlestring, fontsize=fontsize[-1])
    c1=0.7305
    for i,ax in enumerate([axarr]):
        

        if i==0:
            x=np.linspace(0,1,1000)
            cft=np.log(np.sin(np.pi*x))/3. + c1
            ax.plot(x, cft, label='$S_\mathrm{{CFT}}=(1/3)\\log(\\sin(\pi L_n/L))$')

        for size in sizelist: 

            syspar['size']=size

            for key in rules:

                syspar.update(rules[key](syspar[key], syspar))

            print(syspar)
            subsystems=np.arange(1,np.ceil(size/2.)+1,1)
            file=dataSet(desc, modstr,quantity,syspar, modpar)
            file.load_dataSet()
            ave=file.mean_data()
            ave=ave.values

            ax.scatter(subsystems/size, ave[1:, 0]- np.log(size/np.pi)/3., label='$L={}$'.format(size))
            # print(ave[1:,0])
            # print(subsystems)


        prepare_ax(ax)
        ax.set_ylim(0,0.8)
        ax.set_xlim(0.,0.7)
        ax.set_ylabel( '$S_n(L) - \\log(L\pi)/3$'  , fontsize=fontsize[1])
        ax.set_xlabel('$L_n/L$', fontsize=fontsize[1])

    prepare_plt(save=save,desc=desc, plot_type='Entanglement_entropy', savename=savename)


# def plot_ent_entropy_density_plot():
#     #entanglement entropy density plot for half-partitions
#     #first: load data, then coarse-grain the energy interval and average the energy values

#     """
#     What the code does: 

#     For each set of system and model parameters, the entanglement entropy data for the chosen 
#     partitioning are loaded. For each entropy dataset, the minimum and maximum energy are obtained, 
#     the spectrum is then normalized.then
#     epsilon=(E-E0)/(Emax-E0)

#     The epsilon values interval is divided into nbins subintervals; en. entropy values are averaged
#     upon them and then also over disorder realizations.



#     """


#---------------------------------------------------------------------------------------------------------------------------------
#energy and energy density plots

def plot_energies(wlist=[0,4,8,12],desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, save=False, all_cases=True ):
    quantity='Eigvals'
    titlestring='Energijski spekter pri razli\\v{{c}}nih vrednostih nereda, $L={}$, $N_h={}$, $N_u={}$, $t={:.2f}$, $J={:.2f}$'
    fig, axarr, fontsize=prepare_axarr(1, 2, fontsize=[18,21,23], sharey=False)
    titlestring=titlestring.format(syspar['size'],syspar['ne'], syspar['nu'], modpar['T'], modpar['J'] )
    fig.suptitle(titlestring, fontsize=fontsize[-1])

    sweeplist=['W','H']
    titlelist=['Spinski nered', 'Potencialni nered']

    for i,ax in enumerate(axarr.flatten()):
        
        ax.set_title(titlelist[i], fontsize=fontsize[-1])

        # if i==0:
        #     x=np.linspace(0,1,1000)
        #     cft=np.log(np.sin(np.pi*x))/3. + c1
        #     ax.plot(x, cft, label='$S_\mathrm{{CFT}}=(1/3)\\log(\\sin(\pi L_n/L))$')

        for W in wlist: 

            modpar[sweeplist[i]]=W
            modpar[sweeplist[1-i]]=0


            file=dataSet(desc, modstr,quantity,syspar, modpar)
            file.load_dataSet()
            ave=file.mean_data()
            ave=ave.values
            print(ave)
            indx=np.arange(1,len(ave)+1,1)

            ax.plot(indx,ave, label='${}={:.2f}$'.format(sweeplist[i],W))
            # print(ave[1:,0])
            # print(subsystems)


        prepare_ax(ax)
        # ax.set_ylim(0,0.8)
        # ax.set_xlim(0.,0.7)
        ax.set_ylabel( '$E$'  , fontsize=fontsize[1])
        ax.set_xlabel('Indeks stanja', fontsize=fontsize[1])

    savename='Eigenstates_spin_hole_disorder_{}_{}_{}.pdf'.format(syspar['size'], syspar['ne'],syspar['nu'])

    prepare_plt(save=True,desc=desc, plot_type='Eigenstates', savename=savename)

def plot_DOS(wlist=[0,4,8,12],desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, save=False, all_cases=True, hist_bins=100):
    quantity='Eigvals'
    titlestring='Gostota stanj pri razli\\v{{c}}nih vrednostih nereda, $L={}$, $N_h={}$, $N_u={}$, $t={:.2f}$, $J={:.2f}$'
    fig, axarr, fontsize=prepare_axarr(1, 2, fontsize=[20,24,26], sharey=False, sharex=False)
    titlestring=titlestring.format(syspar['size'],syspar['ne'], syspar['nu'], modpar['T'], modpar['J'] )
    fig.suptitle(titlestring, fontsize=fontsize[-1])

    sweeplist=['W','H']
    titlelist=['Spinski nered', 'Potencialni nered']

    for i,ax in enumerate(axarr.flatten()):
        
        ax.set_title(titlelist[i], fontsize=fontsize[-1])

        # if i==0:
        #     x=np.linspace(0,1,1000)
        #     cft=np.log(np.sin(np.pi*x))/3. + c1
        #     ax.plot(x, cft, label='$S_\mathrm{{CFT}}=(1/3)\\log(\\sin(\pi L_n/L))$')

        for W in wlist: 

            modpar[sweeplist[i]]=W
            modpar[sweeplist[1-i]]=0


            file=dataSet(desc, modstr,quantity,syspar, modpar)
            file.load_dataSet()
            ave=file.mean_data()
            ave=ave.values
            print(ave)
            # indx=np.arange(1,len(ave)+1,1)
            bounds=(np.min(ave), np.max(ave))
            bins=int((bounds[1]-bounds[0])/0.1)
            hist,edges=np.histogram(ave, bins=bins,density=True)
            
            ax.step(edges[:-1], hist,label='${}={:.2f}$'.format(sweeplist[i],W))
            # print(ave[1:,0])
            # print(subsystems)


        prepare_ax(ax, fontsize=fontsize)
        # ax.set_ylim(0,0.8)
        # ax.set_xlim(0.,0.7)
        ax.set_ylabel( 'Normalizirana gostota stanj'  , fontsize=fontsize[1])
        ax.set_xlabel('E', fontsize=fontsize[1])

    savename='DOS_spin_hole_disorder_{}_{}_{}.pdf'.format(syspar['size'], syspar['ne'],syspar['nu'])

    prepare_plt(save=True,desc=desc, plot_type='Eigenstates', savename=savename)


#ent_entropy density plot   
def bin_stats(group, bins=10):


    if type(bins)==int:
        vals=np.zeros((bins, len(group.values.T)))
    else:
        vals=np.zeros((len(bins)-1, len(group.values.T)))

    for i, col in enumerate(group.values.T[1:]):

        bin_means, bin_edges, bin_number=stats.binned_statistic(group.values[:,0], col,statistic='mean',bins=bins)
        vals[:,i+1]=bin_means

    vals[:,0]=0.5*(bin_edges[1:] + bin_edges[:-1])

    
    return pd.DataFrame(vals)


def renorm_entropy(df, bins=10):

    #get max min values  object
    # print(data)

    #get energy values
    vals=df.xs(0, level=1, axis=1).values
    #get minimum and maximum energy
    mins=np.min(vals, axis=0)
    maxs=np.max(vals, axis=0)
    idx=pd.IndexSlice
    #renoramalize energies
    vals=(vals - mins[None, :])/(maxs-mins)[None, :]
    #reset energy values in df
    df.loc[:,idx[:,0]]=vals

    df=df.groupby(axis=1, level=0).apply(bin_stats, bins)

    df=df.groupby(axis=1, level=1).mean()
    #perform bin statistics
    bin_means, bin_edges, bin_number=stats.binned_statistic(df.iloc[:,0], df.iloc[:,-1], statistic='mean',bins=bins)
    return bin_means, bin_edges, bin_number

def renorm_entropy_new(df, bins=10, all=False):

    #get max min values  object
    # print(data)

    #get energy values
    vals=df.xs(0, level=1, axis=1).values
    #get minimum and maximum energy
    # print(vals)
    mins=np.min(vals)
    # mins=-50
    print('mins',mins)
    maxs=np.max(vals)
    # maxs=50
    print('maxs',maxs)
    idx=pd.IndexSlice
    #renoramalize energies
    vals=(vals - mins)/(maxs-mins)
    #reset energy values in df
    df.loc[:,idx[:,0]]=vals
    # print(df.loc[:, idx[:,1]])
    #first perform statistics on a single disorder realization
    df=df.groupby(axis=1, level=0).apply(bin_stats, bins)

    #then take the mean over disorder realizations
    df=df.groupby(axis=1, level=1).mean()



    if not all:

        return df.iloc[:,-1].values, df.iloc[:,0].values, []
    #     bin_means, bin_edges, bin_number=stats.binned_statistic(df.iloc[:,0], df.iloc[:,-1], statistic='mean',bins=bins)
    else:

        return df.iloc[:,1:].values.T, df.iloc[:,0].values
    #     return bin_means, bin_edges, bin_number
    # else: #used for plotting the entropy scaling
    #     # bins=[0,0.4,0.6,1]
    #     scaling_means=[]
        
    #     for i,column in enumerate(df.columns[1:]):
    #         # print(column)

    #         bin_means, bin_edges, bin_number=stats.binned_statistic(df.iloc[:,0], df.iloc[:,i+1], statistic='mean',bins=bins)
    #         scaling_means.append(bin_means)
    #         # print('during_stats:',bin_means)

    #     print('after_bin_stats:',np.array(scaling_means))
    #     return np.array(scaling_means), bin_edges

def entropy_density_plot(wlist=[0,2,4,6],sweep='W', desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, importrange=1, bins=20,new_ordering=False, heisenberg=False, save_data=True, load_data=True):


    filesave='{}_{}_sweep_{}_{}_{}.npy'.format(desc,syspar['size'],sweep, syspar['ne'], syspar['nu'] )
    load_name=load_data_file(filesave,'ent_dens',desc=desc,load=False)

    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    
    sweep_string=sweep_types[sweep_dict[sweep]]

    num=len(wlist)
    # rvals=np.zeros((num,num),dtype=np.float)

    w_min=wlist[0];w_max=wlist[-1]; dy=wlist[1]-wlist[0]

    h_min=0
    h_max=1
    dh=1./bins

    w_vals,h_vals=np.meshgrid(np.linspace(w_min,w_max+dy,num+1)-dy/2,np.linspace(h_min,h_max+dh,bins+1)-dh/2)

    # print(w_vals)
    
    fig, axarr, fontsize=prepare_axarr(1,1, fontsize=[17,20,24])


    #if the file is not to be loaded or if it does not exist:
    plot_arr=[]
    if ((not load_data) or not os.path.isfile(load_name)):
        for i,W in enumerate(wlist):

            modpar[sweep_string]=W

            # modpar['H']=H
            file=dataSet(desc, modstr, 'Ent_entropy', syspar, modpar)

            file.load_dataSet(new_ordering=new_ordering)
            if file.data.values is not None:
                df=file.data

                # print(df.values)
                # print(df.values)
                # bins_, edges_, numbers_=renorm_entropy(df,bins)
                bins_,edges_,numbers_=renorm_entropy_new(df,bins)
                # print(bins_, edges_)
                plot_arr.append(bins_)

        plot_arr=np.array(plot_arr).T

    else:
        plot_arr=load_data_file(filesave,'ent_dens', desc=desc)

    if save_data:
        
        save_data_file(plot_arr,filesave,'ent_dens',desc=desc)

    # print(np.array(plot_arr).T)
    # print(rvals)

    if heisenberg:
        rescale_ent=np.floor(syspar['size']/2.)*np.log(2)
        subsize_string='L/2'
    else:
        rescale_ent=np.floor(syspar['size']/2.)*np.log(3)
        subsize_string='(L-1)/2'

    renorm=syspar['size']

    axarr.set_xticks(np.arange(w_min, w_max+dy, 2))
    axarr.set_yticks(np.arange(h_min, h_max+dh, 0.1))

    axarr.axis([w_vals.min(), w_vals.max(), h_vals.min(), h_vals.max()])

    plot_arr*=1/rescale_ent
    middle=(0.5*(np.max(plot_arr)+ np.min(plot_arr)))
    change_list=np.where( (plot_arr>middle - 0.01 ) & (plot_arr<middle+0.01))

    w_points=w_vals[0][change_list[1]]
    h_points=h_vals[:,0][change_list[0]]
    dw=0.5*(w_vals[0][1]-w_vals[0][0])
    im=axarr.pcolormesh(w_vals, h_vals, plot_arr, linewidth=0, rasterized=True)
    # print('w_vals_edge', w_vals[change_list[0]][0])
    axarr.scatter(w_points + dw , h_points - 0.5*dh, color='red')

    cbar=plt.colorbar(im,fraction=0.08,orientation='horizontal')
    cbar.ax.invert_xaxis()
    cbar.ax.tick_params(labelsize=fontsize[-1],direction='out')
    if heisenberg:
        cbar.ax.set_xlabel('$\\frac{S_\\mathrm{A}}{L_\\mathrm{A}\\log(2)}$',fontsize=fontsize[1])
    else:
        cbar.ax.set_xlabel('$\\frac{S_\\mathrm{A}}{L_\\mathrm{A}\\log(3)}$',fontsize=fontsize[1])
    prepare_ax(axarr, fontsize=fontsize, legend=False, grid=False)
    axarr.set_xlabel('${}$'.format(sweep_string), fontsize=fontsize[1])
    axarr.set_ylabel('$\\varepsilon$', fontsize=fontsize[1])
    axarr.set_title('$S_\\mathrm{{A}}$, $L={}$, $N_h={}$, $N_u={}, L_\\mathrm{{A}}={}$'.format(syspar['size'], syspar['ne'], syspar['nu'], subsize_string), fontsize=fontsize[-1])
    savename=sweep_string+'_sweep_ent_entropy_density_plot_{}_{}_{}.pdf'.format(syspar['size'],syspar['ne'],syspar['nu'])
    # plt.show()
    modpar['H']=0.0
    modpar['W']=0.0
    
    prepare_plt(plot_type='ent_entropy',save=True, show=True, savename=savename, desc=desc)
    # plt.show()


def entropy_density_plot_cuts(wlist=[0,2,4,6],sweep='W', desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, importrange=1, bins=20,new_ordering=False, heisenberg=False,cuts=[0], save_data=True, load_data=True):
    

    filesave='{}_{}_sweep_{}_{}_{}.npy'.format(desc,syspar['size'],sweep, syspar['ne'], syspar['nu'] )
    load_name=load_data_file(filesave,'ent_dens',desc=desc,load=False)
    filesave_cut='{}_{}_sweep_{}_{}_{}_zero_cut.npy'.format(desc,syspar['size'],sweep, syspar['ne'], syspar['nu'] )
    load_name_cut=load_data_file(filesave_cut,'ent_dens',desc=desc,load=False)

    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    
    sweep_string=sweep_types[sweep_dict[sweep]]

    num=len(wlist)
    # rvals=np.zeros((num,num),dtype=np.float)

    w_min=wlist[0];w_max=wlist[-1]; dy=wlist[1]-wlist[0]

    h_min=0
    h_max=1
    dh=1./bins

    w_vals,h_vals=np.meshgrid(np.linspace(w_min,w_max+dy,num+1)-dy/2,np.linspace(h_min,h_max+dh,bins+1)-dh/2)

    # print(w_vals)
    
    fig, axarr, fontsize=prepare_axarr(1,2, fontsize=[24,26,30], sharey=False, sharex=False)


    #if the file is not to be loaded or if it does not exist:
 
    plot_arr=[]

    if ((not load_data) or not os.path.isfile(load_name)):
        for i,W in enumerate(wlist):

            modpar[sweep_string]=W

            # modpar['H']=H
            file=dataSet(desc, modstr, 'Ent_entropy', syspar, modpar)

            file.load_dataSet(new_ordering=new_ordering)
            if file.data.values is not None:
                df=file.data
                if W in cuts:
                    ave=file.mean_data()
                    ave=ave.values


                bins_,edges_,numbers_=renorm_entropy_new(df,bins)
                # print(bins_, edges_)
                plot_arr.append(bins_)

        plot_arr=np.array(plot_arr).T

    else:
        ave=load_data_file(filesave_cut,'ent_dens', desc=desc)
        plot_arr=load_data_file(filesave,'ent_dens', desc=desc)

    if save_data:
        save_data_file(ave,filesave_cut,'ent_dens',desc=desc)
        save_data_file(plot_arr,filesave,'ent_dens',desc=desc)

    if heisenberg:
        rescale_ent=np.floor(syspar['size']/2.)*np.log(2)
        subsize_string='L/2'
        log_string='\\log(2)'
    else:
        rescale_ent=np.floor(syspar['size']/2.)*np.log(3)
        subsize_string='(L-1)/2'
        log_string='\\log(3)'


    for ax in axarr.flatten():
        prepare_ax(ax, fontsize=fontsize, legend=False, grid=False)



    ax1=axarr[0]
    # ax1.set_xticks(np.arange(w_min, w_max+dy, 2))
    ax1.set_xticks(np.arange(w_min, w_max+dy, 1))
    ax1.set_yticks(np.arange(h_min, h_max+dh, 0.1))


    ax1.axis([w_vals.min(), w_vals.max(), h_vals.min(), h_vals.max()])
    # ax1.axis([w_vals.min(), 6, h_vals.min(), h_vals.max()])
    # axarr.axis([w_vals.min(), w_vals.max(), h_vals.min()+0.1, h_vals.max()-0.1])
    plot_arr*=1/rescale_ent
    middle=(0.5*(np.max(plot_arr)+ np.min(plot_arr)))
    change_list=np.where( (plot_arr>middle - 0.01 ) & (plot_arr<middle+0.01))
    # print('change_list',change_list)
    # print('hvals', h_vals[:,0])
    # print('wvals', w_vals[0])
    w_points=w_vals[0][change_list[1]]
    # print('success')
    h_points=h_vals[:,0][change_list[0]]
    # print('success')
    # plot_arr[change_list]=np.NaN
    dw=0.5*(w_vals[0][1]-w_vals[0][0])
    im=ax1.pcolormesh(w_vals, h_vals, plot_arr, linewidth=0, rasterized=True)
    # print('w_vals_edge', w_vals[change_list[0]][0])
    ax1.scatter(w_points + dw , h_points - 0.5*dh, color='red')
    # print('change_list',change_list[0])
    cbar=plt.colorbar(im,fraction=0.08,orientation='horizontal')
    # cmap.set_bad(color = 'k', alpha = 1.)
    cbar.ax.invert_xaxis()
    cbar.ax.tick_params(labelsize=fontsize[-1],direction='out')
    if heisenberg:
        cbar.ax.set_xlabel('$\\frac{S_\\mathrm{A}}{L_\\mathrm{A}\\log(2)}$',fontsize=fontsize[1])
    else:
        cbar.ax.set_xlabel('$\\frac{S_\\mathrm{A}}{L_\\mathrm{A}\\log(3)}$',fontsize=fontsize[1])
    
    ax1.set_xlabel('${}$'.format(sweep_string), fontsize=fontsize[1])
    ax1.set_ylabel('$\\varepsilon$', fontsize=fontsize[1])
    ax1.set_title('$L={}$, $N_h={}$, $N_u={}'.format(syspar['size'], syspar['ne'], syspar['nu']), fontsize=fontsize[-1])
    
    ax2=axarr[1]
    ax2.scatter(ave.T[0], ave.T[-1]/rescale_ent)
    ax2.set_xlabel('$E$', fontsize=fontsize[1])
    ax2.set_ylabel('$S_\\mathrm{{A}}/L_\\mathrm{{A}}{}$'.format(log_string), fontsize=fontsize[1])
    ax2.set_title('$W=H=0.0$',fontsize=fontsize[1])
    ax2.grid()
    fig.suptitle('$S_\\mathrm{{A}}$, $L_\\mathrm{{A}}={}$'.format(subsize_string), fontsize=fontsize[-1])
    savename=sweep_string+'_sweep_ent_entropy_density_plot_{}_{}_{}_cuts.pdf'.format(syspar['size'],syspar['ne'],syspar['nu'])
    # plt.show()
    modpar['H']=0.0
    modpar['W']=0.0

    prepare_plt(plot_type='ent_entropy',save=True,top=0.87, show=True, savename=savename, desc=desc)
    # plt.show()

def plot_ent_entropy_scaling_subsystems(wlist=[0,6],sweep='W', desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, importrange=1,new_ordering=False, heisenberg=True,cuts=[0], save_data=True, load_data=False):
    
    filesave='subsystems_{}_{}_sweep_{}_{}_{}.npy'.format(desc,syspar['size'],sweep, syspar['ne'], syspar['nu'] )
    load_name=load_data_file(filesave,'ent_scaling',desc=desc,load=False)

    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    
    sweep_string=sweep_types[sweep_dict[sweep]]

    num=len(wlist)
    # rvals=np.zeros((num,num),dtype=np.float)

    # w_min=wlist[0];w_max=wlist[-1]; dy=wlist[1]-wlist[0]


    bins=[0,0.45,0.55,1]
    # h_min=0
    # h_max=1
    # dh=1./bins

    # w_vals,h_vals=np.meshgrid(np.linspace(w_min,w_max+dy,num+1)-dy/2,np.linspace(h_min,h_max+dh,bins+1)-dh/2)

    # print(w_vals)
    
    fig, axarr, fontsize=prepare_axarr(1,1, fontsize=[20,22,24], figsize=(10,7))


    #if the file is not to be loaded or if it does not exist:
    ent_arr=[]
    if ((not load_data) or not os.path.isfile(load_name)):
        for i,W in enumerate(wlist):

            modpar[sweep_string]=W

            # modpar['H']=H
            file=dataSet(desc, modstr, 'Ent_entropy', syspar, modpar)

            file.load_dataSet(new_ordering=new_ordering)
            if file.data.values is not None:
                df=file.data

                # print(df.values)
                # print(df.values)
                # bins_, edges_, numbers_=renorm_entropy(df,bins)
                vals,edges_,=renorm_entropy_new(df,bins, all=True)
                # print('plot_info: plot_arr',plot_arr)
                # print(bins_, edges_)
                states=vals[:,1]
                subs=np.arange(1,len(states)+1,1)
                if i==0:
                    ent_arr.append(subs)
                
                ent_arr.append(states)
                
                # axarr.scatter(subs,states/subs)

        plot_arr=np.array(ent_arr)
        print(plot_arr)
        # plot_arr=np.array(plot_arr).T

    else:
        plot_arr=load_data_file(filesave,'ent_scaling', desc=desc)

    if save_data:
        
        save_data_file(plot_arr,filesave,'ent_scaling',desc=desc)

    # print(np.array(plot_arr).T)
    # print(rvals)

    if heisenberg:
        rescale_ent=np.floor(syspar['size']/2.)*np.log(2)
        subsize_string='L/2'
    else:
        rescale_ent=np.floor(syspar['size']/2.)*np.log(3)
        subsize_string='(L-1)/2'

    renorm=syspar['size']

    for i,el in enumerate(plot_arr[1:]):

        axarr.scatter(plot_arr[0], el, label='${}={:.2f}$'.format(sweep_string, wlist[i]))

    # axarr.plot(np.arange(1,8,1),plot_arr[:,1])
    # axarr.set_xticks(np.arange(w_min, w_max+dy, 2))
    # axarr.set_yticks(np.arange(h_min, h_max+dh, 0.1))
    axarr.legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5)
    # axarr.axis([w_vals.min(), w_vals.max(), h_vals.min(), h_vals.max()])
    # axarr.axis([w_vals.min(), w_vals.max(), h_vals.min()+0.1, h_vals.max()-0.1])
    # im=axarr.pcolormesh(w_vals, h_vals, plot_arr/rescale_ent, linewidth=0, rasterized=True)
    # cbar=plt.colorbar(im,fraction=0.08,orientation='horizontal')
    # cbar.ax.invert_xaxis()
    # # cbar.ax.tick_params(labelsize=fontsize[-1],direction='out')
    # if heisenberg:
    #     cbar.ax.set_xlabel('$\\frac{S_\\mathrm{A}}{L_\\mathrm{A}\\log(2)}$',fontsize=fontsize[1])
    # else:
    #     cbar.ax.set_xlabel('$\\frac{S_\\mathrm{A}}{L_\\mathrm{A}\\log(3)}$',fontsize=fontsize[1])
    prepare_ax(axarr, fontsize=fontsize, legend=False, grid=True)
    axarr.set_xlabel('$L_\\mathrm{A}$', fontsize=fontsize[1])
    axarr.set_ylabel('$S_\\mathrm{A}$', fontsize=fontsize[1])
    axarr.set_title('$S_\\mathrm{{A}}$ v odvisnosti od $L_\\mathrm{{A}}$, $L={}$, $N_h={}$, $N_u={}$'.format(syspar['size'], syspar['ne'], syspar['nu']), fontsize=fontsize[-1])
    savename=sweep_string+'_sweep_ent_entropy_scaling_{}_{}_{}.pdf'.format(syspar['size'],syspar['ne'],syspar['nu'])
    # plt.show()
    modpar['H']=0.0
    modpar['W']=0.0
    
    prepare_plt(plot_type='ent_entropy',save=True, show=True, savename=savename, desc=desc)

def plot_ent_entropy_scaling_systems(wlist=[0,8],sizelist=[6,8,10,12,14], sweep='W', desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, importrange=1,new_ordering=False, heisenberg=True,cuts=[0], save_data=True, load_data=False):
    
    filesave='systems_{}_{}_sweep_{}_{}_{}.npy'.format(desc,syspar['size'],sweep, syspar['ne'], syspar['nu'] )
    load_name=load_data_file(filesave,'ent_scaling',desc=desc,load=False)

    sweep_dict={'W':0, 'H':1}
    sweep_types=['W','H']
    
    sweep_string=sweep_types[sweep_dict[sweep]]

    num=len(wlist)


    bins=[0,0.45,0.55,1]

    
    fig, axarr, fontsize=prepare_axarr(1,1, fontsize=[20,22,24],figsize=(10,7))


    #if the file is not to be loaded or if it does not exist:
    ent_arr=[]
    if ((not load_data) or not os.path.isfile(load_name)):
        for i,W in enumerate(wlist):

            modpar[sweep_string]=W


            entropy_list=np.zeros(len(sizelist))
            for j,size in enumerate(sizelist):
                syspar['size']=size
                syspar['nu']=int((size -syspar['ne'])/2)
                # modpar['H']=H
                file=dataSet(desc, modstr, 'Ent_entropy', syspar, modpar)

                file.load_dataSet(new_ordering=new_ordering)
                if file.data.values is not None:
                    df=file.data
                    vals,edges_,=renorm_entropy_new(df,bins, all=True)
                    print('edges_:',edges_)
                    print('vals:', vals)
                    states=vals[:,1][-1]
                    print('states:',states)
                    entropy_list[j]=states
                
                # axarr.scatter(subs,states/subs)

            ent_arr.append(entropy_list)
        plot_arr=np.array(ent_arr)
        print(plot_arr)
        # plot_arr=np.array(plot_arr).T

    else:
        plot_arr=load_data_file(filesave,'ent_scaling', desc=desc)

    if save_data:
        
        save_data_file(plot_arr,filesave,'ent_scaling',desc=desc)

    if heisenberg:
        rescale_ent=np.floor(syspar['size']/2.)*np.log(2)
        subsize_string='L/2'
    else:
        rescale_ent=np.floor(syspar['size']/2.)*np.log(3)
        subsize_string='(L-1)/2'

    renorm=syspar['size']

    for i,el in enumerate(plot_arr):
        sizes=np.array(sizelist)
        axarr.scatter(sizes, 2*el/sizes, label='${}={:.2f}$'.format(sweep_string, wlist[i]))


    axarr.legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5)

    prepare_ax(axarr, fontsize=fontsize, legend=False, grid=True)
    axarr.set_xlabel('$L$', fontsize=fontsize[1])
    axarr.set_ylabel('$S_\\mathrm{A}/L_\\mathrm{A}$', fontsize=fontsize[1])
    axarr.set_title('$S_\\mathrm{{A}}/L_\\mathrm{{A}}$ v odvisnosti od $L$,\\\\ $L_\\mathrm{{A}}=L/2$, $N_h={}$, $S^z=0$'.format(syspar['ne']), fontsize=fontsize[-1])
    savename=sweep_string+'_sweep_ent_entropy_scaling_systems_{}_{}_{}.pdf'.format(syspar['size'],syspar['ne'],syspar['nu'])
    # plt.show()
    modpar['H']=0.0
    modpar['W']=0.0
    
    prepare_plt(plot_type='ent_entropy',save=True, show=True, savename=savename, desc=desc)

def plot_ent_entropy_eigenstates(desc, modstr, syspar, modpar, nrows=1, ncols=2, titlestring='',save='False', savename='', new_ordering=True, heisenberg=False):


    """
    INPUT: 

    rules - a dictionary of rules on how to assign values to system parameters. How to write rules,
    an example code snippet:

    # sizelist=[5,7,9,11,13]
    # for size in sizelist:
    #     seznam={'ne':5, 'size':size, 'nu':3}
    #     print('before change',seznam)

    #     rules={'ne': lambda x,y: {'ne': 0 },
    #            'nu': lambda x,y:{'nu': int((y['size']-1)/2)},

    #     }

    #     for key in rules:

    #         seznam.update(rules[key](seznam[key], seznam))

    #         print('during change', seznam)

        

    """
    quantity='Ent_entropy'
    fig, axarr, fontsize=prepare_axarr(nrows=nrows, ncols=ncols, sharey=False, sharex=False,fontsize=[18,21,23])

    fig.suptitle(titlestring, fontsize=fontsize[-1])
    

    file=dataSet(desc, modstr,quantity,syspar, modpar)
    file.load_dataSet(new_ordering=new_ordering)

    ave=file.mean_data()
    ave=ave.values

    # ax.scatter(ave.T[0],ave.T[-1])
    eng_arr=check_deg(ave.T[0])
    degs=get_level_deg(eng_arr, eps=1e-013)
    # print(degs[1])
    degs_=np.where(degs[1]==1)[0]

    
    nondeg_ener=np.take(ave.T[0], degs_)
    
    nondeg_entro=np.take(ave.T[-1], degs_)
    
    # print(len(ave.T[0]))
    if heisenberg:
        renorm=np.floor(syspar['size']/2.)*np.log(2)
    else:
        renorm=np.floor(syspar['size']/2.)*3

    
    
    
    axarr[0].scatter(ave.T[0],ave.T[-1]/renorm, color='blue')
    axarr[0].scatter(nondeg_ener, nondeg_entro/renorm, color='red')


    axarr[1].scatter(degs[0],degs[1])
    # print(len(degs[0]))
    # print(degs[0])
    # print(degs[1])
    for ax in axarr.flatten():
        prepare_ax(ax)
        # ax.set_ylim(0,0.8)
        # ax.set_xlim(0.,0.7)
        # ax.set_ylabel( '$S_n(L) - \\log(L\pi)/3$'  , fontsize=fontsize[1])
        # ax.set_xlabel('$L_n/L$', fontsize=fontsize[1])

    # prepare_plt(save=save,desc=desc, plot_type='Entanglement_entropy', savename=savename)
    plt.show()


def plot_ent_entropy_eigenstates_overlap(sybr,desc, modstr, syspar, modpar, nrows=1, ncols=2, titlestring='',save='False', savename='', new_ordering=True):


    """
    INPUT: 

    rules - a dictionary of rules on how to assign values to system parameters. How to write rules,
    an example code snippet:

    # sizelist=[5,7,9,11,13]
    # for size in sizelist:
    #     seznam={'ne':5, 'size':size, 'nu':3}
    #     print('before change',seznam)

    #     rules={'ne': lambda x,y: {'ne': 0 },
    #            'nu': lambda x,y:{'nu': int((y['size']-1)/2)},

    #     }

    #     for key in rules:

    #         seznam.update(rules[key](seznam[key], seznam))

    #         print('during change', seznam)

        

    """
    quantity='Ent_entropy'
    fig, axarr, fontsize=prepare_axarr(nrows=nrows, ncols=ncols, sharey=False, sharex=False,fontsize=[18,21,23])

    fig.suptitle(titlestring, fontsize=fontsize[-1])
    
    #nondeg
    file=dataSet(desc, modstr,quantity,syspar, modpar)
    file.load_dataSet(new_ordering=new_ordering)
    ave=file.mean_data()
    ave=ave.values

    # ax.scatter(ave.T[0],ave.T[-1])
    eng_arr=check_deg(ave.T[0])
    degs=get_level_deg(eng_arr, eps=1e-013)
    # print(degs[1])
    degs_=np.where(degs[1]==1)[0]    
    nondeg_ener=np.take(ave.T[0], degs_)
    nondeg_entro=np.take(ave.T[-1], degs_)


    modpar['HSYM']=sybr
    modpar['WSYM']=sybr
    file=dataSet(desc, modstr,quantity,syspar, modpar)
    file.load_dataSet(new_ordering=new_ordering)
    ave=file.mean_data()
    ave=ave.values


    # print(len(ave.T[0]))
    #nondeg
    axarr[0].scatter(ave.T[0],ave.T[-1], color='blue')
    #nondeg
    axarr[0].scatter(nondeg_ener, nondeg_entro, color='red')


    axarr[1].scatter(degs[0],degs[1])
    # print(len(degs[0]))
    # print(degs[0])
    # print(degs[1])
    for ax in axarr.flatten():
        prepare_ax(ax)
        # ax.set_ylim(0,0.8)
        # ax.set_xlim(0.,0.7)
        # ax.set_ylabel( '$S_n(L) - \\log(L\pi)/3$'  , fontsize=fontsize[1])
        # ax.set_xlabel('$L_n/L$', fontsize=fontsize[1])

    # prepare_plt(save=save,desc=desc, plot_type='Entanglement_entropy', savename=savename)
    plt.show()
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#to do: parsable arguments, 
if __name__=='__main__':

    """
    TYPICAL PARAMETER VALUES:
    SYM_BREAK graphs: typical range of sym_break parameters HSYM and WSYM: np.logspace(-3,1,25)

    W AND H VALUES:

        Between 0 and 20, either steps of 1 or 2; plotting code takes care of the missing values and does not raise an error - it only skips them 

    TYPICAL SYSTEMS: (ORDERING: L - Nh - Nu)


        Single hole:

        13 - 1 - 6
        11 - 1 - 5 => mainly for scaling analysis
        9  - 1 - 4 => mainly for scaling analysis
        7  - 1 - 3 => mainly for scaling analysis

        Finite doping:

        12 - 2 - 5
        12 - 6 - 3

    SYMBREAK VALUES (either both, or only HSYM and WSYM):

        0.5, 0.3, 0.1, 0.0

    """

    modules=['mbl_hop_ex', 'rd_field_ex']
    modstr=mod_str(modules)
    syspar={'size':9, 'nu':4, 'ne':0, 'dim':1, 'base':'F'}
    # desc='SS12_NH4_NU4_staggered'
    desc='XXZ_model_no_disorder_entanglement_entropy'
    wlist=[0]
    # h_stagg_list=[0,1,2,4]
    # h_stagg_list=[0,2,4]
    # modpar={'T':-1, 'J':1, 'JOF':1, 'W':2, 'H':0, 'WSYM':0.,'HSYM':0., 'H_STAGG':0}
    modpar={'T':-1, 'J':0.5, 'JOF':1, 'W':0}
    quantity='Ent_entropy'

    titlestring='Entanglement entropy for the PBC XXZ model,\n $J={}$, $\\Delta={}$'.format(modpar['JOF'], modpar['J'])
    savename='XXZ_entanglement_CFT_comparison.pdf'
    # plot_ent_entropy([5,7,9,11,13],desc, modstr, syspar, modpar, titlestring=titlestring,save=True, savename=savename)



    #---------------------------------------------------------------------------------------------------------
    #plot level statistics for one_third_doping case with sym_break

    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    print(modstr)
    syspar={'size':12, 'nu':4, 'ne':4, 'dim':1, 'base':'F'}

    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    desc='one_third_doping_scaling_sym_break_hole_spin_dis'


    sybr_list=[0.5]
    # wlist=np.arange(0,12,2)
    wlist=[1,2,3,4,5,6,7,8,]
    sizelist=[12]
    rules_third_doping={'ne': lambda x,y: {'ne': y['size']/3},
             'nu': lambda x,y: {'nu': y['size']/3},
              } #rules for one third doping and Sz=0 case

    sweep='W'


    # plot_sff(wlist,sweep,   sybr_list, desc, modstr, syspar, modpar)
    # r_mean_sweep_graph_double(sizelist, wlist, sybr_list,rules_third_doping, desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, save=True)

    #---------------------------------------------------------------------------------------------------------------
    #plot level statistics for the staggered case
    modules=['mbl_hop_ex', 'rd_field_ex', 'e_field_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    print(modstr)
    syspar={'size':6, 'nu':2, 'ne':2, 'dim':1, 'base':'F'}

    modpar={'T': -1, 'J':1, 'JOF':1., 'H_STAGG':0.2,'W':0, 'H':0 }
    desc='one_third_doping_scaling_staggered_hole_spin_dis'


    sybr_list=[0.2]
    wlist=np.arange(0,21,1)
    sizelist=[6,9,12]
    rules_third_doping={'ne': lambda x,y: {'ne': y['size']/3},
             'nu': lambda x,y: {'nu': y['size']/3},
              } #rules for one third doping and Sz=0 case

    
    # r_mean_sweep_graph_double(sizelist, wlist, sybr_list,rules_third_doping, desc=desc, modstr=modstr, syspar=syspar, modpar=modpar )
    #-------------------------------------------------------------------------------------------------------------
    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    print(modstr)
    syspar={'size':13, 'nu':6, 'ne':1, 'dim':1, 'base':'F'}
    wlist=np.arange(0,22,1)
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    rules_one_hole_doping={'ne': lambda x,y: {'ne': 1},
         'nu': lambda x,y: {'nu': (y['size']-y['ne'])/2},
          } #rules for one third doping and Sz=0 case

    desc='one_hole_doping_scaling_sym_break_hole_spin_dis'
    sizelist=[13]
    sybr_list=[0.5]
    sweep='H'
    wlist=[2,4,6,10]
    # plot_sff(wlist,sweep,   sybr_list, desc, modstr, syspar, modpar)
    # plot_DOS(wlist=[0,4,8,12],desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, save=False, all_cases=True, hist_bins=30)
    # plot_energies(wlist=[0,4,8,12],desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, save=True, all_cases=True )
    # plot_unfolding_three_demo(Wlist=[0,4,12], n=15, bounds=(0.25,0.75), hist_bins=30, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True, importrange=1)
    # plot_unfolding_three_error(Wlist=[0,4,12], n=15, bounds=(0.25,0.75), hist_bins=200, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True)
    # plot_unfolding_kolm_smir(Wlist=[0,4,12], n=7, bounds=(0.25,0.75), hist_bins=40, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True)
    # r_mean_sweep_graph_double(sizelist, wlist, sybr_list,rules_one_hole_doping, desc=desc, modstr=modstr, syspar=syspar, modpar=modpar )
    
    #-------------------------------------------------------------------------------------------------------------
    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    # print(modstr)
    syspar={'size':12, 'nu':4, 'ne':4, 'dim':1, 'base':'F'}
    wlist=np.arange(0,22,2)
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    # rules_one_hole_doping={'ne': lambda x,y: {'ne': 1},
    #      'nu': lambda x,y: {'nu': (y['size']-y['ne'])/2},
    #       } #rules for one third doping and Sz=0 case

    desc='one_third_doping_scaling_sym_break_hole_spin_dis'
    sizelist=[6,9,12]
    sybr_list=[0.5]
    # plot_DOS(wlist=[0,4,8,12],desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, save=False, all_cases=True, hist_bins=30)
    # plot_energies(wlist=[0,4,8,12],desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, save=True, all_cases=True )
    # plot_unfolding_three_demo(Wlist=[0,4,12], n=15, bounds=(0.25,0.75), hist_bins=30, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True)
    # plot_unfolding_three_error(Wlist=[0,4,12], n=15, bounds=(0.25,0.75), hist_bins=200, hist_bounds=(0,5), desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, all_cases=True)
    
    # r_mean_sweep_graph_double(sizelist, wlist, sybr_list,rules_third_doping, desc=desc, modstr=modstr, syspar=syspar, modpar=modpar )
    # plot_unfolding_schematics()
    # fig=plt.figure()

    #------------------------------------------------------------------------------------------------------------
    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)

    syspar={'size':11, 'nu':5, 'ne':1, 'dim':1, 'base':'F'}
    # wlist=np.arange(1,12,0.5)
    wlist=[1,2,3,4,8,14]
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    desc='one_hole_doping_check_ergodicity_sym_break_hole_spin_dis_new_ordering'

    # r_density_plot(wlist=wlist, desc=desc, modstr=modstr, syspar=syspar, modpar=modpar,load=False, importrange=1, new_ordering=True)
    sybr_list=[0.5]
    wlist=[1,2,3,4,6,10,12,14]
    # plot_sff_schematic()
    # plot_sff_four_new([1,2,8,12], [1,2,6,14],sybr_list, desc, modstr, syspar,modpar,new_ordering=True)
    # plot_sff_four([1,2,4,14], 'H',sybr_list, desc, modstr, syspar,modpar,new_ordering=True)
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    # plot_sff(wlist, 'H',sybr_list, desc, modstr, syspar,modpar,xlim=(5*10**(-3),3),ylim1=(7*10**(-3),3), ylim2=(7*10**(-3),3),new_ordering=True)

    #-------------------------------------------------------------------------------------------------------------
    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)

    syspar={'size':9, 'nu':3, 'ne':3, 'dim':1, 'base':'F'}
    wlist=[1,2,3,4,6,8,14]
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    desc='one_third_doping_check_ergodicity_sym_break_hole_spin_dis_new_ordering'

    # plot_sff_four_new([1,4,8,14], [1,4,8,14],sybr_list, desc, modstr, syspar,modpar,n_window=100,new_ordering=True)
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    # plot_sff(wlist, 'H',sybr_list, desc, modstr, syspar,modpar,xlim=(5*10**(-3),3),ylim1=(7*10**(-3),3), ylim2=(7*10**(-3),3),new_ordering=True)
    # r_density_plot(wlist=wlist, desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, importrange=1, load=True,save=True, new_ordering=True)
    
    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)

    syspar={'size':14, 'nu':7, 'ne':0, 'dim':1, 'base':'F'}
    wlist=np.arange(1,4.5,0.5)
    wlist=[0.5,1,1.5,2.,2.5,3.5]
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0, 'H':0 }
    desc='no_doping_XXX_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    # plot_sff_four([1,1.5,2,3.5], 'W',sybr_list, desc, modstr, syspar,modpar,new_ordering=True)
    # plot_sff(wlist, 'W',sybr_list, desc, modstr, syspar,modpar,xlim=(5*10**(-3),3),ylim1=(7*10**(-3),3), ylim2=(7*10**(-3),3),new_ordering=True)
    # r_density_plot(wlist=wlist, desc=desc, modstr=modstr, syspar=syspar, modpar=modpar, importrange=1, load=True,save=True, new_ordering=True)
    




    #-------------------------------------------------------------------------------------------------------------
    #plot level statistics for the staggered field and L=12 N_h=4 N_u=4 case
    modules=['e_field_ex', 'mbl_hop_ex', 'rd_field_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    syspar={'size':12, 'nu':4, 'ne':4, 'dim':1, 'base':'F'}

    modpar={'T':-1, 'J':1, 'JOF':1, 'H_STAGG':0.2, 'W':0, 'H':0}
    desc='SS12_NH4_NU4_staggered'
    wlist=[1,2,3,4,5]
    # r_mean_sweep_graph(wlist,sweep='H', sybr_list=[0.2],modstr=modstr, desc=desc, syspar=syspar, modpar=modpar,filetype='txt')


    #---------------------------------------------------------------------------------------------------------------
    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)

    Wlist=np.logspace(-3,1,25)[:-1]
    # Wsym=np.logspace(-3,1,25)
    sizelist=[7,9,11,13]

    syspar={'size':11, 'nu':5, 'ne':1, 'dim':1, 'base':'F'}
    modpar={'T':-1, 'J':1, 'JOF':1, 'W':0, 'H':0, 'WSYM':0.5,'HSYM':0.5}
    desc='one_hole_sym_break_scaling_analysis'


    # desc='SS13_NH1_NU6_spinhole_sym_break'
    # # desc='SS12_NH2_NU5_spinhole_sym_break'
    # desc='SS12_NH6_NU3_spinhole_sym_break'
    # sym_break_test_graph(Wlist, sizelist,syspar=syspar,desc=desc, modpar=modpar, modstr=modstr)


    #test ent. entropy loading




    quantity='Ent_entropy'

    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    print(modstr)
    syspar={'size':9, 'nu':3, 'ne':3, 'dim':1, 'base':'F'}

    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0., 'H':0. }
    # desc='one_hole_doping_scaling_sym_break_hole_spin_dis'
    # desc='one_third_doping_scaling_sym_break_hole_spin_dis'
    # desc='test_new_folder_order_1'
    # desc='one_third_doping_check_ergodicity_sym_break_hole_spin_dis'
    # desc='one_hole_doping_check_ergodicity_sym_break_hole_spin_dis'
    # desc='one_hole_doping_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    desc='one_third_doping_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    # wlist=np.arange(0.,2.,2.)
    wlist=np.arange(0.,14.5,0.5)
    # entropy_density_plot(wlist,'H', desc, modstr, syspar, modpar, bins=21, new_ordering=True)
    #-------------------------------------------------------------------------------
    #plot heisenberg

    modules=['mbl_hop_ex', 'rd_field_ex', 'sybr_s_ex', 'sybr_h_ex', 'rd_holes_ex']
    modstr=mod_str(modules)
    print(modstr)
    syspar={'size':14, 'nu':7, 'ne':0, 'dim':1, 'base':'F'}
    # desc='heisenberg_XXX_check_entanglement_entropy'
    # desc='small_nsamples_one_third_doping_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    desc='no_doping_XXX_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    # desc='one_hole_doping_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    # desc='one_third_doping_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0.0, 'H':0. }


    wlist=np.arange(0.,14.5,0.5)
    # for dis in [ 'W']:

    #     entropy_density_plot_cuts(wlist,dis, desc, modstr, syspar, modpar, importrange=1,bins=41, new_ordering=True, heisenberg=True, load_data=True)

    #     entropy_density_plot(wlist,dis, desc, modstr, syspar, modpar, importrange=1,bins=41, new_ordering=True, heisenberg=True)

    modpar={'T': -1, 'J':1., 'JOF':1., 'HSYM':0.5, 'WSYM':0.5,'W':0., 'H':0. }
    syspar={'size':14, 'nu':7, 'ne':0, 'dim':1, 'base':'F'}
    # desc='one_hole_doping_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    # desc='check_entanglement_entropy_no_disorder'
    desc='no_doping_XXX_check_ergodicity_sym_break_hole_spin_dis_new_ordering'
    # desc='no_doping_XXX_entropy_scaling_new_ordering_new'


    # plot_ent_entropy_scaling_systems([0,8],[6,8,10,12],'W',desc, modstr,syspar, modpar, new_ordering=True )
    syspar={'size':14, 'nu':7, 'ne':0, 'dim':1, 'base':'F'}
    plot_ent_entropy_scaling_systems([0,8],[6,8,10,12,14],'W',desc, modstr,syspar, modpar, new_ordering=True )
    plot_ent_entropy_scaling_subsystems([0,2,4,8],'W',desc, modstr,syspar, modpar, new_ordering=True )
    # plot_ent_entropy_eigenstates(desc, modstr, syspar, modpar, nrows=1, ncols=2, titlestring='',save='False', savename='', new_ordering=True, heisenberg=False)
    # plot_ent_entropy_eigenstates_overlap(0.25,desc, modstr, syspar, modpar, nrows=1, ncols=2, titlestring='',save='False', savename='', new_ordering=True)




    # plot_degeneracy([0], [0], 'W', desc, modstr, syspar, modpar,all_cases=True,save=False, importrange=1, new_ordering=True)