import sys
import numpy as np
import scipy
import scipy.sparse
try:
    from PIL import Image
except Exception as e:
    print "No module PIL.\nYou need to install it if you want to run TrainDL tests\n"
    raise e

import spams
import time
from test_utils import *

if not ('rand' in scipy.sparse.__dict__):
    import myscipy_rand as ssp
else:
    import scipy.sparse as ssp

def _extract_lasso_param(f_param):
    lst = [ 'L','lambda1','lambda2','mode','pos','ols','numThreads','length_path','verbose','cholesky']
    l_param = {'return_reg_path' : False}
    for x in lst:
        if x in f_param:
            l_param[x] = f_param[x]
    return l_param
    
def test_trainDL():
    img_file = '../extdata/boat.png'
    try:
        img = Image.open(img_file)
    except:
        print "Cannot load image %s : skipping test" %img_file
        return None
    I = np.array(img) / 255.
    if I.ndim == 3:
        A = np.asfortranarray(I.reshape((I.shape[0],I.shape[1] * I.shape[2])))
        rgb = True
    else:
        A = np.asfortranarray(I)
        rgb = False

    m = 8;n = 8;
    X = spams.im2col_sliding(A,m,n,rgb)

    X = X - np.tile(np.mean(X,0),(X.shape[0],1))
    X = np.asfortranarray(X / np.tile(np.sqrt((X * X).sum(axis=0)),(X.shape[0],1)))
    param = { 'K' : 100, # learns a dictionary with 100 elements
              'lambda1' : 0.15, 'numThreads' : 4, 'batchsize' : 400,
              'iter' : 1000}

    ########## FIRST EXPERIMENT ###########
    tic = time.time()
    D = spams.trainDL(X,**param)
    tac = time.time()
    t = tac - tic
    print 'time of computation for Dictionary Learning: %f' %t

    ##param['approx'] = 0
    print 'Evaluating cost function...'
    lparam = _extract_lasso_param(param)
    alpha = spams.lasso(X,D = D,**lparam)
    xd = X - D * alpha
    R = np.mean(0.5 * (xd * xd).sum(axis=0) + param['lambda1'] * np.abs(alpha).sum(axis=0))
    #* display ????

    print "objective function: %f" %R

    #### SECOND EXPERIMENT ####
    print "*********** SECOND EXPERIMENT ***********"

    X1 = X[:,0:X.shape[1]/2]
    X2 = X[:,X.shape[1]/2 -1:]
    param['iter'] = 500
    tic = time.time()
    (D,model) = spams.trainDL(X1,return_model = True,**param)
    tac = time.time()
    t = tac - tic
    print 'time of computation for Dictionary Learning: %f\n' %t
    print 'Evaluating cost function...'
    alpha = spams.lasso(X,D = D,**lparam)
    xd = X - D * alpha
    R = np.mean(0.5 * (xd * xd).sum(axis=0) + param['lambda1'] * np.abs(alpha).sum(axis=0))
    print "objective function: %f" %R

    # Then reuse the learned model to retrain a few iterations more.
    param2 = param.copy()
    param2['D'] = D
    tic = time.time()
    (D,model) = spams.trainDL(X2,return_model = True,model = model,**param2)
    tac = time.time()
    t = tac - tic
    print 'time of computation for Dictionary Learning: %f' %t
    print 'Evaluating cost function...'
    alpha = spams.lasso(X,D = D,**lparam)
    xd = X - D * alpha
    R = np.mean(0.5 * (xd * xd).sum(axis=0) + param['lambda1'] * np.abs(alpha).sum(axis=0))
    print "objective function: %f" %R

    #################### THIRD & FOURTH EXPERIMENT ######################
    # let us add sparsity to the dictionary itself

    print '*********** THIRD EXPERIMENT ***********'
    param['modeParam'] = 0
    param['iter'] = 1000
    param['gamma1'] = 0.3
    param['modeD'] = 1

    tic = time.time()
    D = spams.trainDL(X,**param)
    tac = time.time()
    t = tac - tic
    print 'time of computation for Dictionary Learning: %f' %t
    print 'Evaluating cost function...'
    alpha = spams.lasso(X,D = D,**lparam)
    xd = X - D * alpha
    R = np.mean(0.5 * (xd * xd).sum(axis=0) + param['lambda1'] * np.abs(alpha).sum(axis=0))
    print "objective function: %f" %R

    #* DISPLAY
    print '*********** FOURTH EXPERIMENT ***********'
    param['modeParam'] = 0
    param['iter'] = 1000
    param['gamma1'] = 0.3
    param['modeD'] = 3

    tic = time.time()
    D = spams.trainDL(X,**param)
    tac = time.time()
    t = tac - tic
    print 'time of computation for Dictionary Learning: %f' %t
    print 'Evaluating cost function...'
    alpha = spams.lasso(X,D = D,**lparam)
    xd = X - D * alpha
    R = np.mean(0.5 * (xd * xd).sum(axis=0) + param['lambda1'] * np.abs(alpha).sum(axis=0))
    print "objective function: %f" %R

    
    return None

def test_trainDL_Memory():
    img_file = '../extdata/lena.png'
    try:
        img = Image.open(img_file)
    except:
        print "Cannot load image %s : skipping test" %img_file
        return None
    I = np.array(img) / 255.
    if I.ndim == 3:
        A = np.asfortranarray(I.reshape((I.shape[0],I.shape[1] * I.shape[2])))
        rgb = True
    else:
        A = np.asfortranarray(I)
        rgb = False

    m = 8;n = 8;
    X = spams.im2col_sliding(A,m,n,rgb)

    X = X - np.tile(np.mean(X,0),(X.shape[0],1))
    X = np.asfortranarray(X / np.tile(np.sqrt((X * X).sum(axis=0)),(X.shape[0],1)))
    X = np.asfortranarray(X[:,np.arange(0,X.shape[1],10)])

    param = { 'K' : 200, # learns a dictionary with 100 elements
          'lambda1' : 0.15, 'numThreads' : 4,
          'iter' : 100}

    ############# FIRST EXPERIMENT  ##################
    tic = time.time()
    D = spams.trainDL_Memory(X,**param)
    tac = time.time()
    t = tac - tic
    print 'time of computation for Dictionary Learning: %f' %t

    print 'Evaluating cost function...'
    lparam = _extract_lasso_param(param)
    alpha = spams.lasso(X,D = D,**lparam)
    xd = X - D * alpha
    R = np.mean(0.5 * (xd * xd).sum(axis=0) + param['lambda1'] * np.abs(alpha).sum(axis=0))
    print "objective function: %f" %R
    #* ? DISPLAY

    ############# SECOND EXPERIMENT  ##################
    tic = time.time()
    D = spams.trainDL(X,**param)
    tac = time.time()
    t = tac - tic
    print 'time of computation for Dictionary Learning: %f' %t
    print 'Evaluating cost function...'
    alpha = spams.lasso(X,D = D,**lparam)
    xd = X - D * alpha
    R = np.mean(0.5 * (xd * xd).sum(axis=0) + param['lambda1'] * np.abs(alpha).sum(axis=0))
    print "objective function: %f" %R

    #* ? DISPLAY

    return None

def test_nmf():
    img_file = '../extdata/boat.png'
    try:
        img = Image.open(img_file)
    except:
        print "Cannot load image %s : skipping test" %img_file
        return None
    I = np.array(img) / 255.
    if I.ndim == 3:
        A = np.asfortranarray(I.reshape((I.shape[0],I.shape[1] * I.shape[2])))
        rgb = True
    else:
        A = np.asfortranarray(I)
        rgb = False

    m = 16;n = 16;
    X = spams.im2col_sliding(A,m,n,rgb)
    X = X[:,::10]
    X = np.asfortranarray(X / np.tile(np.sqrt((X * X).sum(axis=0)),(X.shape[0],1)))
    ########## FIRST EXPERIMENT ###########
    tic = time.time()
    (U,V) = spams.nmf(X,return_lasso= True,K = 49,numThreads=4,iter = -5)
    tac = time.time()
    t = tac - tic
    print 'time of computation for Dictionary Learning: %f' %t

    print 'Evaluating cost function...'
    Y = X - U * V
    R = np.mean(0.5 * (Y * Y).sum(axis=0))
    print 'objective function: %f' %R
    return None

tests = [
    'trainDL' , test_trainDL,
    'trainDL_Memory' , test_trainDL_Memory,
    'nmf' , test_nmf,
]
