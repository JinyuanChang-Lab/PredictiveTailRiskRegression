import torch, sys
import numpy as np
np.seterr(all='ignore')
import numpy.random as rgt
from numpy.random import default_rng
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from quantes.nonlinear import FNN, LocPoly
from quantes.linear import joint
from sklearn.model_selection import KFold
from scipy.stats import norm, t


def quantile_loss(yhat, y, tau):
    '''
    Quantile Loss Function
    '''
    error = y - yhat
    return np.maximum(tau*error, (tau-1)*error).mean()

a1 = lambda x : 0.138 + (0.316 + 0.982 * x) * np.exp(-3.89 * x**2)
a2 = lambda x : -0.437 - (0.659 + 1.26 * x) * np.exp(-3.89 * x**2)

def gen_data_betamixing(n, type = 'EXPAR', err_dist='normal', df=2,tau = 0.1, err_scale = 1, random_state=0, **kwargs):
    # print(type)
    rgt.seed(random_state)
    return eval("gen_data_" + type)(n, err_dist, df,tau,err_scale, random_state, **kwargs)


def gen_data_EXPAR_v1(n, err_dist='normal', df=2, tau = 0.1, err_scale = 1, random_state=0, **kwargs):
    # print(n_train, err_dist, df, random_state)
    fix_err = 0.2
    p = 8
    n_rm = 100
    n_test= 1024
    rgt.seed(random_state)
    N = n + n_rm + n_test
    Y = np.zeros(N)
    X = np.zeros((N, p))
    if err_dist == 'normal':
        err = rgt.normal(0, 1, N)
        qt = norm.ppf(tau)
        es = norm.expect(lambda x : (x if x < qt else 0))/tau
    elif err_dist == 't':
        err = rgt.standard_t(df, N)
        qt = t.ppf(tau, df=df)
        es = t.expect(lambda x : (x if x < qt else 0), args=(df,))/tau
    for i in range(N):
        if i == 0:
            Y[i] = fix_err * err_scale * err[i]
        else:
            ind = 8 if i>8 else i
            ind1 = max(0, i-8)
            X[i,0:ind] = Y[ind1:i][::-1]
            Y[i] = a1(X[i, 0]) * X[i, 0] + a2(X[i, 0]) * X[i, 1] \
                    + a1(X[i, 2]) * X[i, 2] + a2(X[i, 2]) * X[i, 3] \
                    + a1(X[i, 4]) * X[i, 4] + a2(X[i, 4]) * X[i, 5] \
                    + a1(X[i, 6]) * X[i, 6] + a2(X[i, 6]) * X[i, 7] \
                    + fix_err * err_scale * err[i]
    X_test = X[-(n+n_test):, :]
    Y_test = Y[-(n+n_test):]
    test_es = a1(X_test[:, 0]) * X_test[:, 0] + a2(X_test[:, 0]) * X_test[:, 1] \
            + a1(X_test[:, 2]) * X_test[:, 2] + a2(X_test[:, 2]) * X_test[:, 3] \
            + a1(X_test[:, 4]) * X_test[:, 4] + a2(X_test[:, 4]) * X_test[:, 5] \
            + a1(X_test[:, 6]) * X_test[:, 6] + a2(X_test[:, 6]) * X_test[:, 7] \
            + fix_err * err_scale * es
    test_qt = a1(X_test[:, 0]) * X_test[:, 0] + a2(X_test[:, 0]) * X_test[:, 1] \
            + a1(X_test[:, 2]) * X_test[:, 2] + a2(X_test[:, 2]) * X_test[:, 3] \
            + a1(X_test[:, 4]) * X_test[:, 4] + a2(X_test[:, 4]) * X_test[:, 5] \
            + a1(X_test[:, 6]) * X_test[:, 6] + a2(X_test[:, 6]) * X_test[:, 7] \
            + fix_err * err_scale * qt
    return X_test, Y_test, test_qt, test_es
    

def gen_data_FAR(n, mode='train', err_dist='normal', df=2, tau = 0.1, err_scale = 1, random_state=0, X_test=None, **kwargs):
    # print(n_train, err_dist, df, random_state)
    n_rm = 100
    rgt.seed(random_state)
    Y = np.zeros(n + n_rm)
    X = np.zeros((n + n_rm,2))
    if err_dist == 'normal':
        err = rgt.normal(0, 1, n + n_rm)
        qt = norm.ppf(tau)
        es = norm.expect(lambda x : (x if x < qt else 0))/tau
    elif err_dist == 't':
        err = rgt.standard_t(df, n + n_rm)
        qt = t.ppf(tau, df=df)
        es = t.expect(lambda x : (x if x < qt else 0), args=(df,))/tau
    if mode == 'train':
        for i in range(n + n_rm):
            if i == 0:
                Y[i] = 0.5 * err_scale * err[i]
            elif i == 1:
                Y[i] = Y[i-1] + 0.5 * err_scale * err[i]
                X[i, 0] = Y[i-1]
            else:
                Y[i] = -Y[i-2] * np.exp(-Y[i-2]**2 / 2) \
                    + (1 / (1 + Y[i-2]**2)) * np.cos(1.5 * Y[i-2]) * Y[i-1] + 0.5 * err_scale * err[i]
                X[i, :] = np.array([Y[i-1], Y[i-2]])
        return X[-n:, :], Y[-n:]
    elif mode == 'test':
        if X_test is None:
            X_test = rgt.uniform(0, 1, (n+n_rm, 2))
            X_test[1:, 1] = X_test[:-1, 0]
        test_es = -X_test[:, 1]*np.exp(-X_test[:, 1]**2 / 2) \
            + (1 / (1 + X_test[:, 1]**2)) * np.cos(1.5 * X_test[:, 1]) * X_test[:, 0] + 0.5 * err_scale * es
        test_qt = -X_test[:, 1]*np.exp(-X_test[:, 1]**2 / 2) \
            + (1 / (1 + X_test[:, 1]**2)) * np.cos(1.5 * X_test[:, 1]) * X_test[:, 0] + 0.5 * err_scale * qt
        return X_test[-n:, :], test_qt[-n:], test_es[-n:]
    

def gen_data_FAR_v1(n, err_dist='normal', df=2, tau = 0.1, err_scale = 1, random_state=0, **kwargs):
    # print(n_train, err_dist, df, random_state)
    p = 8
    n_rm = 100
    n_test= 1024
    N = n + n_rm + n_test
    rgt.seed(random_state)
    Y = np.zeros(N)
    X = np.zeros((N, p))
    if err_dist == 'normal':
        err = rgt.normal(0, 1, N)
        qt = norm.ppf(tau)
        es = norm.expect(lambda x : (x if x < qt else 0))/tau
    elif err_dist == 't':
        err = rgt.standard_t(df, N)
        qt = t.ppf(tau, df=df)
        es = t.expect(lambda x : (x if x < qt else 0), args=(df,))/tau
    # if mode == 'train':
    for i in range(N):
        if i == 0:
            Y[i] = 0.5 * err_scale * err[i]
        else:
            ind = p if i>p else i
            ind1 = max(0, i-p)
            X[i,0:ind] = Y[ind1:i][::-1]
            ind3 = np.arange(0, p ,2)
            ind4 = np.arange(1, p ,2)
            # Y[i] = - X[i, 1] * np.exp(-X[i, 1]**2 / 2) \
            #     + (1 / (1 + X[i, 1]**2)) * np.cos(1.5 * X[i, 1]) * X[i, 0] + 0.5 * err_scale * err[i
            Y[i] = np.sum(-X[i, ind4] * np.exp(-X[i, ind4]**2 / 2) + (1 / (1 + X[i, ind4]**2)) * np.cos(1.5 * X[i, ind4]) * X[i, ind3]) + 0.5 * err_scale * err[i]
    # elif mode == 'test':
    # ind3 = np.arange(0, p ,2)
    # ind4 = np.arange(1, p ,2)
        # if X_test is None:
        #     X_test = rgt.uniform(0, 1, (n+n_rm, p))
        #     for j in range(p-1):
        #         X_test[(j+1):, (j+1)] = X_test[:-(j+1), 0]
    X_test = X[-(n+n_test):, :]
    Y_test = Y[-(n+n_test):]
    test_es = np.sum(-X_test[:, ind4]*np.exp(-X_test[:, ind4]**2 / 2) \
        + (1 / (1 + X_test[:, ind4]**2)) * np.cos(1.5 * X_test[:, ind4]) * X_test[:, ind3], axis = 1) + 0.5 * err_scale * es
    test_qt = np.sum(-X_test[:, ind4]*np.exp(-X_test[:, ind4]**2 / 2) \
        + (1 / (1 + X_test[:, ind4]**2)) * np.cos(1.5 * X_test[:, ind4]) * X_test[:, ind3], axis = 1) + 0.5 * err_scale * qt
    return X_test, Y_test, test_qt, test_es


def gen_data_SIM(n, mode='train', err_dist='normal', df=2, tau = 0.1, err_scale = 1, random_state=0, X_test=None, **kwargs):
    # print(n_train, err_dist, df, random_state)
    n_rm = 100
    rgt.seed(random_state)
    Y = np.zeros(n + n_rm)
    X = np.zeros((n + n_rm,2))
    if err_dist == 'normal':
        err = rgt.normal(0, 1, n + n_rm)
        qt = norm.ppf(tau)
        es = norm.expect(lambda x : (x if x < qt else 0))/tau
    elif err_dist == 't':
        err = rgt.standard_t(df, n + n_rm)
        qt = t.ppf(tau, df=df)
        es = t.expect(lambda x : (x if x < qt else 0), args=(df,))/tau
    if mode == 'train':
        for i in range(n + n_rm):
            if i == 0:
                Z = -0.6
                Y[i] = np.exp(-8 * Z**2) + 0.1 * err_scale * err[i]
            elif i == 1:
                Z = 0.8 * Y[i-1] - 0.6
                Y[i] = np.exp(-8 * Z**2) + 0.5 * np.sin(2 * np.pi * Z) * Y[i-1] + 0.1 * err_scale * err[i]
                X[i, 0] = Y[i-1]
            else:
                Z = 0.8 * Y[i-1] + 0.6 * Y[i-2] - 0.6
                Y[i] = np.exp(-8 * Z**2) + 0.5 * np.sin(2 * np.pi * Z) * Y[i-1] + 0.1 * err_scale * err[i]
                X[i, :] = np.array([Y[i-1], Y[i-2]])
        return X[-n:, :], Y[-n:]
    elif mode == 'test':
        if X_test is None:
            X_test = rgt.uniform(0, 1, (n+n_rm, 2))
            X_test[1:, 1] = X_test[:-1, 0]
        Z = 0.8 * X_test[:, 0] + 0.6 * X_test[:, 1] - 0.6
        test_es = np.exp(-8 * Z**2) + 0.5 * np.sin(2 * np.pi * Z) * X_test[:, 0] + 0.1 * err_scale * es
        test_qt = np.exp(-8 * Z**2) + 0.5 * np.sin(2 * np.pi * Z) * X_test[:, 0] + 0.1 * err_scale * qt
        return X_test[-n:, :], test_qt[-n:], test_es[-n:]
    
def gen_data_SIM_v1(n, err_dist='normal', df=2, tau = 0.1, err_scale = 1, random_state=0, **kwargs):
    # print(n_train, err_dist, df, random_state)
    n_rm = 100
    n_test= 1024
    N = n + n_rm + n_test
    rgt.seed(random_state)
    Y = np.zeros(N)
    X = np.zeros((N,8))
    if err_dist == 'normal':
        err = rgt.normal(0, 1, N)
        qt = norm.ppf(tau)
        es = norm.expect(lambda x : (x if x < qt else 0))/tau
    elif err_dist == 't':
        err = rgt.standard_t(df, N)
        qt = t.ppf(tau, df=df)
        es = t.expect(lambda x : (x if x < qt else 0), args=(df,))/tau
    for i in range(N):
        if i == 0:
            Z1 = -0.6
            Z2 = -0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.1 * err_scale * err[i]
        elif i == 1:
            Z1 = 0.8 * Y[i-1] - 0.6
            Z2 = -0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * Y[i-1] + 0.1 * err_scale * err[i]
            X[i, 0] = Y[i-1]
        elif i == 2:
            Z1 = 0.8 * Y[i-1] + 0.6 * Y[i-2] - 0.6
            Z2 = - 0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * Y[i-1] + 0.1 * err_scale * err[i]
            X[i, 0:2] = np.array([Y[i-1], Y[i-2]])
        elif i == 3:
            Z1 = 0.8 * Y[i-1] + 0.6 * Y[i-2] - 0.4 * Y[i-3] - 0.6
            Z2 = - 0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * Y[i-1] + 0.1 * err_scale * err[i]
            X[i, 0:3] = np.array([Y[i-1], Y[i-2], Y[i-3]])
        elif i == 4:
            Z1 = 0.8 * Y[i-1] + 0.6 * Y[i-2] - 0.4 * Y[i-3] - 0.2 * Y[i-4] - 0.6
            Z2 = - 0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * Y[i-1] + 0.1 * err_scale * err[i]
            X[i, 0:4] = np.array([Y[i-1], Y[i-2], Y[i-3], Y[i-4]])
        elif i == 5:
            Z1 = 0.8 * Y[i-1] + 0.6 * Y[i-2] - 0.4 * Y[i-3] - 0.2 * Y[i-4] - 0.6
            Z2 = 0.8 * Y[i-5] - 0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * Y[i-1] + 0.1 * err_scale * err[i]
            X[i, 0:5] = np.array([Y[i-1], Y[i-2], Y[i-3], Y[i-4], Y[i-5]])
        elif i == 6:
            Z1 = 0.8 * Y[i-1] + 0.6 * Y[i-2] - 0.4 * Y[i-3] - 0.2 * Y[i-4] - 0.6
            Z2 = 0.8 * Y[i-5] + 0.6 * Y[i-6] - 0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * Y[i-1] + 0.1 * err_scale * err[i]
            X[i, 0:6] = np.array([Y[i-1], Y[i-2], Y[i-3], Y[i-4], Y[i-5], Y[i-6]])
        elif i == 7:
            Z1 = 0.8 * Y[i-1] + 0.6 * Y[i-2] - 0.4 * Y[i-3] - 0.2 * Y[i-4] - 0.6
            Z2 = 0.8 * Y[i-5] + 0.6 * Y[i-6] - 0.4 * Y[i-7] - 0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * Y[i-1] + 0.1 * err_scale * err[i]
            X[i, 0:7] = np.array([Y[i-1], Y[i-2], Y[i-3], Y[i-4], Y[i-5], Y[i-6], Y[i-7]])
        else:
            Z1 = 0.8 * Y[i-1] + 0.6 * Y[i-2] - 0.4 * Y[i-3] - 0.2 * Y[i-4] - 0.6
            Z2 = 0.8 * Y[i-5] + 0.6 * Y[i-6] - 0.4 * Y[i-7] - 0.2 * Y[i-8] - 0.6
            Y[i] = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * Y[i-1] + 0.1 * err_scale * err[i]
            X[i, :] = np.array([Y[i-1], Y[i-2], Y[i-3], Y[i-4], Y[i-5], Y[i-6], Y[i-7], Y[i-8]])
    X_test = X[-(n+n_test):, :]
    Y_test = Y[-(n+n_test):]
    Z1 = 0.8 * X_test[:, 0] + 0.6 * X_test[:, 1] - 0.4 * X_test[:, 2] - 0.2 * X_test[:, 3] - 0.6
    Z2 = 0.8 * X_test[:, 4] + 0.6 * X_test[:, 5] - 0.4 * X_test[:, 6] - 0.2 * X_test[:, 7] - 0.6
    test_es = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * X_test[:, 0] + 0.1 * err_scale * es
    test_qt = np.exp(-8 * Z1**2) + 0.5 * np.sin(2 * np.pi * Z2) * X_test[:, 0] + 0.1 * err_scale * qt
    return X_test, Y_test, test_qt, test_es



###############################################################################
######################## Local Polynomial Regression ##########################
###############################################################################
def Qt_LP(X, Y, X_val, Y_val, tau, kernel, grid_q, degree=1, plot=False):
    model = LocPoly(X, Y, kernel=kernel)
    val_err = []
    for h in grid_q:
        pred_q = model.qt_predict(x0=X_val, bw=h, tau=tau, degree=degree)
        val_err.append(quantile_loss(pred_q, Y_val, tau))
    val_err = np.array(val_err)
    if plot:
        print('minimum qt-llr val error:', val_err.min().round(4))
        plt.plot(grid_q, val_err)
        plt.ylabel('validation error')
        plt.xlabel('bandwidth')
        plt.title('qt-llr')
        plt.show()

    bw_q = grid_q[val_err.argmin()]
    fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree)
    return {'bw_q': bw_q,
            'fit_q': fit_q,
            'surrogate_y': np.minimum(Y - fit_q, 0)/tau + fit_q}


def ES_LP(X, Y, X_val, Y_val, tau, 
          kernel_q, bw_q, kernel_e, grid_e, 
          degree=1, plot=False):
    model1 = LocPoly(X, Y, kernel=kernel_q)
    fit_q = model1.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree)
    Y0 = np.minimum(Y - fit_q, 0)/tau + fit_q
    pred_q = model1.qt_predict(x0=X_val, bw=bw_q, tau=tau, degree=degree)
    Y0_val = np.minimum(Y_val - pred_q, 0)/tau + pred_q
    
    val_err = []
    model2 = LocPoly(X, Y0, kernel=kernel_e)
    for h in grid_e:
        pred_e = model2.ls_predict(x0=X_val, bw=h, degree=degree)
        val_err.append(np.mean((Y0_val - pred_e)**2))
    val_err = np.array(val_err)

    if plot:
        print('minimum es-llr val error:', val_err.min().round(4))
        plt.plot(grid_e, val_err)
        plt.ylabel('validation error')
        plt.xlabel('bandwidth')
        plt.title('es-llr')
        plt.show()

    return {'bw_q': bw_q, 'bw_e': grid_e[val_err.argmin()], 'surrogate_y': Y0}


def QtES_LP(X, Y, X_val, Y_val, tau,
            kernel_q=norm.pdf, grid_q=np.linspace(.1, 1, 10), degree_q=1,
            kernel_e=norm.pdf, grid_e=np.linspace(.25, 1, 10), degree_e=1,
            plot=False):
    
    model = LocPoly(X, Y, kernel=kernel_q)
    if type(grid_q) == float or type(grid_q) == int:
        bw_q = grid_q
        fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree_q)
        pred_q = model.qt_predict(x0=X_val, bw=bw_q, tau=tau, degree=degree_q)
    elif type(grid_q) == np.ndarray and len(grid_q) == 1:
        bw_q = grid_q[0]
        fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree_q)
        pred_q = model.qt_predict(x0=X_val, bw=bw_q, tau=tau, degree=degree_q)
    else:
        val_err_q = []
        qt_pred = np.empty(shape=[len(Y_val), len(grid_q)])
        valid_indices = []
        bw_q, pred_q, fit_q = None, None, None
        for m, h in enumerate(grid_q):
            try:
                qt_pred[:, m] = model.qt_predict(x0=X_val, bw=h,
                                                 tau=tau, degree=degree_q)
                val_err_q.append(quantile_loss(qt_pred[:, m], Y_val, tau))
                valid_indices.append(m) 
            except Exception as e:
                print(e)
                continue
        if val_err_q:
            val_err_q = np.array(val_err_q)
            min_idx = val_err_q.argmin() 
            best_h_idx = valid_indices[min_idx]
            bw_q = grid_q[best_h_idx]
            pred_q = qt_pred[:, best_h_idx]
            fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree_q)
        else:
            raise Exception("No valid bandwidth values found for qt.")

        if plot:
            print('minimum qt-localpoly val error:', 
                  val_err_q.min().round(4))
            plt.plot(grid_q, val_err_q)
            plt.ylabel('validation error')
            plt.xlabel('bandwidth')
            plt.title('qt-localpoly ')
            plt.show()

    Y0 = np.minimum(Y - fit_q, 0)/tau + fit_q
    Y0_val = np.minimum(Y_val - pred_q, 0)/tau + pred_q
    
    if type(grid_e) == float or type(grid_e) == int:
        bw_e = grid_e
    elif type(grid_e) == np.ndarray and len(grid_e) == 1:
        bw_e = grid_e[0]
    else:
        val_err_e = []
        valid_indices = []
        model2 = LocPoly(X, Y0, kernel=kernel_e)
        for m, h in enumerate(grid_e):
            # pred_e = model2.ls_predict(x0=X_val, bw=h, degree=degree_e)
            # val_err_e.append(np.mean((Y0_val - pred_e)**2))
            try:
                pred_e = model2.ls_predict(x0=X_val, bw=h, degree=degree_e)
                val_err_e.append(np.mean((Y0_val - pred_e)**2))
                valid_indices.append(m)
            except Exception as e:
                continue
        if val_err_e:
            val_err_e = np.array(val_err_e)
            min_idx = val_err_e.argmin()
            best_h_idx = valid_indices[min_idx]
            bw_e = grid_e[best_h_idx]
        else:
            raise Exception("No valid bandwidth values found for es.")

        if plot:
            print('minimum es-localpoly val error:', 
                  val_err_e.min().round(4))
            plt.plot(grid_q, val_err_e)
            plt.ylabel('validation error')
            plt.xlabel('bandwidth')
            plt.title('es-localpoly ')
            plt.show()

    return {'bw_q': bw_q, 'bw_e': bw_e, 'surrogate_y': Y0, 'fit_q': fit_q}



def Qt_Linear(X, Y, X_val, Y_val, tau,
            grid_vsigma=np.linspace(.1, 1, 10),standardize = True,
            plot=False,  kernel = 'Laplacian', intercept=True):


    val_err_q = []
    qt_pred = np.empty(shape=[len(Y_val), len(grid_vsigma)])
    valid_indices = []
    bvsigma_q, pred_q, fit_q = None, None, None
    for m, h in enumerate(grid_vsigma):
        try:
            X1 = X[:,intercept:]
            X1 = np.clip(X1, -h, h)
            model = joint(X1, Y, intercept = intercept)
            fit_q = model.fit(tau=tau, kernel = 'Laplacian', 
                         standardize = standardize)
            # print(fit_q['beta'].shape)
            qt_pred[:, m] =  X_val @ fit_q['beta']  
            # print(quantile_loss(qt_pred[:, m], Y_val, tau))
            val_err_q.append(quantile_loss(qt_pred[:, m], Y_val, tau))
            valid_indices.append(m)  
        except Exception as e:
            print(e)
            continue
    if val_err_q:
        val_err_q = np.array(val_err_q)
        best_h_idx = int(np.argmin(val_err_q))
        no_clip_err = val_err_q[-1]
        if (no_clip_err - val_err_q[best_h_idx]) < 0.1 * no_clip_err:
            best_h_idx = len(val_err_q) - 1
        bvsigma_q = grid_vsigma[valid_indices[best_h_idx]]
    else:
        raise Exception("No valid bandwidth values found for qt.")

    if plot:
        print('minimum qt val error:', 
              val_err_q.min().round(4))
        plt.plot(grid_vsigma, val_err_q)
        plt.ylabel('validation error')
        plt.xlabel('vsigma')
        plt.title('qt')
        plt.show()

    return {'bvsigma_q':bvsigma_q, 'best_h_idx': best_h_idx}

def generate_dgp_data_with_test_sn(n_train, n_test=1024, d=8, tau=0.1, err_dist='normal', df=2.25,
                                 rho_x=0.8, rho_e=0.5, delta=None, eta=None, n_lag = 2,
                                 n_indep=5, err_scale=1.0, seed=None):
    """
    Generate time series data with autoregressive regressors and errors,
    including test set and quantile / expected shortfall calculation.

    Returns:
    - X_test: (n_test, d+1) test regressors
    - Y_test: (n_test,) test dependent variables
    - test_qt: (n_test,) true conditional quantiles
    - test_es: (n_test,) true conditional expected shortfall
    """
    rng = default_rng(seed)
    n_rm = 20
    n_total = n_train + n_test + n_rm

    # Set true coefficients
    if delta is None:
        delta = rng.choice([-2, 2], size=d+1)
  
    if eta is None:
        eta = rng.binomial(1, 0.5, size=d+1) * 0.25
        eta[:n_lag+1] = 0

    # Construct X matrix
    if d < 1:
        raise ValueError(f"d must be at least 1, but got {d}.")
    if n_lag < 0 or n_indep < 0:
        raise ValueError("n_lag and n_indep must be non-negative.")
    if n_lag + n_indep != d :
        raise ValueError(f"n_lag + n_indep must equal d (= {d}), but got n_lag={n_lag}, n_indep={n_indep}.")
    X = np.zeros((n_total, d))

    
    H = np.zeros((n_total, n_lag))
    eps_x = rng.standard_normal((n_total, n_lag))
    H[0] = eps_x[0]
    for i in range(1, n_total):
        H[i] = rho_x * H[i-1] + eps_x[i]

    X[:, :n_lag] = H
    # add independent variables
    if n_indep > 0:
        start_col = n_lag
        X[:, start_col:] = rng.uniform(
            1, 2, size=(n_total, n_indep)
        )    
    X = np.c_[np.ones(X.shape[0]), X[:]]
        
    # Generate AR(1) errors
    e = np.zeros(n_total)
    if err_dist == 'normal':
        nu = np.sqrt(1 - rho_e ** 2) * rng.normal(0, 1, n_total)
    elif err_dist == 't':
        nu = rng.standard_t(df, n_total) * np.sqrt((1 - rho_e ** 2) * (df - 2) / df)
    else:
        raise ValueError("err_dist must be 'normal' or 't'")

    e[0] = nu[0]
    for i in range(1, n_total):
        e[i] = rho_e * e[i - 1] + nu[i]

    # Generate Y
    Y = X @ delta[:(d+1)] + (X @ eta[:(d+1)]) * err_scale * e

    # Compute Q_tau(e_t) and ES_tau(e_t)
    if err_dist == 'normal':
        qt_scalar = norm.ppf(tau)
        # es_scalar = norm.expect(lambda x: x if x <= qt_scalar else 0) / tau
        es_scalar = -norm.pdf(norm.ppf(tau)) / tau
    elif err_dist == 't':
        qt_scalar, es_scalar = estimate_empirical_es_qt(tau=tau, df=df, rho=rho_e, seed=seed)
    else:
        raise ValueError("err_dist must be 'normal' or 't'")

    X_test = X[-(n_train + n_test):]
    Y_test = Y[-(n_train + n_test):]
    qt_e = qt_scalar * err_scale
    es_e = es_scalar * err_scale

    test_qt = X_test @ delta[:(d+1)] + (X_test @ eta[:(d+1)]) * qt_e
    test_es = X_test @ delta[:(d+1)] + (X_test @ eta[:(d+1)]) * es_e
    alpha = delta[:(d+1)] + eta[:(d+1)] * qt_e
    beta = delta[:(d+1)] + eta[:(d+1)] * es_e

    return X_test, Y_test, test_qt, test_es, alpha, beta, delta, eta
def estimate_empirical_es_qt(tau=0.1, df=3, rho=0.3, n_sim=100000, seed=None):
    rng = np.random.default_rng(seed)
    nu = rng.standard_t(df, size=n_sim) * np.sqrt((1 - rho ** 2) * (df - 2) / df)
    e = np.zeros(n_sim)
    e[0] = nu[0]
    for i in range(1, n_sim):
        e[i] = rho * e[i-1] + nu[i]
    q_tau = np.quantile(e, tau)
    es_tau = e[e <= q_tau].mean()
    return q_tau, es_tau



def CV_Qt_LP(X, Y, tau, kernel, grid_q, degree=1, nfolds=5, random_state=0):
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    cv_err = np.empty([len(grid_q), nfolds])
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        model = LocPoly(X_train, y_train, kernel=kernel)
        for m, h in enumerate(grid_q):
            pred_q = model.qt_predict(x0=X_test, bw=h, tau=tau, degree=degree)
            cv_err[m,k] = quantile_loss(pred_q, y_test, tau)
    cv_mean_err = cv_err.mean(axis=1)  
    return {'bw_q': grid_q[cv_mean_err.argmin()],
            'err': cv_mean_err}


def CV_ES_LP(X, Y, tau, kernel_q, grid_q, kernel_e, grid_e, 
             degree=1, nfolds=5, random_states=[0, 1]):
    cv_qt = CV_Qt_LP(X, Y, tau, kernel_q, grid_q, nfolds=nfolds, 
                     random_state=random_states[0])
    bw_q = cv_qt['bw_q']
    kf = KFold(n_splits=nfolds, random_state=random_states[1], shuffle=True)
    model = LocPoly(X, Y, kernel=kernel_q)
    fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree)

    cv_err = np.empty([len(grid_e), nfolds])
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        Y0 = np.minimum(y_train - fit_q[train_idx], 0)/tau + fit_q[train_idx]
        model2 = LocPoly(X_train, Y0, kernel=kernel_e)
        for m, h in enumerate(grid_e):
            pred_e = model2.ls_predict(x0=X_test, bw=h, degree=degree)
            Y0_val = np.minimum(y_test - fit_q[test_idx], 0)/tau + fit_q[test_idx]
            cv_err[m,k] = np.mean((Y0_val - pred_e)**2)
    cv_mean_err = cv_err.mean(axis=1)
    
    return {'bw_q': bw_q, 
            'bw_e': grid_e[cv_mean_err.argmin()],
            'err': cv_mean_err, 
            'model': model, 
            'fit_q': fit_q,
            'surrogate_y': np.minimum(Y - fit_q, 0)/tau + fit_q}