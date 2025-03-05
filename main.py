from pycgp import CGP, CGP_with_cste
from pycgp.cgp import CGPFunc
from pycgp.evaluators import SREvaluator
from pycgp.viz import draw_net, net_hist_validation
from pycgp.cgpfunctions import *
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import pandas
import seaborn as sn

from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import pmlb
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def fit_me(x):
    # return np.sin(x*x) + np.cos(0.5*x)
    # return 10+x 
    return np.sin(x)

def base(folder_name):
    
    Ntrain = 100
    x_train = np.random.uniform(-5, 5, (Ntrain, 1))
    sig = 0.01
    y_train = fit_me(x_train) + np.random.normal(0, sig, x_train.shape)

    
    # library = build_funcL

    e = SREvaluator(x_train=x_train, y_train=y_train, n_inputs=1, n_outputs=1, col=30, loss='mse')
    best, hist = e.evolve(num_csts=1, mu=4, nb_ind=8, mutation_rate_const_params=0.2, n_it = 500, folder_name=folder_name)
    input_name = ['x']
    output_name = ['y']


    print("cste table ", best.cst_table)
    sr = None
    sr = e.best_logs(input_name, output_name)
    try :
        sr = e.best_logs(input_name, output_name)
    except : pass
    
    # best.to_dot('best.dot', ['x'], ['y'])
    # os.system('dot -Tpdf ' + 'best.dot' + ' -o ' + 'best.pdf')
    # print("father genome : ", best.genome)
    x = np.linspace(-10, 10, 1000).reshape((1000,1))

    y_pred = best.run(x)[0]


    R2 =  metrics.r2_score(fit_me(x), y_pred)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # ax.plot(x_train[:,0], y_train, 'rx', label='train')
    # ax.plot(x, y_pred, 'b', label='res')
    # plt.legend()

    # ax = fig.add_subplot(1, 2, 2)
    # # plt.savefig("graph.png")
    print("best genome : ", best.genome)
    print("constante table ", best.cst_table)
    G = best.netx_graph(input_name, output_name, active=True, simple_label=False)
    # draw_net(ax, G, 2, 1, y_offset=100, node_color='red')
    # fig.suptitle("basic regression without constant R2 = " + str(R2) + "equation : " + str(sr[0]))

    net_hist_validation(G, hist, x_train[:,0], y_train, x, y_pred, 2, 1, title="basic regression R2 = " + str(R2) + "\n Equation : " + str(sr)) 

def roses(folder_name, col=70, row=1, nb_ind=8, mutation_rate_nodes=0.1, mutation_rate_outputs=0.2,
              n_cpus=1, n_it=500, genome=None):
    # "Deprecated for now"
    n = 1000
    A = 1.995633
    B = 1.27689
    C = 8
    r = np.linspace(0, 1, n)
    th = np.linspace(-2, 20*np.pi, n)
    x = r*np.cos(th)
    y = r*np.sin(th)
    [R ,THETA] = np.meshgrid(r, th)
    petal_number = 3.6 

    x = 1 - 0.5*( 1.25* (1-np.mod(petal_number*THETA, 2*np.pi)/np.pi)**2  - 0.25)**2
    phi = (np.pi/2)*np.exp(-THETA/(C*np.pi))
    y = A*(R**2)*(B*R-1)**2*np.sin(phi)

    R2 = x*(R*np.sin(phi) + y*np.cos(phi))
    X = R2*np.sin(THETA)
    Y = R2*np.cos(THETA)
    Z = x*(R*np.cos(phi) - y*np.sin(phi))
    shape_vec = (n*n, )
    y_train = np.array([X.ravel(), Y.ravel(), Z.ravel()])
    x_train = np.array([R.ravel(), THETA.ravel(),  np.pi*np.ones(shape_vec), A*np.ones(shape_vec), B*np.ones(shape_vec), C*np.ones(shape_vec), 0.25*np.ones(shape_vec)]) # take directly the ravel ? 

    # whole regression
    # library = build_funcLib()
    # e = SREvaluator(x_train=x_train, y_train=y_train, n_inputs=7, n_outputs=3, col=col, row=row, library=library, loss='mae')
    # e.evolve(nb_ind=nb_ind, n_it=10000)

    # e.best_logs(["r", "th", 'PI', 'A', 'B', 'C', 'Quart'], ["X", "Y", "Z"])


    # regression on x
    y_train = np.array(x.ravel())
    x_train = np.array([R.ravel(), THETA.ravel(),  np.pi*np.ones(shape_vec), A*np.ones(shape_vec), B*np.ones(shape_vec), C*np.ones(shape_vec), 0.25*np.ones(shape_vec), petal_number*np.ones(shape_vec)]) # take directly the ravel ? 
    library = build_funcLib()
    # e = SREvaluator(x_train=x_train, y_train=y_train, n_inputs=8, n_outputs=1, col=col, row=row, library=library, loss='mse')
    # e.evolve(n_it=1000, nb_ind=8)

    # e.best_logs(["r", "th", 'PI', 'A', 'B', 'C', 'Quart', 'Pn'], ['y'])

    red_colormap = {
        'red':   ((0.0, 0.5, 0.1),
                  (1.0, 0.8, 0.8)),
        'green': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))
    }
    roses_cm = LinearSegmentedColormap('roses', red_colormap)
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap = 'autumn')
    ax.grid(False)
    ax.set_facecolor('white')

    # y_cgp = e.best_evaluate(x_train)
    # ax = fig.add_subplot(1,2,2, projection='3d')
    # ax.plot(R.ravel(), THETA.ravel(), x.ravel(), 'b.')
    # ax.plot(R.ravel(), THETA.ravel(), y_cgp.ravel(), 'r.')
    
    # ax.tick_params(left = False, right = False , labelleft = False , 
    #             labelbottom = False, bottom = False) 
    # # ax.set_axis_off()
    # ax = fig.add_subplot(1,2,2, projection='3d')
    # y_cgp = e.best_evaluate(x_train).reshape(3,n,n)

    # ax.plot_surface(y_cgp[0], y_cgp[1], y_cgp[2])
    # ax.plot_surface(y_train[0,:].reshape(n,n), y_train[1,:].reshape(n,n), y_train[2,:].reshape(n,n))    
    plt.show()


def winequality():
    dataset_path = f'datasets/winequality/winequality-red.csv'
    df = pandas.read_csv(dataset_path)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    # shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # labels = df.columns

    # ax.matshow(df.corr(method='pearson'))

    # ax.set_xticks([''] + list(df.columns))
    sn.heatmap(df.corr('pearson'), ax=ax)
    plt.xticks(rotation=45)
    # plt.colorbar()
    # plt.show()
    # scaler = StandardScaler()
    col_2keep = ['fixed acidity', 'volatile acidity', 'citric acid', 'sulphates',  'free sulfur dioxide', 'alcohol']
    col_2keep = list(df.columns)[:-1]
    split_index = int(0.8 * len(df))
    train_df = df[:split_index][col_2keep].to_numpy()
    test_df = df[split_index:][col_2keep].to_numpy()
    
  
    x_train = train_df[:,:-1]
    # x_train = scaler.fit_transform(x_train)
    y_train = train_df[:, -1]

    
    x_test = test_df[:,:-1]
    y_test = test_df[:, -1]
    
    wine_lib = [CGP.CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGP.CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGP.CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGP.CGPFunc(f_exp, 'exp', 1, 0, 'exp'),
            CGP.CGPFunc(f_log, 'log', 1, 0, 'ln'),
            CGP.CGPFunc(f_sqrt, 'sqrt', 1, 0, 'sqrt'),
            CGP.CGPFunc(f_div, 'div', 2, 1, '/'),
            CGP.CGPFunc(f_floor, 'floor', 2, 0, '_'),
            CGP.CGPFunc(f_ceil, 'ceil', 2, 0, '~'),
            CGP.CGPFunc(f_const, 'c', 0, 1, 'c')

            ]
    wine_evaluator = SREvaluator(x_train, y_train, n_inputs=x_train.shape[-1], n_outputs=1, col=min(x_train.shape)*10, library=wine_lib, loss='mse')

    hof, hist = wine_evaluator.evolve(mu=4 ,nb_ind=16, n_it=1000, folder_name='winequality')
    ax = fig.add_subplot(1,3,2)
    # input_name = ['fa', 'va', 'ca', 'su', 'fs', 'al']
    input_name = ['fa', 'va', 'ca', 'rs', 'chl', 'fs', 'ts', 'd', 'pH', 'sl', 'al']

    output_name = ['q']    


    try :  
        # input_name = ['fa', 'va', 'ca', 'rs', 'chl', 'fs', 'ts', 'd', 'pH', 'sl', 'al']


        wine_evaluator.best_logs(input_names=input_name, output_names=output_name)
    except Exception as e:
        print('simpy failed ', e)
    
    ax.plot(hist, label='fitness in training')



    G = hof.netx_graph(input_name, output_name, active=True, simple_label=True)
    fig.add_subplot(1,3,3)
    draw_net(G, x_train.shape[-1], 1 ,y_offset=100, node_color='red')
    
    
    R2 = metrics.r2_score(y_test, hof.run(x_test)[0].reshape(y_test.shape))
    print("R2 score : ", R2)
    plt.show()
    # y = df['quality'].to_numpy()
    # x = df.to_numpy()[:,:-1]

def strogatz_glider1():
    '''
        x' = -0.05 * x**2 - sin(y)
    '''
    # EAs paramaters  
    l = 16
    m = 8
    # get data using pmlb
    pdata = pmlb.fetch_data("strogatz_glider1")

    pdata = pdata.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # add constant column
    # pdata['cst'] = c*np.ones(len(pdata))
    y = pdata['target'].to_numpy()
    x = pdata[["x", "y", ]].to_numpy()

    split_ind = round(len(pdata) * 0.5 )
    xtrain = x[:split_ind] 
    ytrain = y[:split_ind]

    xtest = x[split_ind:]
    ytest = y[split_ind:]

    lib = [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            # CGPFunc(f_exp, 'exp', 1, 0, 'exp'),
            # CGPFunc(f_log, 'log', 1, 0, 'ln'),
            # CGP.CGPFunc(f_sqrt, 'sqrt', 1, 0, 'sqrt'),
            CGPFunc(f_div, 'div', 2, 0, '//'),
            CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGPFunc(f_cos, 'cos', 1, 0, '   cos'),
            CGPFunc(f_const, 'c', 0, 1, 'c'),
            # CGPFunc(f_squared, 'square', 1 , '^2')
            ]
    

    evaluator = SREvaluator(xtrain, ytrain, n_inputs=2, n_outputs=1, col=30, library=lib, loss='mae')
    
    hof, hist = evaluator.evolve(mu=m, nb_ind=l, num_csts=1, mutation_rate_nodes=0.4, mutation_rate_outputs=0.1, n_it=5000)


    input_names = ['x', 'y']
    output_names = ['yp']   
    G = hof.netx_graph(input_names, output_names, active=True, simple_label=False)


    eq = ''
    try:
        eq = evaluator.best_logs(input_names,output_names)
    except Exception as e:
        print(e)

    # fig = plt.figure()
    # xtest = np.sort(xtest, axis=0)
    y_sr = hof.run(xtest)[0].reshape(ytest.shape)


    R2 = metrics.r2_score(ytest, y_sr)


    # y_sr = hof.run(xtrain)[0].reshape(ytrain.shape)
    
    title = "R2 score : " + str(R2) + '\n equation : ' +  str(eq)
    
    net_hist_validation(G, hist, xtest[:, :2], ytest,  xtest[:, :2], y_sr, 2, 1, title=title)
    
def regression_benchmark():
    base('basic')   
    strogatz_glider1()
    plt.show()

if __name__ == '__main__':
    # roses('roses')
    regression_benchmark()
    