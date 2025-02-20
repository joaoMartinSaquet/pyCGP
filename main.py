from pycgp import CGP
from pycgp.evaluators import SREvaluator
from pycgp.viz import draw_net
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


def build_funcLib():
    return [CGP.CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGP.CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            # CGP.CGPFunc(f_mult, 'mult', 2, 0, '*'),
            # CGP.CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            # CGP.CGPFunc(f_cos, 'cos', 1, 0, 'cos'),
            # CGP.CGPFunc(f_exp, 'exp', 1, 0, 'exp'),
            CGP.CGPFunc(f_mod, 'mod', 2, 0, '%'),
            CGP.CGPFunc(f_div, 'div', 2, 0, '/')
            ]
def fit_me(x):
    return np.sin(x*x) + np.cos(x)

def base(folder_name, nb_ind=8, mutation_rate_nodes=0.2, mutation_rate_outputs=0.2,
              n_cpus=1, n_it=1000, genome=None):
    
    Ntrain = 100
    x_train = np.random.uniform(-5, 5, (Ntrain, 1))
    sig = 0.0
    y_train = fit_me(x_train) + np.random.normal(0, sig, x_train.shape)
    # library = build_funcLib()
    e = SREvaluator(x_train=x_train, y_train=y_train, n_inputs=1, n_outputs=1, loss='mse')
    
    best, hist = e.evolve(nb_ind, mutation_rate_nodes, mutation_rate_outputs, n_cpus, n_it, folder_name)
    
    e.best_logs(['x'], ['y'])

    
    # best.to_dot('best.dot', ['x'], ['y'])
    # os.system('dot -Tpdf ' + 'best.dot' + ' -o ' + 'best.pdf')
    # print("father genome : ", best.genome)
    x = np.linspace(-10, 10, 1000).reshape((1000,1))

    y_pred = best.run(x)[0]
    plt.plot(x_train, y_train, 'rx', label='train')
    plt.plot(x, y_pred, 'b', label='res')
    # plt.savefig("graph.png")
    plt.figure()
    G = best.netx_graph(['x'], ['y'], active=True, simple_label=False)
    draw_net(G, 1, 1, y_offset=100, node_color='red')
    G = best.netx_graph(['x'], ['y'], active=False, simple_label=False)
    draw_net(G, 1, 1, y_offset=400, node_shape='s')
    print("R2 score : ", metrics.r2_score(fit_me(x), y_pred))
    plt.figure()
    plt.plot(hist)
    plt.show()

# def load(file_name):
#     print('loading ' + file_name)
#     library = build_funcLib()
#     c = CGP.load_from_file(file_name, library)
#     print(e.evaluate(c, 0, displayTrace=True))
def roses(folder_name, col=70, row=1, nb_ind=8, mutation_rate_nodes=0.1, mutation_rate_outputs=0.2,
              n_cpus=1, n_it=500, genome=None):

    n = 100
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
    e = SREvaluator(x_train=x_train, y_train=y_train, n_inputs=8, n_outputs=1, col=col, row=row, library=library, loss='mse')
    e.evolve(n_it=1000, nb_ind=8)

    e.best_logs(["r", "th", 'PI', 'A', 'B', 'C', 'Quart', 'Pn'], ['y'])

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
    ax.plot_surface(X, Y, Z, cmap=roses_cm)
    ax.grid(False)
    ax.set_facecolor('white')

    y_cgp = e.best_evaluate(x_train)
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot(R.ravel(), THETA.ravel(), x.ravel(), 'b.')
    ax.plot(R.ravel(), THETA.ravel(), y_cgp.ravel(), 'r.')
    
    # ax.tick_params(left = False, right = False , labelleft = False , 
    #             labelbottom = False, bottom = False) 
    # # ax.set_axis_off()
    # ax = fig.add_subplot(1,2,2, projection='3d')
    # y_cgp = e.best_evaluate(x_train).reshape(3,n,n)

    # ax.plot_surface(y_cgp[0], y_cgp[1], y_cgp[2])
    # ax.plot_surface(y_train[0,:].reshape(n,n), y_train[1,:].reshape(n,n), y_train[2,:].reshape(n,n))    
    plt.show()

def toDot(file_name, out_name):
    print('Exporting ' + file_name + ' in dot ' + out_name + '.dot')
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    c.to_dot(out_name+'.dot', ['x'], ['y'])
    print('Converting dot file into pdf in ' + out_name + '.pdf')
    os.system('dot -Tpdf ' + out_name + '.dot' + ' -o ' + out_name + '.pdf')

def displayFunctions(file_name):
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    c.to_function_string(['x'], ['y'])



def sr_benchmark():
    pass

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
            CGP.CGPFunc(f_ceil, 'ceil', 2, 0, '~')

            ]
    wine_evaluator = SREvaluator(x_train, y_train, n_inputs=x_train.shape[-1], n_outputs=1, col=min(x_train.shape)*10, library=wine_lib, loss='mse')

    hof, hist = wine_evaluator.evolve(nb_ind=16, n_it=1000, mutation_rate_nodes=0.2, mutation_rate_outputs=0.2, folder_name='winequality')
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
    # get data using pmlb
    pdata = pmlb.fetch_data("strogatz_glider2")
    pdata = pdata.sample(frac=1, random_state=42).reset_index(drop=True)
    y = pdata['target'].to_numpy()
    x = pdata[["x", "y"]].to_numpy()

    split_ind = round(len(pdata) * 0.5 )
    xtrain = x[:split_ind]
    ytrain = y[:split_ind]

    xtest = x[split_ind:]
    ytest = y[split_ind:]

    lib = [CGP.CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGP.CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGP.CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGP.CGPFunc(f_exp, 'exp', 1, 0, 'exp'),
            CGP.CGPFunc(f_log, 'log', 1, 0, 'ln'),
            CGP.CGPFunc(f_sqrt, 'sqrt', 1, 0, 'sqrt'),
            CGP.CGPFunc(f_div, 'div', 2, 0, '//'),
            CGP.CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGP.CGPFunc(f_cos, 'cos', 1, 0, '   cos')
            # CGP.CGPFunc(f_floor, 'floor', 2, 0, '_'),
            # CGP.CGPFunc(f_ceil, 'ceil', 2, 0, '~')

            ]
    

    evaluator = SREvaluator(xtrain, ytrain, n_inputs=2, n_outputs=1, col=60, library=lib, loss='mse')
    
    hof, hist = evaluator.evolve(nb_ind=16, n_it=3000)


    try:
        evaluator.best_logs(['x', 'y'], ['yp'])
    except Exception as e:
        print(e)



    fig = plt.figure()

    ax = fig.add_subplot(1,3,1, projection='3d')

    ax.plot(xtest[:,0], xtest[:,1], ytest, '.', label='true data')
    
    y_sr = hof.run(xtest)

    ax.plot(xtest[:,0], xtest[:,1], y_sr, 'r.', label='cgp regression')
    print("validation R2 : ", metrics.r2_score(ytest, y_sr.reshape(ytest.shape)))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(hist)
    





    plt.show()


if __name__ == '__main__':
    # print (sys.argv)
    # if len(sys.argv) == 1:
    #     print('Starting for roses')
    #     # base('test')
    #     roses('test')
    # if len(sys.argv)==2:
    #     print('Starting evolution from random genome')
    #     evolve(sys.argv[1])
    # elif len(sys.argv)==3:
    #     print('Starting evolution from genome saved in ', sys.argv[2])
        # evo(sys.argv[1], genome=sys.argv[2])
    winequality()
    # base('test')
    strogatz_glider1()