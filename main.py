from pycgp import CGP, CGP_with_cste
from pycgp.cgp import CGPFunc
from pycgp.evaluators import SREvaluator
from pycgp.viz import draw_net, net_hist_validation
import matplotlib.gridspec as gridspec
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
import yaml
import pickle


# Suppress all warnings
warnings.filterwarnings("ignore")

PAR_FILE = 'cgp_parameters.yaml'

def read_parameters(par_file):
    with open(par_file, 'r') as file:
        data = yaml.safe_load(file)
    return data
    
def fit_me(x):
    return np.sin(x*x) + np.cos(0.5*x)
    # return 10+x 
    # return np.sin(x)

def base(folder_name):
    
    Ntrain = 200
    x_train = np.random.uniform(-10, 10, (Ntrain, 1))
    sig = 0.001
    y_train = fit_me(x_train) + np.random.normal(0, sig, x_train.shape)

    
    # library = build_funcL

    
    
    data = read_parameters(PAR_FILE)
    e = SREvaluator(x_train=x_train, y_train=y_train, n_inputs=1, n_outputs=1, col=data['col'], loss='mse')
    best, hist = e.evolve(num_csts=1, mu=data['mu'], nb_ind=data['lbd'], mutation_rate_nodes=data['m_node'],
                           mutation_rate_outputs=data['m_output'], mutation_rate_const_params=data['m_const'], n_it = data['n_gen'], 
                           folder_name=folder_name)
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
    y_test = fit_me(x)
    y_pred = best.run(x)[0]


    R2 =  metrics.r2_score(y_test, y_pred)
    G = best.netx_graph(input_name, output_name, active=True, simple_label=False)
    # draw_net(ax, G, 2, 1, y_offset=100, node_color='red')
    # fig.suptitle("basic regression without constant R2 = " + str(R2) + "equation : " + str(sr[0]))

    net_hist_validation(G, hist, x[:,0], y_test, x, y_pred, 1, 1, title="basic regression R2 = " + str(R2) + "\n Equation : " + str(sr)) 
    return R2, hist

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
    # shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # labels = df.columns

    # ax.matshow(df.corr(method='pearson'))

    # ax.set_xticks([''] + list(df.columns))
    # sn.heatmap(df.corr('pearson'), ax=ax)
    # plt.xticks(rotation=45)
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
    
    wine_lib = [CGPFunc(f_sum, 'sum', 2, 0, '+'),
                CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
                CGPFunc(f_mult, 'mult', 2, 0, '*'),
                CGPFunc(f_exp, 'exp', 1, 0, 'exp'),
                CGPFunc(f_log, 'log', 1, 0, 'ln'),
                CGPFunc(f_sqrt, 'sqrt', 1, 0, 'sqrt'),
                CGPFunc(f_div, 'div', 2, 1, '/'),
                # CGPFunc(f_floor, 'floor', 2, 0, '_'),
                # CGPFunc(f_ceil, 'ceil', 2, 0, '~'),
                CGPFunc(f_const, 'c', 0, 1, 'c')]


    wine_evaluator = SREvaluator(x_train, y_train, n_inputs=x_train.shape[-1], n_outputs=1, col=200, library=wine_lib, loss='mae')

    hof, hist = wine_evaluator.evolve(mu=10 ,nb_ind=30, num_csts=4, n_it=5000, folder_name='winequality')

    # input_name = ['fa', 'va', 'ca', 'su', 'fs', 'al']
    input_name = ['fa', 'va', 'ca', 'rs', 'chl', 'fs', 'ts', 'd', 'pH', 'sl', 'al']

    output_name = ['q']    


    try :  
        # input_name = ['fa', 'va', 'ca', 'rs', 'chl', 'fs', 'ts', 'd', 'pH', 'sl', 'al']


        wine_evaluator.best_logs(input_names=input_name, output_names=output_name)
    except Exception as e:
        print('simpy failed ', e)
    
    
    
    ax = fig.add_subplot(1,3,1)
    ax.plot(hist, label='fitness in training')

    G = hof.netx_graph(input_name, output_name, active=True, simple_label=True)
    ax = fig.add_subplot(1,3,3)
    draw_net(ax, G, x_train.shape[-1], 1 ,y_offset=100, node_color='red')
    
    
    R2 = metrics.r2_score(y_test, hof.run(x_test)[0].reshape(y_test.shape))

    fig.suptitle('R2 score : ' + str(R2))

    plt.show()
    # y = df['quality'].to_numpy()
    # x = df.to_numpy()[:,:-1]

def strogatz_glider1():
    '''
        x' = -0.05 * x**2 - sin(y)
    '''


    # TODO the equation got from genes and from CGP give different values investigate ! 
    
    data = read_parameters(PAR_FILE)
    # EAs paramaters  
    l = data['lbd']
    m = data['mu']

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
    

    evaluator = SREvaluator(xtrain, ytrain, n_inputs=2, n_outputs=1, col=data['col'], library=lib, loss='mae')
    
    hof, hist = evaluator.evolve(mu=m, nb_ind=l, num_csts=1, mutation_rate_nodes=data['m_node'], mutation_rate_outputs=data['m_output'], mutation_rate_const_params=data['m_const'], n_it=data['n_gen'], folder_name='strogatz_glider1')


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
    return R2, hist

def mars_express():
    """Mars Express Benchmark constante esitmation
    https://zenodo.org/records/6417900
    """
    # read parameters 
    data = read_parameters(PAR_FILE)
    # EAs paramaters  
    l = data['lbd']
    m = data['mu']
    m_node = data['m_node']
    m_out = data['m_output']
    m_const = data['m_const']
    n_gen = data['n_gen']
    col = data['col']
    row = data['row']

    # read data from csv 
    input_col = ['lvah', 'sh', 'eclipse_l' , 'average_xtx', 'full_off', 'right_flag' ]
    output_col = ['average_power']
    train_data = pandas.read_csv("datasets/MEX/MEX1.csv")
    test_data = pandas.read_csv("datasets/MEX/MEX2.csv")


    ytrain = train_data[output_col].to_numpy()
    xtrain = train_data[input_col].to_numpy()

    ytest = test_data[output_col].to_numpy()
    xtest = test_data[input_col].to_numpy()
    
    # lib definition
    lib = [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGPFunc(f_div, 'div', 2, 0, '//'),
            CGPFunc(f_log, 'log', 1, 0, 'log'),
            CGPFunc(f_const, 'c', 0, 1, 'c'),
            ]
    
    # Evaluator definition
    evaluator = SREvaluator(xtrain, ytrain, n_inputs=min(xtrain.shape), n_outputs=1, col=col, library=lib, loss='rmse')
    hof, hist = evaluator.evolve(mu=m, nb_ind=l, num_csts=7, mutation_rate_nodes=m_node, mutation_rate_outputs=m_out, mutation_rate_const_params=m_const, n_it=n_gen, folder_name='mars_express')

    eq = ''
    try:
        eq = evaluator.best_logs(input_col, output_col)
    except Exception as e:
        pass
    G = hof.netx_graph(input_col, output_col, active=True, simple_label=False)
    
    # validation
    y_cgp = hof.run(xtest)[0].reshape(ytest.shape)
    R2 = metrics.r2_score(ytest, y_cgp)
    title = "R2 score : " + str(R2) + '\n equation : ' +  str(eq)
    print("eq ", title )    

    # plot data 
    fdata = pandas.concat([train_data, test_data])
    xdata = fdata[input_col].to_numpy()
    orbit_time = fdata['timestamp'].to_numpy()
    a_p = fdata['average_power'].to_numpy()
    thp2 = fdata['thp2'].to_numpy()
    ycgp_data = hof.run(xdata)[0]


    fig = plt.figure(figsize=(10, 10))
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

    # net sub
    ax1 = fig.add_subplot(gs[0, :])
    draw_net(ax1, G, min(xtrain.shape), 1, node_shape='o')

    fig.suptitle(title)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(hist)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('fitness')
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])

    ax3.plot(orbit_time, a_p, 'r.', label='average_power')
    ax3.plot(orbit_time, ycgp_data, 'g.', label='pred power' )
    ax3.plot(orbit_time, thp2, 'b.', label='reg model')
    ax3.axvline(train_data['timestamp'].to_numpy()[-1], linestyle='--',  linewidth=2, color='k')
    ax3.legend()
    ax3.grid(True)
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(orbit_time, a_p, 'r.', label='average_power')
    ax.plot(orbit_time, ycgp_data, 'g.', label='pred power' )
    ax.plot(orbit_time, thp2, 'b.', label='thp2 model')
    ax.axvline(train_data['timestamp'].to_numpy()[-1], linestyle='--', linewidth = 2, color='k')
    ax.legend()
    ax.set_title('average power prediction')
    ax.set_xlabel("timestamps in [s]")
    ax.set_ylabel('average power in [W]')
    ax.grid(True)

    return R2, hist 


def regression_benchmark():

    N = 10
    R2s = []
    hists = []
    # print("------------- SIN APPROX -------------------")
    # for i in range(N):
    #     R2, h = base('basic')   
    #     R2s.append(R2)
    #     hists.append(h)

    # np.save("R2s_complex.npy", R2s)

    # with open('hists_complex.pkl','wb') as f:
    #     pickle.dump(hists, f)

    # print("------------- Strogatz APPROX -------------------")

    
    # N = 10
    # R2s = []
    # hists = []
    # for i in range(N):
    #    R2, h = strogatz_glider1()
    #    R2s.append(R2)
    #    hists.append(h)

    # np.save("R2s_strogatz.npy", R2s)
    # with open('hists_strogatz.pkl','wb') as f:
    #     pickle.dump(hists, f)


    N = 10
    R2s = []
    hists = []
    for i in range(N):
       R2, h = mars_express()
       R2s.append(R2)
       hists.append(h)
    
    np.save("R2s_mars.npy", R2s)
    np.save("hists_mars.npy", hists)
     
    # winequality()
    # plt.show()

if __name__ == '__main__':
    # roses('roses')
    
    regression_benchmark()
    plt.show()
    