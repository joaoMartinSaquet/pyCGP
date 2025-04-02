import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def draw_net(ax, G,  n_input, n_output, node_size=2000, y_offset=0, node_color = '#99ff99', edge_color = 'black', node_shape='o', alpha=0.5, delta = 20):
    # TODO add diff color for input, + handle n_output
    pos = nx.spring_layout(G)

    pos = {}
    n_color = []
    k = 0
    # output_idx = 0
    for i, node in enumerate(G.nodes()):
        if i < n_input:
            if n_input % 2 == 0:
                t = n_input / 2
            else: 
                t = 0
            pos[node] =  (0, y_offset + (n_input/2 - (i if i < t else (i + 1)))*delta)
            n_color.append('#99ccff')
        elif i > (len(G) -1) - n_output :              
            pos[node] = (len(G) - n_output, y_offset + (k - 1/n_output)*delta )
            n_color.append('#ff9999')
            k += 1
        else:
            pos[node] = (i, y_offset )
            n_color.append(node_color)

    node_size=2000
    # nx.draw(G, pos, )
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_color=n_color, 
        edgecolors=None,
        node_size=node_size, 
        node_shape=node_shape, 
        alpha=alpha)
    node_radius = (node_size ** 0.5) / 2 
    for i, (u, v, k) in enumerate(G.edges(keys=True)):  # Correct way to iterate edges in MultiDiGraph
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], 
            connectionstyle=f'arc3,rad={(0.2 if k == 0 else -0.2)}', 
            edge_color='black', 
            arrowstyle='-|>',  # Ensure arrowheads are shown
            arrowsize=20,  # Increase arrow size for visibility
            min_source_margin=node_radius,  # Offset arrow start from node
            min_target_margin=node_radius,  # Offset arrow end from node
        
        )


    nx.draw_networkx_labels(G, pos, font_size=16, font_color='black')


def net_hist_validation(net, hist, val_x, val_y, x_cgp, y_cgp, n_input, n_output, title=None):

    fig = plt.figure(figsize=(8, 6))
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

    # net sub
    ax1 = fig.add_subplot(gs[0, :])
    draw_net(ax1, net, n_input, n_output, node_shape='o')

    # hist sub
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(hist)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('fitness')
    ax2.grid(True)

    n_x = len(val_x.shape)

    # val sub
    if n_x == 1:
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(val_x, val_y)
        ax3.plot(x_cgp, y_cgp, 'r', linewidth=2)
        ax3.set_ylabel('y')
        ax3.set_xlabel('x')
        ax3.grid(True)
    elif n_x == 2:
        ax3 = fig.add_subplot(gs[1, 1], projection='3d')
        ax3.scatter(val_x[:, 0], val_x[:, 1], val_y)
        ax3.plot(x_cgp[:, 0], x_cgp[:, 1], y_cgp, 'r.', linewidth=2)
        ax3.set_ylabel('y')

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    