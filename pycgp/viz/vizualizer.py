import networkx as nx

def draw_net(ax, G,  n_input, n_output, node_size=2000, y_offset=0, node_color = 'lightblue', edge_color = 'black', node_shape='o', alpha=0.5, delta = 2):
    # TODO add diff color for input, + handle n_output
    pos = nx.spring_layout(G)

    pos = {}
    for i, node in enumerate(G.nodes()):
        if i < n_input:
            pos[node] =  (0, y_offset - (n_input/2 - i)*delta)
        else :      
            pos[node] = (i, y_offset )
    node_size=2000
    # nx.draw(G, pos, )
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_color=node_color, 
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