import random

def random_newick( n : int ) -> str :
    '''
    Generate a random phylogenetic tree in NEWICK format with n leaf nodes.
    
    Arguments :
        n: Number of leaf nodes (must be >= 1)
    
    Returns :
        String representation of tree in NEWICK format
    '''
    
    if n < 1 :
        raise ValueError( 'Number of leaves must be at least 1' )

    # handle the trivial case
    if n == 1 :
        return 'leaf_1;'
    
    # each node is represented as: [newick_string, is_leaf]
    # start with n leaf nodes
    nodes = [ [ f'leaf_{i+1}', True ] for i in range(n) ]
    
    # continue until we have only one node (the root)
    while len(nodes) > 1 :
        # randomly select two nodes to join
        id1   = random.randint( 0, len(nodes) - 1 )
        node1 = nodes.pop( id1 )
        id2   = random.randint( 0, len(nodes) - 1 )
        node2 = nodes.pop( id2 )
        
        # create parent node with random branch lengths
        branch_length1 = round( random.uniform( 0.1, 2.0 ), 4 )
        branch_length2 = round( random.uniform( 0.1, 2.0 ), 4 )
        
        parent_newick = f'({node1[0]}:{branch_length1},{node2[0]}:{branch_length2})'
        
        # add parent back to list
        nodes.append( [ parent_newick, False ] )
    
    # don't forget the semicolon
    return nodes[0][0] + ';'
