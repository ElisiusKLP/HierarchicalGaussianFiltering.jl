"""

Function for checking if the specified HGF structure is valid
"""
function check_hgf(hgf::HGFStruct)

    ## Check for duplicate names
    #Get node names
    node_names = getfield.(hgf.ordered_nodes.all_nodes, :name)
    #If there are any duplicate names
    if length(node_names) > length(unique(node_names))
        #Throw an error
        throw(ArgumentError("Some nodes have been given identical names. This is not supported"))
    end

    ## Check each node
    for node in hgf.ordered_nodes.all_nodes
        check_hgf(node)
    end
end

"""
    check_hgf(node::ContinuousStateNode)

Function for checking the validity of a single continous state node
"""
function check_hgf(node::ContinuousStateNode)

    #Extract node name for error messages
    node_name = node.name

    #Disallow binary input node value children
    if any(isa.(node.value_children, BinaryInputNode))
        throw(
            ArgumentError(
                "The continuous state node $node_name has a value child which is a binary input node. This is not supported.",
            ),
        )
    end

    #Disallow binary input node volatility children
    if any(isa.(node.volatility_children, BinaryInputNode))
        throw(
            ArgumentError(
                "The continuous state node $node_name has a volatility child which is a binary input node. This is not supported.",
            ),
        )
    end

    #Disallow having volatility children if a value child is a continuous inputnode 
    if any(isa.(node.value_children, ContinuousInputNode))
        if length(node.volatility_children) > 0
            throw(
                ArgumentError(
                    "The state node $node_name has a continuous input node as a value child. It also has volatility children, which disrupts the update order. This is not supported.",
                ),
            )
        end
    end

    return nothing
end

"""
    check_hgf(node::BinaryStateNode)

Function for checking the validity of a single binary state node
"""
function check_hgf(node::BinaryStateNode)

    #Extract node name for error messages
    node_name = node.name

    #Require exactly one value parent
    if length(node.value_parents) != 1
        throw(
            ArgumentError(
                "The binary state node $node_name does not have exactly one value parent. This is not supported.",
            ),
        )
    end

    #Require exactly one value child
    if length(node.value_children) != 1
        throw(
            ArgumentError(
                "The binary state node $node_name does not have exactly one value child. This is not supported.",
            ),
        )
    end
    
    #Allow only binary input node children
    if any(.!isa.(node.value_children, BinaryInputNode))
        throw(
            ArgumentError(
                "The binary state node $node_name has a child which is not a binary input node. This is not supported.",
            ),
        )
    end

    #Allow only continuous state node node parents
    if any(.!isa.(node.value_parents, ContinuousStateNode))
        throw(
            ArgumentError(
                "The binary state node $node_name has a parent which is not a continuous state node. This is not supported.",
            ),
        )
    end

    return nothing
end

"""
    check_hgf(node::ContinuousInputNode)

Function for checking the validity of a single continuous input node
"""
function check_hgf(node::ContinuousInputNode)

    #Extract node name for error messages
    node_name = node.name

    #Allow only continuous state node node parents
    if any(.!isa.(node.value_parents, ContinuousStateNode))
        throw(
            ArgumentError(
                "The continuous input node $node_name has a parent which is not a continuous state node. This is not supported.",
            ),
        )
    end

    #Require at least one value parent
    if length(node.value_parents) == 0
        throw(
            ArgumentError(
                "The input node $node_name does not have any value parents. This is not supported.",
            ),
        )
    end

    #Disallow multiple value parents if there are volatility parents
    if length(node.volatility_parents) > 0
        if length(node.value_parents) > 1
            throw(
                ArgumentError(
                    "The input node $node_name has multiple value parents and at least one volatility parent. This is not supported.",
                ),
            )
        end
    end

    return nothing
end

"""
    check_hgf(node::BinaryInputNode)

Function for checking the validity of a single binary input node
"""
function check_hgf(node::BinaryInputNode)

    #Extract node name for error messages
    node_name = node.name

    #Require exactly one value parent
    if length(node.value_parents) != 1
        throw(
            ArgumentError(
                "The binary input node $node_name does not have exactly one value parent. This is not supported.",
            ),
        )
    end

    #Allow only binary state nodes as parents
    if any(.!isa.(node.value_parents, BinaryStateNode))
        throw(
            ArgumentError(
                "The binary input node $node_name has a parent which is not a binary state node. This is not supported.",
            ),
        )
    end

    return nothing
end
