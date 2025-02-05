"""
get_surprise(hgf::HGF, node_name::String = "u")

Calculates the surprisal at the last input of a specified input node in an HGF. If an agent is passed instead of an HGF, the HGF is extracted from the substruct in the agent.
If no node is specified, the surprisal of all input nodes is summed.
"""
function get_surprise end

##Specific node
function get_surprise(hgf::HGF, node_name::String)

    #Get out the input node
    node = hgf.input_nodes[node_name]

    #Calculate its surprise
    return get_surprise(node)
end

function get_surprise(agent::Agent, node_name::String)

    #Get prediction form the HGF
    surprise = get_surprise(agent.substruct, node_name)

    return surprise
end

##Sum all surprises
function get_surprise(hgf::HGF)

    #Initialize surprise counter
    surprise = 0

    #Go through each input node
    for node in hgf.ordered_nodes.input_nodes
        #Sum their surprises
        surprise += get_surprise(node)
    end

    return surprise
end

function get_surprise(agent::Agent)

    #Get prediction form the HGF
    surprise = get_surprise(agent.substruct)

    return surprise
end

### Single node functions ###
@doc raw"""
    get_surprise(node::ContinuousInputNode)

Calculate the surprise of an input node on seeing the last input.

Equation:
``\hat{\mu}'_j={\sum_{j=1}^{j\;value\;parents} \hat{\mu}_j}``

`` \Im= -log(pdf(\mathcal{N}(\hat{\mu}'_j, \hat{\pi}_j), u))``
"""
function get_surprise(node::ContinuousInputNode)

    #If there was no input
    if ismissing(node.states.input_value)
        #Return no surprise
        return 0
    end

    #Sum the predictions of the vaue parents
    parents_prediction_mean = 0
    for parent in node.edges.observation_parents
        parents_prediction_mean += parent.states.prediction_mean
    end

    #Get the surprise
    -log(
        pdf(
            Normal(parents_prediction_mean, node.states. prediction_precision),
            node.states.input_value,
        ),
    )
end

"""
    get_surprise(node::BinaryInputNode)

Calculate the surprise of a binary input node on seeing the last input.

"""

function get_surprise(node::BinaryInputNode)

    #If there was no input
    if ismissing(node.states.input_value)
        #Return no surprise
        return 0
    end

    #Sum the predictions of the vaue parents
    parents_prediction_mean = 0
    for parent in node.edges.observation_parents
        parents_prediction_mean += parent.states.prediction_mean
    end

    #If a 1 was observed
    if node.states.input_value == 0
        #Get surprise
        surprise = -log(1 - parents_prediction_mean)

        #If a 0 was observed
    elseif node.states.input_value == 1
        #Get surprise
        surprise = -log(parents_prediction_mean)
    end

    return surprise
end

"""
    get_surprise(node::CategoricalInputNode)

Calculate the surprise of a categorical input node on seeing the last input.
"""
function get_surprise(node::CategoricalInputNode)

    #If there was no input
    if ismissing(node.states.input_value)
        #Return no surprise
        return 0
    end

    #Get value parent
    parent = node.edges.observation_parents[1]

    #Get surprise
    surprise = sum(-log.(exp.(log.(parent.states.prediction) .* parent.states.posterior)))

    return surprise
end


""" ### MODIFICATION ### """
# get surprise of a specific family
function get_surprise_by_family(node::AbstractInputNode, family_name::String)

    # If no input was provided, return no surprise
    if ismissing(node.states.input_value)
        return 0
    end

    # collect family input Nodes
    # Collect parents belonging to the specified family
    relevant_parents = filter(
        parent -> family_name in parent.families,
        node.edges.observation_parents,
    )
    for parent in relevant_parents
        print("relevant_parents for input_node, ", node.name, "relevant_parents, ", parent.name)
    end

    # If no parents belong to the specified family, return no surprise
    if isempty(relevant_parents)
        return 0
    end

    # Sum predictions from relevant parents
    parents_prediction_mean = sum(parent.states.prediction_mean for parent in relevant_parents)

    # Calculate surprise based on node type
    if node isa ContinuousInputNode
        return -log(
            pdf(
                Normal(parents_prediction_mean, node.states.prediction_precision), node.states.input_value))
    elseif node isa BinaryInputNode
        if node.states.input_value == 0
            return -log(1 - parents_prediction_mean)
        elseif node.states.input_value == 1
            return -log(parents_prediction_mean)
        end
    elseif node isa CategoricalInputNode
        # Assuming the first parent is relevant for categorical nodes
        parent = relevant_parents[1]
        return sum(-log.(exp.(log.(parent.states.prediction) .* parent.states.posterior)))
    else
        error("Unsupported node type for get_surprise_by_family.")
    end
end

