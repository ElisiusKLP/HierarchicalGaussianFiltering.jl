"""
    update_hgf!(
        hgf::HGF,
        inputs::Union{
            Real,
            Missing,
            Vector{<:Union{Real,Missing}},
            Dict{String,<:Union{Real,Missing}},
        },
    )

Update all nodes in an HGF based on an input. The input can either be missing, a single value, a vector of values, or a dictionary of input node names and corresponding values.
"""
function update_hgf!(
    hgf::HGF,
    inputs::Union{
        Real,
        Missing,
        Vector{<:Union{Real,Missing}},
        Dict{String,<:Union{Real,Missing}},
    };
    stepsize::Real = 1,
    dynamic_coupling = true, # Default dynamic_coupling to ON
)

    ### Update node predictions from last timestep ###
    #For each node (in the opposite update order)
    for node in reverse(hgf.ordered_nodes.all_state_nodes)
        #Update its prediction from last trial
        update_node_prediction!(node, stepsize)
    end

    #For each input node, in the specified update order
    for node in reverse(hgf.ordered_nodes.input_nodes)
        #Update its prediction from last trial
        update_node_prediction!(node, stepsize)
    end

    ### Supply inputs to input nodes ###
    enter_node_inputs!(hgf, inputs)

    """ MODIFICATION"""
    ### Model Comparison Node ###
    # For each model comparison node we collect the combined suprise for each family
    # by running each input node through a surprise collection

    # Update model comparison nodes.
    for node in values(hgf.input_nodes)
        if node isa Main.HierarchicalGaussianFiltering.ModelComparisonNode
            update_model_comparison_node!(hgf, node, stepsize)
            if dynamic_coupling
                adjust_all_coupling_strengths!(hgf, node, stepsize)
            end
        end
    end

    println("--- Succesfully updated all model comparison nodes ---")

    """MODIFICATION END"""

    ### Update input node value prediction errors ###
    #For each input node, in the specified update order
    for node in hgf.ordered_nodes.input_nodes
        #Update its value prediction error
        update_node_value_prediction_error!(node)
    end

    ### Update input node value parent posteriors ###
    #For each node that is a value parent of an input node
    for node in hgf.ordered_nodes.early_update_state_nodes
        #Update its posterior    
        update_node_posterior!(node, node.update_type)
        #And its value prediction error
        update_node_value_prediction_error!(node)
        #And its precision prediction error
        update_node_precision_prediction_error!(node)
    end

    ### Update input node precision prediction errors ###
    #For each input node, in the specified update order
    for node in hgf.ordered_nodes.input_nodes
        #Update its value prediction error
        update_node_precision_prediction_error!(node)
    end

    ### Update remaining state nodes ###
    #For each state node, in the specified update order
    for node in hgf.ordered_nodes.late_update_state_nodes
        #Update its posterior    
        update_node_posterior!(node, node.update_type)
        #And its value prediction error
        update_node_value_prediction_error!(node)
        #And its volatility prediction error
        update_node_precision_prediction_error!(node)
    end

    ### Save the history for each node ###
    #If save history is enabled
    if hgf.save_history

        #Update the timepoint
        push!(hgf.timesteps, hgf.timesteps[end] + stepsize)

        #Go through each node
        for node in hgf.ordered_nodes.all_nodes

            #Go through each state
            for state_name in fieldnames(typeof(node.states))
                #Add that state to the history
                push!(getfield(node.history, state_name), getfield(node.states, state_name))
            end
        end

        # add model comparison node history (not an ordered node)
        for node in values(hgf.input_nodes)
            if node isa Main.HierarchicalGaussianFiltering.ModelComparisonNode
                
                # go through each state
                for state_name in fieldnames(typeof(node.states))
                        #Add that state to the history
                    push!(getfield(node.history, state_name), getfield(node.states, state_name))
                end
            end
        end

    end

    return nothing
end

"""
    enter_node_inputs!(hgf::HGF, input)

Set input values in input nodes. Can either take a single value, a vector of values, or a dictionary of input node names and corresponding values.
"""
function enter_node_inputs!(hgf::HGF, input::Union{Real,Missing})

    #Update the input node by passing the specified input to it
    update_node_input!(first(hgf.ordered_nodes.input_nodes), input)

    return nothing
end

function enter_node_inputs!(hgf::HGF, inputs::Vector{<:Union{Real,Missing}})

    #If the vector of inputs only contain a single input
    if length(inputs) == 1
        #Just input that into the first input node
        enter_node_inputs!(hgf, first(inputs))

    elseif typeof(first(hgf.input_nodes)[2]) == NoisyCategoricalInputNode #----------------- NEW-NOISY -----------------
        update_node_input!(first(hgf.input_nodes)[2], inputs)

    else

        #For each input node and its corresponding input
        for (input_node, input) in zip(hgf.ordered_nodes.input_nodes, inputs)
            #Enter the input
            update_node_input!(input_node, input)
        end
    end

    return nothing
end

function enter_node_inputs!(hgf::HGF, inputs::Dict{String,<:Union{Real,Missing}})

    #Update each input node by passing the corresponding input to it
    for (node_name, input) in inputs
        #Enter the input
        update_node_input!(hgf.input_nodes[node_name], input)
    end

    return nothing
end


"""
    update_node_input!(node::AbstractInputNode, input::Union{Real,Missing})

Update the prediction of a single input node.
"""
function update_node_input!(node::AbstractInputNode, input::Union{Real,Missing})
    #Receive input
    node.states.input_value = input

    return nothing
end

function update_node_input!(node::AbstractInputNode, input::Vector{T} where T <: Real) #----------------- NEW-NOISY -----------------

    #Receive input
    node.states.input_value = input

    return nothing
end