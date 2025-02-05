#######################################
# Model Comparison Node Functions #####
#######################################

# UPDATE model comparison node
"""
    update_model_comparison_node!(hgf::HGF, node::ModelComparisonNode, stepsize::Real)

Updates the model comparison node by aggregating surprise from input nodes,
computing evidence, and converting it to probabilities using a softmax.
"""
function update_model_comparison_node!(hgf::HGF, node::ModelComparisonNode, stepsize::Real)
    println("Updating ModelComparisonNode: $(node.name)")
    calculate_total_surprise_per_family!(node, hgf)
    calculate_evidence_per_family!(node)
    calculate_softmax_probabilities!(node)
    println("Updated probabilities: $(node.states.probabilities)")
end


"""
    calculate_total_surprise_per_family!(node::ModelComparisonNode, hgf::HGF)

Aggregates the surprise from each input node (except model comparison nodes)
for every defined family.
"""
function calculate_total_surprise_per_family!(node::ModelComparisonNode, hgf::HGF)
    println("=== Calculating Total Surprise Per Family ===")
    # reset the surprise
    node.states.total_surprise = Dict{String, Real}()

    # for each input node 
    for base_node in values(hgf.input_nodes)
        println("Processing base node: $(base_node.name)")
        # Check if node is an input node
        if base_node isa AbstractInputNode && !(base_node isa AbstractModelComparisonNode)

            println("This is type of base_node $(typeof(base_node))")

            # if input node doesn't have a family, throw a warning
            if isempty(base_node.families)
                @warn "Input node $(base_node.name) has no families defined."
            end
            
            # for each of the input nodes families
            for family in base_node.families
                println("  Processing family: $family")
                # calculate the surprise for the input node given its connections to the family parent nodes (family tree)
                node_surprise = get_surprise_by_family(base_node, family)
                println("    Surprise for node $(base_node.name) in family $family: $node_surprise")
                
                # add that to the model comparison node's total surprise dict
                node.states.total_surprise[family] = get(node.states.total_surprise, family, 0.0) + node_surprise
                println("    Updated total surprise for family $family: $(node.states.total_surprise[family])")
            end
        end
    end
    println("Total surprise: $(node.states.total_surprise)")
end

"""
    calculate_evidence_per_family!(node::ModelComparisonNode)

Transforms the aggregated surprise into evidence (by negation) for each family.
"""
function calculate_evidence_per_family!(node::ModelComparisonNode)
    println("=== Calculating Evidence Per Family ===")
    # init empty dictionary
    node.states.evidence = Dict{String, Real}()

    for (family, total_surprise) in node.states.total_surprise
        evidence = -total_surprise
        node.states.evidence[family] = evidence
        println("  Evidence for family $family: $evidence")
    end
    println("Evidence: $(node.states.evidence)")
end

"""
    calculate_softmax_probabilities!(node::ModelComparisonNode)

Converts the evidence for each family into a probability distribution using softmax.
"""

function calculate_softmax_probabilities!(node::ModelComparisonNode)
    println("=== Calculating Softmax Probabilities ===")
    evidences = collect(values(node.states.evidence))
    families = collect(keys(node.states.evidence))
    println("Evidences: $evidences")
    println("Families: $families")

    probs = softmax(evidences)
    println("Softmax probabilities: $probs")

    node.states.probabilities = Dict(zip(families, probs))
    println("Updated probabilities: $(node.states.probabilities)")
end

function softmax(x::Vector{Real}, β::Real = 3)
    println("=== Calculating Softmax with precision β = $β ===")
    # For numerical stability, subtract the maximum, then scale
    scaled_x = β .* (x .- maximum(x))
    println("Scaled input: $scaled_x")
    
    exp_x = exp.(scaled_x)
    println("Exponentials: $exp_x")
    
    probs = exp_x ./ sum(exp_x)
    println("Softmax output: $probs")
    
    @assert isapprox(sum(probs), 1.0; atol=1e-6) "Softmax probabilities do not sum to ~1"
    return probs
end

### Adjust coupling strengths ###

# Adjust all coupling strengths
function adjust_all_coupling_strengths!(hgf::HGF, node::ModelComparisonNode, stepsize::Real)
    println("=== Adjusting All Coupling Strengths ===")
    mc_probabilities = node.states.probabilities
    println("Model comparison probabilities: $mc_probabilities")

    # Organize all non-model-comparison nodes by family.
    family_nodes = Dict{String, Vector{AbstractNode}}()
    for current_node in values(hgf.all_nodes)
        if !(current_node isa AbstractModelComparisonNode)
            for family in current_node.families
                if !haskey(family_nodes, family)
                    family_nodes[family] = Vector{AbstractNode}()
                end
                push!(family_nodes[family], current_node)
            end
        end
    end
    #println("Family nodes: $family_nodes")

    # Update coupling strengths for edges within the same family
    for (family, nodes) in family_nodes
        println("Processing family: $family")
        # Skip families without model comparison probabilities
        if !(family in keys(mc_probabilities))
            println("  Skipping family $family (no probabilities defined)")
            continue
        end

        # Probability for this family
        family_prob = mc_probabilities[family]
        println("  Family probability: $family_prob")

        # Iterate over all nodes in the family
        for child_node in nodes
            println("    Processing child node: $(child_node.name)")
            # Ensure the child node has edges defined and parents
            if child_node.edges !== nothing && hasfield(typeof(child_node.edges), :observation_parents)
                for parent_node in child_node.edges.observation_parents
                    println("      Processing parent node: $(parent_node.name)")
                    # Check if parent is in the same family
                    if parent_node in nodes
                        
                        coupling_strength = child_node.parameters.coupling_strengths[parent_node.name]
                        
                        #define the updated strength
                        updated_strength = family_prob
                        #updated_strength = coupling_strength + stepsize * (family_prob - coupling_strength)

                        # set the child nodes parameter
                        child_node.parameters.coupling_strengths[parent_node.name] = updated_strength
                        println("      Updated coupling strength for parent $(parent_node.name): $updated_strength")

                        println("Updated coupling_strengths for $(child_node.name): ",
                                    child_node.parameters.coupling_strengths)
                    end
                end
            else
                println("    No edges defined for child node $(child_node.name)")
            end
        end
    end
end
