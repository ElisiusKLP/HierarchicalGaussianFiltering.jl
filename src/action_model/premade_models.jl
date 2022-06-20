"""
    function premade_agent(
        model_name::String,
        perception_model = (;),
        params = Dict(),
        states = Dict(),
        specifications = (;),
    )

Function for initializing the structure of an agent model.
"""
function premade_agent(
    model_name::String,
    perception_model = (;),
    params = Dict(),
    states = Dict(),
    specifications = (;),
)

    #A list of all the included premade models
    premade_models = Dict(
        "hgf_gaussian_response" => create_gaussian_response(; specifications...),    #A gaussian response based on an hgf
    )

    #If the user asked for help
    if model_name == "help"
        #Return the list of keys
        print(keys(premade_models))
        return nothing
    end

    #If the specified model is not in the list of keys
    if model_name ∉ keys(premade_models)
        #Raise an error
        throw(
            ArgumentError(
                "the specified string does not match any model. Type premade_agent('help') to see a list of valid input strings",
            ),
        )

        #Otherwise
    else
        #Create an agent with the corresponding model
        agent = HGF.init_agent(premade_models[model_name], perception_model, params, states)

        #Return the agent
        return agent
    end
end


"""
    gaussian_response(agent::AgentStruct, input)

Gaussian response action model. Updates the hgf, extracts the posterior mean for x1, and reports it with some noise
"""
function gaussian_response(agent::AgentStruct, input)

    #Get out the HGF
    hgf = agent.perception_struct

    #Update the HGF
    hgf.perception_model(hgf, input)

    #Extract the posterior belief about x1
    μ1 = hgf.state_nodes["x1"].state.posterior_mean

    #Create normal distribution with mean μ1 and a standard deviation from parameters
    distribution = Distributions.Normal(μ1, agent.params["action_noise"])

    #Return the action dsitribution
    return distribution
end



"""
    create_gaussian_response(node::String, state::String)

Function for creating a customized gaussian response action model. Takes a node name and a state as arguments. Outputs a function which reports the chosen state from the chosen node with some noise.
"""
function create_gaussian_response(; node::String = "x1", state::String = "posterior_mean")

    #Change the state to a symbol
    state = Symbol(state)

    #Evaluate the function definition
    eval(
        quote

            #Create the function
            function gaussian_response(action_struct, input)

                #Get out the HGF
                hgf = action_struct.perception_struct

                #Update the HGF
                hgf.perception_model(hgf, input)

                #Extract the specified state from the specified node
                target_state = hgf.state_nodes[$node].state.$state

                #Create normal distribution with mean of the target value and a standard deviation from parameters
                distribution = Distributions.Normal(
                    target_state,
                    action_struct.params["action_noise"],
                )

                #Return the action distribution
                return distribution
            end
        end,
    )

    #Return the action model
    return gaussian_response
end
