"""
"""
function ActionModels.init_agent(
    action_model::Vector{Function};
    substruct::HGF,
    params::Dict = Dict(),
    states::Dict = Dict(),
    settings::Dict = Dict(),
)

    #Create action model struct
    agent = Agent(
        action_model = hgf_multiple_actions,
        substruct = substruct,
        params = params,
        states = states,
        settings = settings,
    )

    #If an action state was not specified
    if !("action" in keys(agent.states))
        #Add an empty action state
        agent.states["action"] = missing
    end

    #For each specified state
    for (state_key, state_value) in states
        #Add it to the history
        agent.history[state_key] = [state_value]
    end

    #If a setting called action_models has been specified manually
    if "action_models" in keys(settings)
        #Throw an error
        throw(
            ArgumentError(
                "Using a setting called 'action_models' with multiple hgf action models is not supported",
            ),
        )
    else
        #Add vector of action models to settings
        agent.settings["action_models"] = action_model
    end

    return agent
end


"""
"""
function hgf_multiple_actions(agent::Agent, input)

    #Update the hgf
    hgf = agent.substruct
    update_hgf!(hgf, input)

    #Extract vector of action models
    action_models = agent.settings["action_models"]

    #Initialize vector for action distributions
    action_distributions = []

    #Do each action model separately
    for action_model in action_models
        #And append them to the vector of action distributions
        push!(action_distributions, action_model(agent, input))
    end

    return action_distributions
end