import torch

from networks import ECO_Lite

def load_pretrained_ECO(model_dict, pretrained_model_dict):
    """
    function to loads the trained model of ECO
    The ECO constructed this time has the same layer order as the trained model, 
    but the name is different.
    """
 
    # Parameter name of the current network model
    param_names = []
    for name, param in model_dict.items():
        param_names.append(name)

    # Creating a new state_dict by copying the current network information
    new_state_dict = model_dict.copy()

    # Assigning the learned value to the new state_dict
    print('Loading the trained parameters...')
    for index, (key_name, value) in enumerate(pretrained_model_dict.items()):
        name = param_names[index]     # Getting the parameter name in the current network
        new_state_dict[name] = value  # Putting in that value

    return new_state_dict

if __name__ == '__main__':
    # Model instantiation
    net = ECO_Lite()
    net.eval()

    # Loading of trained model
    net_model_ECO = './models/ECO_Lite_rgb_model_Kinetics.pth.tar'
    pretrained_model = torch.load(net_model_ECO, map_location='cpu')
    pretrained_model_dict = pretrained_model['state_dict']
    # Get the variable name of the current model, etc.
    model_dict = net.state_dict()

    new_state_dict = load_pretrained_ECO(model_dict, pretrained_model_dict)

    net.eval()  # To evaluate the model put ECO network in inference mode
    net.load_state_dict(new_state_dict)

    # Save loaded weights
    torch.save(net.state_dict(), './models/pretrained.pth')
