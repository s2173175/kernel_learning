import torch



def get_analytics(outputs, targets):
    """
    for l1 loss only
    """

    #sample stats ----------------------

    #outputs.size [batch_size, output_dims]
    #targets.size [batch_size, output_dims]
    #error.size = [batch_size]

    error = torch.mean(torch.abs(outputs-targets), dim=1)
    
    sample_stats = {
        "mean":error.mean().item(),
        "std":error.std().item(),
        "max":error.max().item(),
        "min":error.min().item()
    }


    #sample stats per output dim ----------------------
    
    #outputs.size [batch_size, output_dims]
    #targets.size [batch_size, output_dims]
    #attribute_error.size = [batch_size, output_dims]

    attribute_error = torch.abs(outputs-targets)

    attribute_stats = {
        "mean":torch.mean(attribute_error, dim=0).numpy(),
        "std":torch.std(attribute_error, dim=0).numpy(),
        "max":torch.max(attribute_error, dim=0)[0].numpy(),
        "min":torch.min(attribute_error, dim=0)[0].numpy()
    }

    

    return (sample_stats, attribute_stats)





if __name__ == "__main__":

    x = torch.randn((5,2))
    y = torch.randn((5,2))

    t = torch.Tensor()

    t = torch.cat((t,x,y),0)

    print(t.size())

    
    # get_analytics(x,y)

    