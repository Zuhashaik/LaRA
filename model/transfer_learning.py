import torch
#===================================================================================Adding hubert representations=========================
def transfer_learning(model, external_representations, start_index, end_index):
    '''This is when you have some kind of representations present for the tokens you've added that can help in the traning!
    rather than training / teaching / learning this representations from the scratch is a huge work and with adapter we only
    tweak the model the gradients we obtain from the training wont be sufficient to teach what exactly are those tokens are!
    Addding this will be a huge deal.'''
    if model.model.embed_tokens.weight.shape[1] != external_representations.shape[1]:
        projection_layer = torch.nn.Linear(768, 4096)
        external_representations = projection_layer(external_representations)
        
    with torch.no_grad():
        start_index = 32000
        end_index = 32999
        assert external_representations.shape[0] == (end_index - start_index + 1), "Mismatch in the number of rows to be inserted"
        model.model.embed_tokens.weight[start_index:end_index + 1, :] = external_representations
    return model