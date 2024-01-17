import torch

def get_old_result(X, W1, W2, b1, b2, scale=1, skip_connect=True):
    # print(X.shape, W1.shape,  b1.shape)
    old_result = (W1 @ X.T).T
    if b1 is not None:
        old_result = old_result + b1.unsqueeze(0)
    old_result = (W2 @ old_result.T).T
    if b2 is not None:
        old_result = old_result + b2.unsqueeze(0)
    if skip_connect:
        return X + old_result * scale
    else:
        return old_result * scale

def get_new_result(X, W1, b1=None):
    new_result = (W1 @ X.T).T
    if b1 is not None:
        new_result = new_result + b1.unsqueeze(0)
    return new_result

def reparameterize_LinearAdapter(D_fc1_W, D_fc2_W, D_fc1_b=None, D_fc2_b=None, scale=1., skip_connect=True):
    '''
    Without skip_connect:
    [W2 @ (W1 @ X + b1) + b2] * scale = [W2 @ W1 @ X + W2 @ b1 + b2] * scale
    new_W = [W2 @ W1] * scale
    new_b = [W2 @ b1 + b2] * scale
    With skip_connect:
    [W2 @ (W1 @ X + b1) + b2] * scale + X = ([W2 @ W1] * scale + I) @ X + [W2 @ b1  + b2] * scale
    new_W = [W2 @ W1] * scale + I
    new_b = [W2 @ b1  + b2] * scale
    '''
    
    print(f'reparam: W1:{D_fc1_W.shape} W2:{D_fc2_W.shape} b1:{D_fc1_b.shape if D_fc1_b is not None else "None"} b2:{D_fc2_b.shape if D_fc2_b is not None else "None"} scale:{scale} skip_connect:{skip_connect}')
    # dummy_input = torch.rand((2, 768))
    # old_result = get_old_result(dummy_input, D_fc1_W, D_fc2_W, D_fc1_b, D_fc2_b, scale, skip_connect)
    new_W = (D_fc2_W @ D_fc1_W) * scale
    if skip_connect:
        new_W = new_W + torch.eye(D_fc1_W.shape[1],D_fc2_W.shape[0]).to(D_fc1_W.device)
    new_b = 0
    if D_fc1_b is not None:
        new_b = D_fc2_W @ D_fc1_b
    if D_fc2_b is not None:
        new_b = new_b + D_fc2_b
    new_b = new_b * scale
    
    # new_result = get_new_result(dummy_input, new_W, new_b)
    # if not torch.allclose(old_result, new_result, atol=1e-6):
    #     print('hhh')
    #     print(old_result)
    #     print(new_result)
    #     # exit(0)
    if D_fc1_b is None and D_fc2_b is None:
        print(f'new_W:{new_W.shape} new_b:None')
        return new_W, None
    else:
        print(f'new_W:{new_W.shape} new_b:{new_b.shape}')
        return new_W, new_b

def fuse_Adapter2Proj(adapter_W, proj_W, adapter_b=None, proj_b=None, adapter_fisrt=True):
    '''
    (X @ W1  + b1) @ W2 + b2 = X @ W1 @ W2 + b1 @ W2 + b2
    new_W = W1 @ W2  new_b = b1 @ W2 + b2
    '''

    if adapter_fisrt:
        return reparameterize_LinearAdapter(adapter_W, proj_W, adapter_b, proj_b, scale=1., skip_connect=False)
    else:
        return reparameterize_LinearAdapter(proj_W, adapter_W, proj_b, adapter_b, scale=1., skip_connect=False)
    
def reparam_vit_zero_clip(state_dict, scale=1., skip_connect=True,num_tadapter = 2, num_layers = 12):
    model_prefix = 'backbone.transformer.resblocks'
    for layer_id in range(num_layers):
        # reparmeterize attention       
        T_Adapter_weight, T_Adapter_bias = reparameterize_LinearAdapter(D_fc1_W=state_dict[f'{model_prefix}.{layer_id}.T_Adapter_in.D_fc1.weight'],
                                                                        D_fc2_W=state_dict[f'{model_prefix}.{layer_id}.T_Adapter_in.D_fc2.weight'],
                                                                        D_fc1_b=state_dict[f'{model_prefix}.{layer_id}.T_Adapter_in.D_fc1.bias'],
                                                                        D_fc2_b=state_dict[f'{model_prefix}.{layer_id}.T_Adapter_in.D_fc2.bias'],
                                                                        scale=scale,
                                                                        skip_connect=skip_connect
                                                                        )
        in_proj_weight = state_dict[f'{model_prefix}.{layer_id}.attn.in_proj_weight']
        in_proj_bias = state_dict[f'{model_prefix}.{layer_id}.attn.in_proj_bias']
        in_proj_weight, in_proj_bias = fuse_Adapter2Proj(T_Adapter_weight, in_proj_weight, T_Adapter_bias, in_proj_bias, adapter_fisrt=True)
        state_dict[f'{model_prefix}.{layer_id}.attn.in_proj_weight'] = in_proj_weight
        state_dict[f'{model_prefix}.{layer_id}.attn.in_proj_bias'] = in_proj_bias
        del state_dict[f'{model_prefix}.{layer_id}.T_Adapter_in.D_fc1.weight']
        del state_dict[f'{model_prefix}.{layer_id}.T_Adapter_in.D_fc2.weight']
        del state_dict[f'{model_prefix}.{layer_id}.T_Adapter_in.D_fc1.bias']
        del state_dict[f'{model_prefix}.{layer_id}.T_Adapter_in.D_fc2.bias']

        S_Adapter_weight, S_Adapter_bias = reparameterize_LinearAdapter(D_fc1_W=state_dict[f'{model_prefix}.{layer_id}.S_Adapter.D_fc1.weight'],
                                                                        D_fc2_W=state_dict[f'{model_prefix}.{layer_id}.S_Adapter.D_fc2.weight'],
                                                                        D_fc1_b=state_dict[f'{model_prefix}.{layer_id}.S_Adapter.D_fc1.bias'],
                                                                        D_fc2_b=state_dict[f'{model_prefix}.{layer_id}.S_Adapter.D_fc2.bias'],
                                                                        scale=scale,
                                                                        skip_connect=skip_connect
                                                                        )
        out_proj_weight = state_dict[f'{model_prefix}.{layer_id}.attn.out_proj.weight']
        out_proj_bias = state_dict[f'{model_prefix}.{layer_id}.attn.out_proj.bias']
        out_proj_weight, out_proj_bias = fuse_Adapter2Proj(S_Adapter_weight, out_proj_weight, S_Adapter_bias, out_proj_bias, adapter_fisrt=False)
        state_dict[f'{model_prefix}.{layer_id}.attn.out_proj.weight'] = out_proj_weight
        state_dict[f'{model_prefix}.{layer_id}.attn.out_proj.bias'] = out_proj_bias
        del state_dict[f'{model_prefix}.{layer_id}.S_Adapter.D_fc1.weight']
        del state_dict[f'{model_prefix}.{layer_id}.S_Adapter.D_fc2.weight']
        del state_dict[f'{model_prefix}.{layer_id}.S_Adapter.D_fc1.bias']
        del state_dict[f'{model_prefix}.{layer_id}.S_Adapter.D_fc2.bias']

        # reparmeterize mlp
        MLP_Adapter_weight, MLP_Adapter_bias = reparameterize_LinearAdapter(D_fc1_W=state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter.D_fc1.weight'],
                                                                        D_fc2_W=state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter.D_fc2.weight'],
                                                                        D_fc1_b=state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter.D_fc1.bias'],
                                                                        D_fc2_b=state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter.D_fc2.bias'],
                                                                        scale=scale,
                                                                        skip_connect=skip_connect
                                                                        )
        c_fc_weight = state_dict[f'{model_prefix}.{layer_id}.mlp.c_fc.weight']
        c_fc_bias = state_dict[f'{model_prefix}.{layer_id}.mlp.c_fc.bias']
        c_fc_weight, c_fc_bias = fuse_Adapter2Proj(MLP_Adapter_weight, c_fc_weight, MLP_Adapter_bias, c_fc_bias, adapter_fisrt=True)
        state_dict[f'{model_prefix}.{layer_id}.mlp.c_fc.weight'] = c_fc_weight
        state_dict[f'{model_prefix}.{layer_id}.mlp.c_fc.bias'] = c_fc_bias
        del state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter.D_fc1.weight']
        del state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter.D_fc2.weight']
        del state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter.D_fc1.bias']
        del state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter.D_fc2.bias']

        if num_tadapter == 2:
            MLP_Adapter_out_weight, MLP_Adapter_out_bias = reparameterize_LinearAdapter(D_fc1_W=state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter_out.D_fc1.weight'],
                                                                            D_fc2_W=state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter_out.D_fc2.weight'],
                                                                            D_fc1_b=state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter_out.D_fc1.bias'],
                                                                            D_fc2_b=state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter_out.D_fc2.bias'],
                                                                            scale=scale,
                                                                            skip_connect=skip_connect
                                                                            )
            
            
            c_proj_weight = state_dict[f'{model_prefix}.{layer_id}.mlp.c_proj.weight']
            c_proj_bias = state_dict[f'{model_prefix}.{layer_id}.mlp.c_proj.bias']
            c_proj_weight, c_proj_bias = fuse_Adapter2Proj(MLP_Adapter_out_weight, c_proj_weight, MLP_Adapter_out_bias, c_proj_bias, adapter_fisrt=False)
            state_dict[f'{model_prefix}.{layer_id}.mlp.c_proj.weight'] = c_proj_weight
            state_dict[f'{model_prefix}.{layer_id}.mlp.c_proj.bias'] = c_proj_bias
            del state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter_out.D_fc1.weight']
            del state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter_out.D_fc2.weight']
            del state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter_out.D_fc1.bias']
            del state_dict[f'{model_prefix}.{layer_id}.MLP_Adapter_out.D_fc2.bias']


if __name__ == '__main__':

    pth_path = '.work_dirs/recognition/vit_zero_clip/N003d_vit_zero_base_clip_k400_8x16x1_div12/epoch_40.pth'
    state_dict = torch.load(pth_path, map_location='cpu')['state_dict']
    
    reparam_vit_zero_clip(state_dict, scale=0.5, skip_connect=True, num_tadapter=2, num_layers=12)

    torch.save(state_dict, pth_path.replace('.pt', '_eval.pt'))
