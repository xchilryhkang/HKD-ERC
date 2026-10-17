import numpy as np
import torch
from torch.autograd import Variable


# Pad input tensor to meet target length (extend only on seq_len dimension)
def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):  # If input is Variable type (deprecated), handle similarly
        var = tensor
        if length > var.size(0):  # If length is insufficient, pad with zeros
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var  # Target length met, return directly
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


# Concatenate and align outputs of shape [num_utterance, dim] to [seqlen, batch, dim]
def utterance_to_conversation(outputs, seq_lengths, umask, no_cuda):
    input_conversation_length = torch.tensor(seq_lengths)  # Number of utterances per conversation
    start_zero = input_conversation_length.data.new(1).zero_()  # Build [0]

    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)  # Max conversation length in this batch
    # Calculate start utterance index for each conversation [0, 6, 30, 43]
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    # Slice + pad each conversation to get a uniform length batch tensor and transpose to [seqlen, batch, dim]
    outputs = torch.stack([pad(outputs.narrow(0, s, l), max_len, no_cuda)
                           for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0).transpose(0, 1)
    return outputs


def edge_perms(l, window_past, window_future):
    """
    Function:
        Generates all legal edge pairs for a sequence based on specified time windows (forward & backward) 
        (used for building graph structures).
    """

    all_perms = set() 
    array = np.arange(l)

    for j in range(l): 

        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array

        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]

        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]

        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))

        all_perms = all_perms.union(perms)

    return list(all_perms)

def batch_graphify(features, qmask, lengths, n_speakers, window_past, window_future, graph_type, no_cuda, input_features_mask):
    """
    (Function documentation remains unchanged)
    """
    edge_type_mapping = {
        'past': 0,
        'now': 1,
        'future': 2
    }
    qmask_ = qmask.cpu().data.numpy().astype(int)

    node_features, node_features_mask, edge_index, edge_type, features_speaker_ids, features_node_positions = [], [], [], [], [], []

    length_sum = 0
    batch_size = features.size(1)

    for j in range(batch_size):
        node_feature = features[:lengths[j], j, :, :]
        node_feature = torch.reshape(node_feature, (-1, node_feature.size(-1)))
        node_features.append(node_feature)

        node_feature_mask = input_features_mask[:lengths[j], j, :]
        node_feature_mask = torch.reshape(node_feature_mask, (-1, node_feature_mask.size(-1)))
        node_features_mask.append(node_feature_mask)

        current_qmask = qmask_[j][:lengths[j]]
        features_speaker_ids.append(torch.LongTensor(current_qmask))
        features_node_positions.append(torch.arange(0, lengths[j], dtype=torch.long))
        
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(src + length_sum, tgt + length_sum) for (src, tgt) in perms1]
        length_sum += lengths[j]
        
        for (local_src, local_tgt), (global_src, global_tgt) in zip(perms1, perms2):
            edge_index.append([global_src, global_tgt])

            if local_tgt > local_src:
                order_type = 'past'
            elif local_tgt == local_src:
                order_type = 'now'
            else:
                order_type = 'future'

            edge_type_name = order_type
            
            edge_type.append(edge_type_mapping[edge_type_name])
            
    node_features = torch.cat(node_features, dim=0)
    node_features_mask = torch.cat(node_features_mask, dim=0)
    edge_index = torch.tensor(edge_index).transpose(0, 1)
    edge_type = torch.tensor(edge_type)
    features_speaker_ids = torch.cat(features_speaker_ids, dim=0)
    features_node_positions = torch.cat(features_node_positions, dim=0)
    
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()
        node_features_mask = node_features_mask.cuda()
        features_speaker_ids = features_speaker_ids.cuda()
        features_node_positions = features_node_positions.cuda()

    return node_features, edge_index, edge_type, edge_type_mapping, node_features_mask, features_speaker_ids, features_node_positions



def batch_graphify_dynamic(outputs, qmask, seq_lengths, n_speakers, 
                           past_threshold, future_threshold, no_cuda=False, outputs_mask=None):
    """
    Dynamically build the graph based on cumulative edge weight thresholds.
    (V4: Corrected based on the dimension order [Batch, Time] of qmask)
    """
    device = outputs.device
    
    time_size, batch_size, _, feature_dim = outputs.size()

    if isinstance(seq_lengths, list):
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long, device=device)
    else:
        seq_lengths_tensor = seq_lengths.to(device=device, dtype=torch.long)

    node_features_list = []
    valid_speaker_ids_list = []

    for j in range(batch_size):
        true_len = seq_lengths_tensor[j].item()
        if true_len == 0:
            continue

        node_feature = outputs[:true_len, j, 0, :]
        node_features_list.append(node_feature)

        valid_ids = qmask[j, :true_len]
        valid_speaker_ids_list.append(valid_ids)

    if not node_features_list:
        features = torch.empty((0, feature_dim), device=device)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty((0, 1), dtype=torch.float32, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)
        return (features, edge_index, edge_type, {}, None, None, None, edge_weight)

    features = torch.cat(node_features_list, dim=0)
    speaker_ids_all = torch.cat(valid_speaker_ids_list, dim=0)
    
    all_edges = []
    all_weights = []
    sigma = 5.0
    length_sum = 0

    for j in range(batch_size):
        conv_len = seq_lengths_tensor[j].item()
        if conv_len == 0:
            continue
        
        speaker_ids_local = speaker_ids_all[length_sum : length_sum + conv_len]

        for u_local in range(conv_len):
            u_global = u_local + length_sum
            speaker_u = speaker_ids_local[u_local]
            
            past_weight_sum = 0.0
            for v_local in range(u_local, -1, -1):
                v_global = v_local + length_sum
                speaker_v = speaker_ids_local[v_local]
                
                speaker_weight = 1.0 if speaker_u.item() == speaker_v.item() else 0.5
                dist_sq = (u_local - v_local)**2
                temporal_weight = torch.exp(-torch.tensor(dist_sq, dtype=torch.float32, device=device) / (2 * sigma**2))
                final_weight = speaker_weight * temporal_weight
                
                if (past_weight_sum + final_weight > past_threshold) and (u_local != v_local):
                    break
                
                past_weight_sum += final_weight
                all_edges.append([v_global, u_global])
                all_weights.append(final_weight)

            self_loop_weight = all_weights[-1]
            future_weight_sum = self_loop_weight
            
            for v_local in range(u_local + 1, conv_len):
                v_global = v_local + length_sum
                speaker_v = speaker_ids_local[v_local]
                
                speaker_weight = 1.0 if speaker_u.item() == speaker_v.item() else 0.5
                dist_sq = (v_local - u_local)**2
                temporal_weight = torch.exp(-torch.tensor(dist_sq, dtype=torch.float32, device=device) / (2 * sigma**2))
                final_weight = speaker_weight * temporal_weight
                
                if future_weight_sum + final_weight > future_threshold:
                    break

                future_weight_sum += final_weight
                all_edges.append([v_global, u_global])
                all_weights.append(final_weight)
        
        length_sum += conv_len

    if not all_edges:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty((0, 1), dtype=torch.float32, device=device)
    else:
        edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t().contiguous()
        all_weights_tensor = torch.stack(all_weights) if isinstance(all_weights[0], torch.Tensor) else torch.tensor(all_weights, dtype=torch.float32, device=device)
        edge_weight = all_weights_tensor.view(-1, 1)

    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
    
    return (features, edge_index, edge_type, {}, None, None, None, edge_weight)
