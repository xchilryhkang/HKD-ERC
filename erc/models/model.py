"""
HypergraphERCModel: the VUE + Hypergraph pipeline, wired end to end.

    raw feature -> proj -> prompt -> BiLSTM -> ContextGraphStack (per modality)
                                                        |
                                                        v
                                    VUEFusion (a,t,v -> v_hat, per utterance)
                                                        |
                                                        v
                    HypergraphStack ({a,t,v,e} nodes, cross-modal only, L layers)
                                                        |
                                                        v
                          concat [h_a || h_t || h_v || h_e] -> Linear -> log_prob

One hidden_dim (= lstm_hidden_size * 2) runs the whole pipeline: proj,
BiLSTM, ContextGraphStack's input AND output, VUE's concat_dim, and
Hypergraph's own dim are all exactly hidden_dim — no separate
graph_hidden_size bottleneck. classifier input is 4 * hidden_dim (concat
of the 4 final hypergraph nodes).

forward() returns (log_prob, v_hat, gamma_vue, h_final, gammas_hg): the
classification output plus the VUE/hypergraph intermediates, exposed for
inspection or a future distillation loss (see vue.py / hypergraph.py's
docstrings for what gamma_vue / gammas_hg mean). Training this model — see
train_teacher.py — is a plain masked cross-entropy on log_prob; there's no
confidence/flow/reconstruction loss here, since this architecture has none
of those components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from erc.models.graph import batch_graphify
from erc.models.prompt import MissingModalityPrompt, ConditionalPrototypePrompt
from erc.models.context_graph import ContextGraphStack
from erc.models.vue import VUEFusion
from erc.models.hypergraph import HypergraphStack


class HypergraphERCModel(nn.Module):
    def __init__(self, adim, tdim, vdim, lstm_hidden_size,
                 n_speakers, window_past, window_future, dataset, n_classes,
                 dropout=0.5, no_cuda=False,
                 use_bilstm=True, prompt_type='none', n_proto=10, prompt_dim=None,
                 n_graph_heads=4, graph_layers=2, reliability_type='feature',
                 d_e=None, vue_reliability_type='feature', use_vue_reliability_gate=True,
                 hg_layers=2, **kwargs):
        super(HypergraphERCModel, self).__init__()
        self.dataset = dataset
        self.no_cuda = no_cuda
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future

        # ============================ #
        #     Shared Projection (raw modality dim -> common hidden_dim)
        # ============================ #

        hidden_dim = lstm_hidden_size * 2
        assert hidden_dim % 2 == 0, 'lstm_hidden_size * 2 must be even (BiLSTM splits it across directions)'
        self.hidden_dim = hidden_dim

        self.proj_a = nn.Linear(adim, hidden_dim)
        self.proj_t = nn.Linear(tdim, hidden_dim)
        self.proj_v = nn.Linear(vdim, hidden_dim)

        # ============================ #
        #     Missing-Modality Prompt
        # ============================ #

        self.prompt_type = prompt_type
        if prompt_type == 'constant':
            self.prompt_a = MissingModalityPrompt(hidden_dim)
            self.prompt_t = MissingModalityPrompt(hidden_dim)
            self.prompt_v = MissingModalityPrompt(hidden_dim)
        elif prompt_type == 'prototype':
            self.modality_prompt = ConditionalPrototypePrompt(
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim or hidden_dim,
                n_proto=n_proto,
                modal_keys=('a', 't', 'v')
            )
        elif prompt_type != 'none':
            raise ValueError(f"Unknown prompt_type: {prompt_type}. Expected 'none', 'constant' or 'prototype'.")

        # ============================ #
        #     Unimodal BiLSTM (optional)
        # ============================ #

        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.lstm_a = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True, dropout=dropout)
            self.lstm_t = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True, dropout=dropout)
            self.lstm_v = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True, dropout=dropout)

        # ============================ #
        #     Unimodal Context-Graph (per modality; see context_graph.py)
        # ============================ #

        self.context_graph_a = ContextGraphStack(hidden_dim, graph_layers, n_graph_heads, dropout, self.no_cuda, reliability_type)
        self.context_graph_t = ContextGraphStack(hidden_dim, graph_layers, n_graph_heads, dropout, self.no_cuda, reliability_type)
        self.context_graph_v = ContextGraphStack(hidden_dim, graph_layers, n_graph_heads, dropout, self.no_cuda, reliability_type)

        # ============================ #
        #     VUE (consensus fusion over a/t/v, per utterance) — see vue.py
        # ============================ #

        # hypergraph.py has no input projection: every node must already
        # share one dimension. d_e defaults to hidden_dim so that's
        # automatically satisfied without the caller having to compute it;
        # if overridden, it must still match, or HypergraphStack breaks.
        self.d_e = d_e or hidden_dim
        assert self.d_e == hidden_dim, (
            f'd_e ({self.d_e}) must equal hidden_dim ({hidden_dim}): '
            f'HypergraphStack has no input projection, so VUE output and the '
            f'context-graph output it concatenates with must already match.'
        )

        self.vue = VUEFusion(
            hidden_dim, self.d_e, modal_keys=('a', 't', 'v'),
            reliability_type=vue_reliability_type, use_reliability_gate=use_vue_reliability_gate
        )

        # ============================ #
        #     Hypergraph (cross-modal only, per utterance) — see hypergraph.py
        # ============================ #

        self.hypergraph = HypergraphStack(self.d_e, n_layers=hg_layers)

        # ============================ #
        #     Classifier: concat the 4 final hypergraph nodes [a || t || v || e]
        # ============================ #

        self.classifier = nn.Linear(4 * self.d_e, n_classes)

    def forward(self, inputfeats, qmask, umask, seq_lengths, input_features_mask, adim, tdim, vdim, label=None):
        # ============================ #
        #     Unimodal Feature Extraction + Shared Projection
        # ============================ #

        audio_feats = inputfeats[0][:, :, :adim]
        text_feats = inputfeats[0][:, :, adim:adim + tdim]
        video_feats = inputfeats[0][:, :, adim + tdim:]

        audio_h = self.proj_a(audio_feats)  # [T, B, hidden_dim]
        text_h = self.proj_t(text_feats)
        video_h = self.proj_v(video_feats)

        # ============================ #
        #     Missing-Modality Prompt
        # ============================ #

        outputs_mask_raw = input_features_mask[0]  # [T, B, 3], cols = audio/text/visual

        if self.prompt_type == 'constant':
            audio_h = self.prompt_a(audio_h, outputs_mask_raw[:, :, 0])
            text_h = self.prompt_t(text_h, outputs_mask_raw[:, :, 1])
            video_h = self.prompt_v(video_h, outputs_mask_raw[:, :, 2])
        elif self.prompt_type == 'prototype':
            prompted = self.modality_prompt({'a': audio_h, 't': text_h, 'v': video_h}, outputs_mask_raw)
            audio_h, text_h, video_h = prompted['a'], prompted['t'], prompted['v']

        # ============================ #
        #     Unimodal, Optional BiLSTM Temporal Refinement
        # ============================ #

        if self.use_bilstm:
            audio_out, _ = self.lstm_a(audio_h)
            text_out, _ = self.lstm_t(text_h)
            video_out, _ = self.lstm_v(video_h)
        else:
            audio_out, text_out, video_out = audio_h, text_h, video_h

        # ============================ #
        #     Build a graph for each modality
        # ============================ #

        outputs_mask = outputs_mask_raw.unsqueeze(2)
        outputs_a = audio_out.unsqueeze(2)
        outputs_t = text_out.unsqueeze(2)
        outputs_v = video_out.unsqueeze(2)

        features_a, edge_index, edge_type, edge_type_mapping, umask_flat, features_speaker_ids, features_node_positions = batch_graphify(
            outputs_a, qmask, seq_lengths,
            self.n_speakers, self.window_past, self.window_future,
            'temporal', self.no_cuda, outputs_mask)
        features_t, edge_index, edge_type, edge_type_mapping, umask_flat, features_speaker_ids, features_node_positions = batch_graphify(
            outputs_t, qmask, seq_lengths,
            self.n_speakers, self.window_past, self.window_future,
            'temporal', self.no_cuda, outputs_mask)
        features_v, edge_index, edge_type, edge_type_mapping, umask_flat, features_speaker_ids, features_node_positions = batch_graphify(
            outputs_v, qmask, seq_lengths,
            self.n_speakers, self.window_past, self.window_future,
            'temporal', self.no_cuda, outputs_mask)

        avail_a = umask_flat[:, 0]
        avail_t = umask_flat[:, 1]
        avail_v = umask_flat[:, 2]

        # ============================ #
        #     Context-Graph Learning (per modality)
        # ============================ #

        hidden0_a = self.context_graph_a(
            features_a, edge_index, features_speaker_ids, features_node_positions,
            avail_a, seq_lengths, umask
        )
        hidden0_t = self.context_graph_t(
            features_t, edge_index, features_speaker_ids, features_node_positions,
            avail_t, seq_lengths, umask
        )
        hidden0_v = self.context_graph_v(
            features_v, edge_index, features_speaker_ids, features_node_positions,
            avail_v, seq_lengths, umask
        )
        # each hidden0_* : [T, B, hidden_dim] = tilde_h_i^m

        # ============================ #
        #     VUE: consensus fusion over a/t/v, per utterance
        # ============================ #

        availability = {
            'a': outputs_mask_raw[:, :, 0],
            't': outputs_mask_raw[:, :, 1],
            'v': outputs_mask_raw[:, :, 2],
        }
        v_hat, gamma_vue = self.vue({'a': hidden0_a, 't': hidden0_t, 'v': hidden0_v}, availability)

        # ============================ #
        #     Hypergraph: cross-modal-only refinement, per utterance
        # ============================ #

        h_init = {'a': hidden0_a, 't': hidden0_t, 'v': hidden0_v, 'e': v_hat}
        h_final, gammas_hg = self.hypergraph(h_init)

        # ============================ #
        #     Classifier: concat [a || t || v || e] -> log_prob
        # ============================ #

        concat_features = torch.cat(
            [h_final['a'], h_final['t'], h_final['v'], h_final['e']], dim=-1
        )  # [T, B, 4 * d_e]
        logits = self.classifier(concat_features)
        log_prob = F.log_softmax(logits, dim=-1)

        return log_prob, v_hat, gamma_vue, h_final, gammas_hg
