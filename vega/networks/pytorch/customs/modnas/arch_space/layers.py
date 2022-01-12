# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers of nested network modules."""
from typing import Dict, List, Optional, Tuple, Type, Union, Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import Module
from modnas.registry.layer_def import build as build_layer_def
from modnas.utils.logging import get_logger
from .slot import Slot
from .slot import register_slot_ccs
from . import layer_defs


logger = get_logger('arch_space')


@register_slot_ccs
class DAGLayer(nn.Module):
    """Directed Acyclic Graph Layer."""

    def __init__(self,
                 chn_in: Tuple[int, int],
                 chn_out: None,
                 stride: int,
                 n_nodes: int,
                 allocator: str,
                 merger_state: str,
                 merger_out: str,
                 enumerator: str,
                 preproc: Optional[List[Type[Module]]] = None,
                 edge_cls: Type[Slot] = Slot,
                 edge_kwargs: Optional[Dict[str, Any]] = None,
                 name: Optional[str] = None) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.stride = stride
        chn_in = (chn_in, ) if isinstance(chn_in, int) else chn_in
        self.chn_in = chn_in
        self.n_input = len(chn_in)
        self.n_states = self.n_input + self.n_nodes
        edge_kwargs = edge_kwargs or {}
        e_chn_in = edge_kwargs.get('_chn_in')
        self.n_input_e = 1 if e_chn_in is None or isinstance(e_chn_in, int) else len(e_chn_in)
        self.allocator = build_layer_def(allocator, self.n_input, self.n_nodes)
        self.merger_state = build_layer_def(merger_state)
        self.merger_out = build_layer_def(merger_out, start=self.n_input)
        self.merge_out_range = self.merger_out.merge_range(self.n_states)
        self.enumerator = build_layer_def(enumerator)
        self.topology = []

        chn_states = []
        if preproc is None:
            self.preprocs = None
            chn_states.extend(chn_in)
        else:
            chn_cur = edge_kwargs.get('_chn_in')
            chn_cur = chn_cur if self.n_input == 1 else chn_cur[0]
            self.preprocs = nn.ModuleList()
            for i in range(self.n_input):
                self.preprocs.append(preproc[i](chn_in[i], chn_cur))
                chn_states.append(chn_cur)

        self.fixed = False
        self.dag = nn.ModuleList()
        self.num_edges = 0
        for i in range(n_nodes):
            cur_state = self.n_input + i
            self.dag.append(nn.ModuleList())
            for sidx in self.enumerator.enum(cur_state, self.n_input_e):
                e_chn_in = self.allocator.chn_in([chn_states[s] for s in sidx], sidx, cur_state)
                edge_kwargs['_chn_in'] = e_chn_in
                edge_kwargs['_stride'] = stride if all(s < self.n_input for s in sidx) else 1
                if chn_out is not None:
                    edge_kwargs['_chn_out'] = chn_out
                if name is not None:
                    edge_kwargs['name'] = '{}_{}'.format(name, self.num_edges)
                e = edge_cls(**edge_kwargs)
                self.dag[i].append(e)
                self.num_edges += 1
            chn_states.append(self.merger_state.chn_out([ei.chn_out for ei in self.dag[i]]))
            self.chn_out = self.merger_out.chn_out(chn_states)
        logger.debug('DAGLayer: etype:{} chn_in:{} chn:{} #n:{} #e:{}'.format(str(edge_cls), self.chn_in,
                                                                              edge_kwargs['_chn_in'][0], self.n_nodes,
                                                                              self.num_edges))
        self.chn_out = self.merger_out.chn_out(chn_states)
        self.chn_states = chn_states

    def forward(self, x: Union[Tensor, List[Tensor]]) -> Tensor:
        """Compute Layer output."""
        states = x if isinstance(x, list) else [x]
        if self.preprocs is not None:
            states = [prep(s) for prep, s in zip(self.preprocs, states)]

        for nidx, edges in enumerate(self.dag):
            res = []
            n_states = self.n_input + nidx
            topo = self.topology[nidx] if self.fixed else None
            for eidx, sidx in enumerate(self.enumerator.enum(n_states, self.n_input_e)):
                if topo is not None and eidx not in topo:
                    continue
                e_in = self.allocator.alloc([states[i] for i in sidx], sidx, n_states)
                e_in = e_in[0] if isinstance(e_in, list) and len(e_in) == 1 else e_in
                res.append(edges[eidx](e_in))
            s_cur = self.merger_state.merge(res)
            states.append(s_cur)
        out = self.merger_out.merge(states)
        return out

    def to_arch_desc(self, k: Union[int, List[int]] = 2) -> Any:
        """Return archdesc from Layer."""
        desc = []
        edge_k = 1
        k_states = k
        if isinstance(k_states, int):
            k_states = [k_states] * len(self.dag)
        for nidx, edges in enumerate(self.dag):
            topk_edges = []
            n_states = self.n_input + nidx
            topo = self.topology[nidx] if self.fixed else None
            for eidx, sidx in enumerate(self.enumerator.enum(n_states, self.n_input_e)):
                if topo is not None and eidx not in topo:
                    continue
                g_edge_child = edges[eidx].to_arch_desc(k=edge_k + 1)
                if not isinstance(g_edge_child, (list, tuple)):
                    g_edge_child = [g_edge_child]
                g_edge_child = [g for g in g_edge_child if g != 'NIL'][:edge_k]
                try:
                    w_edge = torch.max(edges[eidx].ent.prob().detach()[:-1])
                except AttributeError:
                    continue
                g_edge = [g_edge_child, list(sidx), n_states]
                if len(topk_edges) < k_states[nidx]:
                    topk_edges.append((w_edge, g_edge))
                    continue
                for i in range(len(topk_edges)):
                    w, _ = topk_edges[i]
                    if w_edge > w:
                        topk_edges[i] = (w_edge, g_edge)
                        break
            desc.append([g for w, g in topk_edges])
        return desc

    def build_from_arch_desc(self, desc: Any, *args, **kwargs) -> None:
        """Build layer ops from desc."""
        chn_states = self.chn_states[:self.n_input]
        num_edges = 0
        self.topology = []
        for nidx, (edges, dag_rows) in enumerate(zip(desc, self.dag)):
            cur_state = self.n_input + nidx
            e_chn_out = []
            topo = []
            dag_topology = list(self.enumerator.enum(cur_state, self.n_input_e))
            for g_child, sidx, _ in edges:
                eidx = dag_topology.index(tuple(sidx))
                topo.append(eidx)
                e = dag_rows[eidx]
                e.build_from_arch_desc(g_child, *args, **kwargs)
                num_edges += 1
                e_chn_out.append(e.chn_out)
            self.topology.append(topo)
            chn_states.append(self.merger_state.chn_out(e_chn_out))
        self.num_edges = num_edges
        self.chn_states = chn_states
        self.chn_out = self.merger_out.chn_out(chn_states)
        self.fixed = True
        logger.debug('DAGLayer: chn_in:{} #n:{} #e:{}'.format(self.chn_in, self.n_nodes, self.num_edges))


@register_slot_ccs
class MultiChainLayer(nn.Module):
    """Layer with multiple chains of network modules."""

    def __init__(self,
                 chn_in,
                 chn_out,
                 stride,
                 n_chain,
                 n_chain_nodes,
                 allocator,
                 merger_out,
                 preproc=None,
                 edge_cls=Slot,
                 edge_kwargs=None,
                 name=None):
        super().__init__()
        chn_in = (chn_in, ) if isinstance(chn_in, int) else chn_in
        edge_kwargs = edge_kwargs or {}
        self.chn_in = chn_in
        self.chn_out = chn_out
        self.stride = stride
        self.n_chain = n_chain
        if isinstance(n_chain_nodes, int):
            n_chain_nodes = [n_chain_nodes] * n_chain
        else:
            if len(n_chain_nodes) != n_chain:
                raise ValueError("Chains of network modules are wrong.")
        self.n_chain_nodes = n_chain_nodes
        self.n_nodes = sum(n_chain_nodes)
        self.n_input = len(chn_in)
        self.n_states = self.n_input + self.n_nodes
        self.n_input_e = 1
        self.allocator = build_layer_def(allocator, self.n_input, self.n_chain)
        self.merger_out = build_layer_def(merger_out, start=self.n_input)
        self.merge_out_range = self.merger_out.merge_range(self.n_states)
        chn_states = []
        if preproc is None:
            self.preprocs = None
            chn_states.extend(chn_in)
        else:
            chn_cur = edge_kwargs.get('chn_in')
            chn_cur = chn_cur if self.n_input == 1 else chn_cur[0]
            self.preprocs = nn.ModuleList()
            for i in range(self.n_input):
                self.preprocs.append(preproc[i](chn_in[i], chn_cur))
                chn_states.append(chn_cur)

        self.fixed = False
        chains = nn.ModuleList()
        sidx = range(self.n_input)
        for cidx in range(self.n_chain):
            edges = []
            cur_state = self.n_input + cidx
            e_chn_in = self.allocator.chn_in([chn_states[s] for s in sidx], sidx, cur_state)
            for nidx in range(self.n_chain_nodes[cidx]):
                edge_kwargs['_chn_in'] = e_chn_in
                edge_kwargs['_stride'] = stride if nidx == 0 else 1
                if nidx == 0:
                    edge_kwargs['_chn_out'] = sum([chn_out * e // s for e, s in zip(e_chn_in, self.chn_in)])
                if name is not None:
                    edge_kwargs['name'] = '{}_{}_{}'.format(name, cidx, nidx)
                e = edge_cls(**edge_kwargs)
                e_chn_in = e.chn_out
                edges.append(e)
            chn_states.append(e_chn_in)
            chains.append(nn.Sequential(*edges))
        self.chains = chains
        self.chn_out = self.merger_out.chn_out(chn_states)
        self.chn_states = chn_states

    def forward(self, x):
        """Compute Layer output."""
        states = x if isinstance(x, list) else [x]
        if self.preprocs is not None:
            states = [self.preprocs[i](x[i]) for i in range(self.n_input)]
        sidx = range(self.n_input)
        for cidx, chain in enumerate(self.chains):
            cur_state = self.n_input + cidx
            out = self.allocator.alloc([states[i] for i in sidx], sidx, cur_state)
            out = out[0] if isinstance(out, list) and len(out) == 1 else out
            out = chain(out)
            states.append(out)
        out = self.merger_out.merge(states)
        return out

    def to_arch_desc(self, *args, **kwargs):
        """Return archdesc from Layer."""
        desc = []
        for chain in self.chains:
            g_chain = [e.to_arch_desc(*args, **kwargs) for e in chain]
            desc.append(g_chain)
        return desc

    def build_from_arch_desc(self, desc, *args, **kwargs):
        """Build layer ops from desc."""
        if len(desc) != len(self.chains):
            raise ValueError('Failed to build layer ops from desc.')
        for g_chain, chain in zip(desc, self.chains):
            if len(g_chain) != len(chain):
                raise ValueError('Failed to build layer ops from desc.')
            for g_edge, e in zip(g_chain, chain):
                e.build_from_arch_desc(g_edge, *args, **kwargs)
