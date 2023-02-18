import torch

from typing import List

from config import args
from triplet import EntityDict
from dict_hub import get_link_graph
from doc import Example


def rerank_by_graph(batch_score: torch.tensor,
                    examples: List[Example],
                    entity_dict: EntityDict):

    if args.task == 'wiki5m_ind':
        assert args.neighbor_weight < 1e-6, 'Inductive setting can not use re-rank strategy'

    if args.neighbor_weight < 1e-6:
        return

    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]
        n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
                                                                  entity_dict=entity_dict,
                                                                  n_hop=args.rerank_n_hop)
        delta = torch.tensor([args.neighbor_weight for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, n_hop_indices, delta)

        # The test set of FB15k237 removes triples that are connected in train set,
        # so any two entities that are connected in train set will not appear in test,
        # however, this is not a trick that could generalize.
        # by default, we do not use this piece of code .

        # if args.task == 'FB15k237':
        #     n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
        #                                                               entity_dict=entity_dict,
        #                                                               n_hop=1)
        #     n_hop_indices.remove(entity_dict.entity_to_idx(cur_ex.head_id))
        #     delta = torch.tensor([-0.5 for _ in n_hop_indices]).to(batch_score.device)
        #     n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)
        #
        #     batch_score[idx].index_add_(0, n_hop_indices, delta)

## paths inside each batch
def rerank_by_path(batch_score: torch.tensor, examples: List[Example], entity_dict: EntityDict,
                    train_hr_tensor: torch.tensor, entities_tensor: torch.tensor):

    # if args.task == 'wiki5m_ind':
    #     assert args.neighbor_weight < 1e-6, 'Inductive setting can not use re-rank strategy'

    # if args.neighbor_weight < 1e-6:
    #     return
    graph = get_link_graph()
    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]
        head_id, tail_id = cur_ex.head_id, cur_ex.tail_id
        if head_id not in graph.path_dict: ## current head entity has not been searched, search
            get_link_graph().get_n_hop_path_indices(cur_ex.head_id, n_hop=3)
        ## we have searched the head entity for k-hop paths
        ## return paths if exist for each candidate entity
        path_indices = torch.tensor([i for i in range(batch_score.size(1))]).to(batch_score.device)
        path_reward = torch.zeros(batch_score.size(1)).to(batch_score.device)
        for t_idx in range(batch_score.size(1)):
            n_hop_paths = []
            t_id = entity_dict.entity_exs[t_idx].entity_id
            if t_id in graph.path_dict[head_id]:
                n_hop_paths = graph.path_dict[head_id][t_id]
            # print(head_id, tail_id, n_hop_paths)
            # print(head_id, graph.head2rt[head_id])

            t_idx_tensor = entities_tensor[t_idx]
            ## compute path_score based on hr and t embeddings
            for p, conf in n_hop_paths:
                p = torch.Tensor(list(map(lambda x: int(x), p.split('+')[1:]))).to(torch.int32).to(batch_score.device)
                hr_idx_tensor = torch.cumprod(torch.index_select(train_hr_tensor, 0, p), dim=0)[-1]
                # print('hridx tensor: {}'.format(torch.sum(hr_idx_tensor)))
                path_reward[t_idx] += torch.abs(torch.mm(hr_idx_tensor.unsqueeze(0), t_idx_tensor.unsqueeze(1)).squeeze())

        # print("total path reward: {}".format(torch.sum(path_reward)))
        # print("entity with paths: {}".format(len(graph.path_dict[head_id])))
        # raise
        ## add path_score to the triple
        batch_score[idx].index_add_(0, path_indices, path_reward)