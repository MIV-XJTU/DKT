import copy

import torch
from timm.models.layers import trunc_normal_
from torch import nn

import continual.utils as cutils


class ContinualClassifier(nn.Module):

    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n

    def merge_head(self, head1, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data
        head.weight.data[-n:] = head1.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n


class DKT(nn.Module):

    def __init__(
            self,
            transformer,
            nb_classes,
            individual_classifier='',
    ):
        super().__init__()

        self.nb_classes = nb_classes
        self.embed_dim = transformer.embed_dim
        self.individual_classifier = individual_classifier
        self.in_finetuning = False

        self.nb_classes_per_task = [nb_classes]

        self.patch_embed = transformer.patch_embed
        self.pos_drop = transformer.pos_drop
        self.sabs = transformer.transformers[0].blocks[:-1]

        self.tabs = transformer.transformers[0].blocks[-1]

        self.task_tokens = nn.ParameterList([transformer.cls_token])
        self.general_token = copy.deepcopy(transformer.cls_token)
        trunc_normal_(self.general_token, std=.02)

        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head = nn.ModuleList([
                ContinualClassifier(in_dim, out_dim).cuda()
            ])
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()

    def end_finetuning(self):
        self.in_finetuning = False

    def begin_finetuning(self):
        self.in_finetuning = True

    def add_model(self, nb_new_classes):
        self.nb_classes_per_task.append(nb_new_classes)

        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)

        in_dim, out_dim = self._get_ind_clf_dim()
        if len(self.head) > 1:
            self.head[0].merge_head(self.head[1], self.nb_classes_per_task[-1])
            self.head[1].reset_parameters()
        else:
            self.head.append(
                ContinualClassifier(in_dim, out_dim).cuda()
            )

    def _get_ind_clf_dim(self):
        if self.individual_classifier == '1-1':
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown ind classifier {self.individual_classifier}')
        return in_dim, out_dim

    def freeze(self, names):
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_task_tokens':
                cutils.freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
                # cutils.freeze_parameters(self.general_token[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.freeze_parameters(self.task_tokens, requires_grad=requires_grad)
                # cutils.freeze_parameters(self.general_token, requires_grad=requires_grad)
            elif name == 'sab':
                self.sabs.eval()
                cutils.freeze_parameters(self.patch_embed, requires_grad=requires_grad)
                cutils.freeze_parameters(self.sabs, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.freeze_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            elif name == 'opentokens':
                cutils.freeze_parameters(self.task_tokens[:-1], requires_grad=True)
                cutils.freeze_parameters(self.general_token[:-1], requires_grad=True)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.task_tokens[:-1],
            'task_tokens': self.task_tokens.parameters(),
            'new_task_tokens': [self.task_tokens[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'ca': self.tabs.parameters(),
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                if self.individual_classifier else \
                self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
        }

    def reset_classifier(self):
        if isinstance(self.head, nn.ModuleList):
            for head in self.head:
                head.reset_parameters()
        else:
            self.head.reset_parameters()

    def hook_before_update(self):
        pass

    def hook_after_update(self):
        pass

    def hook_after_epoch(self):
        pass

    def epoch_log(self):
        log = {}

        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        with torch.no_grad():
            for i in range(len(self.task_tokens)):
                for j in range(i + 1, len(self.task_tokens)):
                    dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                    mean_dist.append(dist)

                    min_dist = min(dist, min_dist)
                    max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        return log

    def get_internal_losses(self, clf_loss):
        int_losses = {}
        return int_losses

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        layer_number = 0

        for blk in self.sabs:
            layer_number += 1
            if layer_number == 5:
                # Maybe you can change to other place(like 4) to try again
                general_token = self.general_token.expand(B, 1, -1)
                x = blk(x, general_token)
            else:
                x = blk(x, None)

        return self.forward_features_decode(x)

    def forward_features_decode(self, x):
        B = len(x)
        task_tokenss = None
        if len(self.task_tokens) > 1:
            task_tokenss = torch.cat(
                [task_token.expand(B, 1, -1) for task_token in self.task_tokens[:-1]],
                dim=1
            )
            task = self.task_tokens[-1].expand(B, 1, -1)
            task_cls = torch.cat((task, task_tokenss), dim=1)
        else:
            task_cls = torch.cat(
                [task_token.expand(B, 1, -1) for task_token in self.task_tokens],
                dim=1
            )
        task_tokens = self.tabs(
            torch.cat((task_cls, x), dim=1), task_tokenss
        )
        cls = task_tokens[:, 0]
        return cls

    def forward_classifier(self, tokens):
        logits = []

        for i, head in enumerate(self.head):
            logits.append(head(tokens))
        logits = torch.cat(logits, dim=1)

        return {
            'logits': logits,
            'tokens': tokens
        }

    def forward(self, x):
        tokens = self.forward_features(x)
        return self.forward_classifier(tokens)


def eval_training_finetuning(mode, in_ft):
    if 'tr' in mode and 'ft' in mode:
        return True
    if 'tr' in mode and not in_ft:
        return True
    if 'ft' in mode and in_ft:
        return True
    return False
