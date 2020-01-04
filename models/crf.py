import torch
import torch.nn as nn
import torch.nn.functional as F

def to_scalar(var):
    return var.view(-1).detach().tolist()[0]

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx

def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_

class CRF(nn.Module):
    def __init__(self,tagset_size,tag_dictionary,device,is_bert=None):
        super(CRF,self).__init__()

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        if is_bert:
            self.START_TAG = "[CLS]"
            self.STOP_TAG = "[SEP]"
        self.tag_dictionary = tag_dictionary
        self.tagset_size = tagset_size
        self.device = device
        self.transitions = torch.randn(tagset_size, tagset_size)
        # self.transitions = torch.zeros(tagset_size, tagset_size)
        self.transitions.detach()[self.tag_dictionary[self.START_TAG], :] = -10000
        self.transitions.detach()[:, self.tag_dictionary[self.STOP_TAG]] = -10000
        self.transitions = self.transitions.to(device)
        self.transitions = nn.Parameter(self.transitions)

    def _viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        scores = []
        init_vvars = (torch.FloatTensor(1, self.tagset_size).to(self.device).fill_(-10000.0))
        init_vvars[0][self.tag_dictionary[self.START_TAG]] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = (
                    forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size)
                    + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
                forward_var
                + self.transitions[self.tag_dictionary[self.STOP_TAG]]
        )
        terminal_var.detach()[self.tag_dictionary[self.STOP_TAG]] = -10000.0
        terminal_var.detach()[self.tag_dictionary[self.START_TAG]] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())
        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())
            scores.append([elem.item() for elem in softmax.flatten()])
        swap_best_path, swap_max_score = (
            best_path[0],
            scores[-1].index(max(scores[-1])),
        )
        scores[-1][swap_best_path], scores[-1][swap_max_score] = (
            scores[-1][swap_max_score],
            scores[-1][swap_best_path],
        )
        start = best_path.pop()
        assert start == self.tag_dictionary[self.START_TAG]
        best_path.reverse()
        return best_scores, best_path, scores

    def _forward_alg(self, feats, lens_):
        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary[self.START_TAG]] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=self.device,
        )
        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)
        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]
            tag_var = (
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                + transitions
                + forward_var[:, i, :][:, :, None]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1)
            )
            max_tag_var, _ = torch.max(tag_var, dim=2)
            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )
            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))
            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_
            forward_var = cloned
        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        terminal_var = forward_var + self.transitions[self.tag_dictionary[self.STOP_TAG]][None, :].repeat(forward_var.shape[0], 1)
        alpha = log_sum_exp_batch(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, lens_):
        start = torch.LongTensor([self.tag_dictionary[self.START_TAG]]).to(self.device)
        start = start[None, :].repeat(tags.shape[0], 1)
        stop = torch.LongTensor([self.tag_dictionary[self.STOP_TAG]]).to(self.device)
        stop = stop[None, :].repeat(tags.shape[0], 1)
        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)
        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = self.tag_dictionary[self.STOP_TAG]
        score = torch.FloatTensor(feats.shape[0]).to(self.device)
        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(self.device)
            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])
        return score

    def _obtain_labels(self, feature, id2label,input_lens):
        tags = []
        all_tags = []
        for feats, length in zip(feature, input_lens):
            confidences, tag_seq, scores = self._viterbi_decode(feats[:length])
            tags.append([id2label[tag] for tag in tag_seq])
            all_tags.append([[id2label[score_id] for score_id, score in enumerate(score_dist)] for score_dist in scores])
        return tags, all_tags

    def calculate_loss(self, scores, tag_list,lengths):
        return self._calculate_loss_old(scores, lengths, tag_list)

    def _calculate_loss_old(self, features, lengths, tags):
        forward_score = self._forward_alg(features, lengths)
        gold_score = self._score_sentence(features, tags, lengths)
        score = forward_score - gold_score
        return score.mean()


