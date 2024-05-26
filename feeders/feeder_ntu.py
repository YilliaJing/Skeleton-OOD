import numpy as np

from torch.utils.data import Dataset

from feeders import tools
from scipy.special import logsumexp
from sklearn.metrics import precision_recall_curve, auc


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            # print(self.data.shape)
            if npz_data['y_test'].shape[1] == 55:
                seen_label = np.where(npz_data['y_test'] > 0)[1]
                unseen_label = np.ones((len(npz_data['y_test']) - len(seen_label))) * 55
                self.label = np.concatenate((seen_label, unseen_label), axis=0)
            else:
                self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]  # 3, 300, 25, 2
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)  # 3, T, V
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        return data_numpy, label, index  # 3, 64, 25, 2

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def get_score(self, logits, type_score):
        if type_score == 'Energy':
            scores = logsumexp(logits, axis=1)
        elif type_score == 'MSP':
            exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
            scores = np.max(softmax, axis=1)
        
        return scores
    
    def get_measures(self, score, origin_fc_x, n_classes, type, dice=False):
        global seen_data, energy_in, energy_ood, seen_label
        metric = {}
        # lobit->score
        scores = self.get_score(logits=score, type_score=type)

        # ------------ unseen FPR95+AUROC ------------
        if n_classes == 56:
            length = 4730
            energy_in = scores[length+1:]
            energy_ood = scores[:length]
            if dice:
                seen_data = origin_fc_x[length:, :55]
            else:
                seen_data = score[length+1:, :55]
            seen_label = self.label[length+1:]
        elif n_classes in [55, 58, 111]:
            if n_classes == 55 or n_classes == 58:
                length = 5184
            elif n_classes == 111:
                length = 10446
            energy_in = scores[:length]
            energy_ood = scores[length+1:]
            if dice:
                seen_data = origin_fc_x[:length]
            else:
                seen_data = score[:length]
            seen_label = self.label[:length]

        energy_in.sort()
        energy_ood.sort()
        all = np.concatenate((energy_in, energy_ood))

        rank = seen_data.argsort()  # 用score测试？看分类结果是不是对  对比原始版本
        hit_top_k = [l in rank[i, -1:] for i, l in enumerate(seen_label)]
        print('Top1: ' + str(sum(hit_top_k) * 1.0 / len(hit_top_k) * 100))

        all.sort()

        num_k = energy_in.shape[0]
        num_n = energy_ood.shape[0]
        threshold = energy_in[round(0.05 * num_k)]
        print('seen-test-5%: ' + str(threshold))
        mix_fpr95 = np.sum(energy_ood > threshold) / float(num_n)
        print('tpr: ' + str(np.sum(energy_in > threshold) / float(num_k) * 100))

        tp = -np.ones([num_k + num_n + 1], dtype=int)
        fp = -np.ones([num_k + num_n + 1], dtype=int)
        tp[0], fp[0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[l + 1:] = tp[l]
                fp[l + 1:] = np.arange(fp[l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[l + 1:] = np.arange(tp[l] - 1, -1, -1)
                fp[l + 1:] = fp[l]
                break
            else:
                if energy_ood[n] < energy_in[k]:
                    n += 1
                    tp[l + 1] = tp[l]
                    fp[l + 1] = fp[l] - 1
                else:
                    k += 1
                    tp[l + 1] = tp[l] - 1
                    fp[l + 1] = fp[l]

        j = num_k + num_n - 1
        for l in range(num_k + num_n - 1):
            if all[j] == all[j - 1]:
                tp[j] = tp[j + 1]
                fp[j] = fp[j + 1]
            j -= 1

        tpr = np.concatenate([[1.], tp / tp[0], [0.]])
        fpr = np.concatenate([[1.], fp / fp[0], [0.]])
        metric['unseen-AUROC'] = -np.trapz(1. - fpr, tpr) * 100
        metric['unseen-fpr95'] = mix_fpr95 * 100
        print('AUROC: ' + str(metric['unseen-AUROC']))
        print('FPR95: ' + str(metric['unseen-fpr95']))

        # ---------- Error ----------
        pred_binary_labels = (scores >= threshold).astype(int)
        true_binary_labels = np.where(self.label == 55, 0, 1)

        # 计算detection error
        TP, FP, TN, FN = 0, 0, 0, 0

        for true, pred in zip(true_binary_labels, pred_binary_labels):
            if true == 1 and pred == 1:
                TP += 1
            elif true == 0 and pred == 1:
                FP += 1
            elif true == 0 and pred == 0:
                TN += 1
            elif true == 1 and pred == 0:
                FN += 1

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        bi_Error = 0.5 * (1 - TPR) + 0.5 * FPR
        metric['Error'] = bi_Error * 100

        # ------------ seen-test -----------
        scores_seen = self.get_score(logits=seen_data, type_score=type)
        pred_label = np.full_like(scores_seen, 55)

        # energy大于阈值的索引视为seen
        seen_indices = scores_seen > threshold
        print(sum(seen_indices))

        # 对于大于阈值的样本，将score第一个维度最大值的下标赋给result_array相应位置
        max_positions = np.argmax(seen_data, axis=1)
        print(sum(max_positions == seen_label))
        pred_label[seen_indices] = max_positions[seen_indices]

        correct_matches = sum(pred_label == seen_label)
        total_samples = len(seen_label)
        metric['seen-acc'] = (correct_matches / total_samples) * 100

        return metric


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
