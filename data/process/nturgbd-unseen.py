import numpy as np

def seenAndunseen(skel_x, label_onehot, label_id):
    unseen_indices = np.empty(0)
    seen_indices = np.empty(0)

    '''# 0-59  randomly generate 5 OOD categories
    unseen_label = list(np.random.randint(0, 60, size=5))'''

    unseen_label = [22, 23, 33, 46, 56]  # 1-60
    print('unseen: ')
    print(unseen_label)

    seen_label = []
    all = list(range(0, 60, 1))  # 0-59
    for idx in all:
        if idx not in unseen_label:
            seen_label.append(idx)

    for idx in unseen_label:
        temp = np.where(label_id == idx)[0]  # 0-based index
        unseen_indices = np.hstack((unseen_indices, temp)).astype(np.int32)

    for idx in seen_label:
        temp = np.where(label_id == idx)[0]  # 0-based index
        seen_indices = np.hstack((seen_indices, temp)).astype(np.int32)

    return seen_indices, unseen_indices, unseen_label

def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 56))  # 55+'unseen'
    for idx, l in enumerate(labels):
            labels_vector[idx, l] = 1

    return labels_vector

def dataProcess(data_path):
    npz_data = np.load(data_path)
    skel_x = npz_data['x_test']  # 56578,300,150
    label_onehot = npz_data['y_test']  # 56578,60
    label_id = npz_data['label_id']  # 56578,

    seen_indices, unseen_indices, unseen_label_id = seenAndunseen(skel_x, label_onehot, label_id)

    label_id_new = label_id
    for i in range(len(label_id_new)):
        if label_id_new[i] in unseen_label_id:
            label_id_new[i] = 55
        elif label_id_new[i] == 55:
            label_id_new[i] = 22
        elif label_id_new[i] == 57:
            label_id_new[i] = 23
        elif label_id_new[i] == 58:
            label_id_new[i] = 33
        elif label_id_new[i] == 59:
            label_id_new[i] = 46

    label_onehot_new = one_hot_vector(label_id_new)

    unseen_x = skel_x[unseen_indices]
    seen_x = skel_x[seen_indices]

    unseen_y = label_onehot_new[unseen_indices]
    seen_y = label_onehot_new[seen_indices]

    print('unseen_x:')
    print(len(unseen_x))

    return seen_x, seen_y, unseen_x, unseen_y


if __name__ == '__main__':
    data_path = '/home/cvlab/xj/Projs/HD-GCN/data/ntu/NTU60_all.npz'
    seen_x, seen_y, unseen_x, unseen_y = dataProcess(data_path)
    np.savez('/home/cvlab/xj/data/ntu/NTU60_unseen_56.npz', x_test=unseen_x, y_test=unseen_y)
    np.savez('/home/cvlab/xj/data/ntu/NTU60_seen_56.npz', x_train=seen_x, y_train=seen_y)
