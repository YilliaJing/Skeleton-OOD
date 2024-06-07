import numpy as np

def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 368))  # 367+'unseen'
    for idx, l in enumerate(labels):
            labels_vector[idx, l] = 1

    return labels_vector

def seenAndUnseen(label):
    num_class = max(label) + 1

    unseen_indices = np.empty(0)
    seen_indices = np.empty(0)

    unseen_label = np.random.choice(num_class, size=33, replace=False)  # select 33 class for 'unseen'
    print(unseen_label)

    seen_label = []
    all = list(range(0, num_class, 1))  # 0-399
    for idx in all:
        if idx not in unseen_label:
            seen_label.append(idx)

    for idx in unseen_label:
        temp = np.where(label == idx)[0]  # 0-based index
        unseen_indices = np.hstack((unseen_indices, temp)).astype(np.int32)

    for idx in seen_label:
        temp = np.where(label == idx)[0]  # 0-based index
        seen_indices = np.hstack((seen_indices, temp)).astype(np.int32)

    return seen_indices, unseen_indices, unseen_label

def reNumber(label, selected_values):
    A = label.copy()
    indices_special = []
    if 367 not in selected_values:
        indices_special = np.where(A == 367)[0]

    A[np.isin(A, selected_values)] = 367

    # Find all unique values that are not equal to 367
    unique_values = np.unique(A[A != 367])

    # Create an empty list to store subscripts of the same value
    indices_list = []

    for value in unique_values:
        indices = np.where(A == value)[0]
        indices_list.append(indices)

    if len(indices_special) != 0:
        indices_list.append(indices_special)

    # Renumber the categories in inices_list from 0 to 366
    new_idxs = np.arange(0, 367)
    for i in range(len(indices_list)):
        A[indices_list[i]] = new_idxs[i]
    return A

def dataProcess():
    data = np.load('/home/ubuntu/xj/data/kinetics400/kinetics_data.npy')  # (260232, 3, 300, 18, 2)
    label = np.load('/home/ubuntu/xj/data/kinetics400/kinetics_label.npy')  # (260232,)
    seen_indices, unseen_indices, unseen_label_id = seenAndUnseen(label)
    label_id_new = reNumber(label, unseen_label_id)
    label_onehot_new = one_hot_vector(label_id_new)

    unseen_x = data[unseen_indices]
    seen_x = data[seen_indices]

    unseen_y = label_onehot_new[unseen_indices]
    seen_y = label_onehot_new[seen_indices]

    print('unseen_x:')
    print(len(unseen_x))

    return seen_x, seen_y, unseen_x, unseen_y


if __name__ == '__main__':
    # record rondomly sellected 33 OOD categories
    '''[  0 176  48  82 131  89 333 386 295 304 129 148 384  31 259 169 160 345
 381  34 158 181   3 235 266 261 343 263  90 172 237 116 275]'''
    seen_x, seen_y, unseen_x, unseen_y = dataProcess()
    print(seen_x.shape)  # (240490, 3, 300, 18, 2)
    print(unseen_x.shape)  # (19742, 3, 300, 18, 2)
    np.savez('/home/ubuntu/xj/data/kinetics400/Kinetics_unseen_368.npz', x_test=unseen_x, y_test=unseen_y)
    np.savez('/home/ubuntu/xj/data/kinetics400/Kinetics_seen_368.npz', x_train=seen_x, y_train=seen_y)



