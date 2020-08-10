import numpy as np


def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    all_idxs = list(range(len(dataset)))
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, num_users, num_shards=200, num_imgs=250):
    num_shards = num_shards
    num_imgs = num_imgs
    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.train_labels)

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users
