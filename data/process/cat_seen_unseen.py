import numpy as np

unseen = np.load('/home/ubuntu/xj/data/ntu/NTU60_unseen_56.npz')
seen = np.load('/home/ubuntu/xj/data/ntu/NTU60_seen_test_56.npz')

unseen_x, unseen_y = unseen['x_test'], unseen['y_test']
seen_x, seen_y = seen['x_test'], seen['y_test']

mix_test_x = np.concatenate((unseen_x, seen_x), axis=0)
mix_test_y = np.concatenate((unseen_y, seen_y), axis=0)
print(mix_test_x.shape)
print(mix_test_y.shape)

np.savez('/home/ubuntu/xj/data/ntu/NTU60_mixtest_56.npz', x_test=mix_test_x, y_test=mix_test_y)