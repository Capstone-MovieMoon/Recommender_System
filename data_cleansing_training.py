import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import spatial
from tensorboardX import SummaryWriter

import bottleneck as bn

batch_size = 500
anneal_cap = 0.2
total_anneal_steps = 200000
log_interval = 100
writer = SummaryWriter()
global data_info

class DataLoader():
    def __init__(self, path):
        self.pro_dir = os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"

        self.n_items = self.load_n_items()
    
    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")
        
    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items
    
    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')
        
        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data
    
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]
    
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
    
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        
        else:
            tr_list.append(group)
        
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:    # p, q�� input ���� output���� ���ƾ� �ϸ�, latent dimension ���� ���ƾ� �Ѵ�.
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # q�� ������ �������� ��հ� �л��� �ִ�.
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        # overfitting ������
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    # latent space �н� �� back prop.�� �ϰ� decoder�� ��������ִ� �κ�
    def forward(self, input):  
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input):
        # input ����ȭ �� �Ϻ� ���x
        h = F.normalize(input)
        h = self.drop(h)
        
        # q, �� latent space �н���.
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)    # Activation Function
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:   # �н� ��
            std = torch.exp(0.5 * logvar)
            # randn_like -> ���Ժ����κ����� ���� �ѹ����� input�� ���� �������� tensor�� ä���� ��ȯ
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    # VAE ouput�� ����� �κ�
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    #BCE = F.binary_cross_entropy(recon_x, x)
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
  
    batch_users = X_pred.shape[0]
    #print("batch_users: {}".format(batch_users))
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def train():
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, batch_size)):
        end_idx = min(start_idx + batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = sparse2tensor(data).to(device)
        data_info = data

        if total_anneal_steps > 0:
            anneal = min(anneal_cap, 
                            1. * update_count / total_anneal_steps)
        else:
            anneal = anneal_cap

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        loss = criterion(recon_batch, data, mu, logvar, anneal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, batch_size)),
                        elapsed * 1000 / log_interval,
                        train_loss / log_interval))
            
            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, batch_size)) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0
      
def evaluate(data_tr, data_te):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    #global recon_batch
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r20_list = []
    r50_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = sparse2tensor(data).to(device)

            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap

            recon_batch, mu, logvar = model(data_tensor)

            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf
           
            
            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n100)
            
            r20_list.append(r20)
            r50_list.append(r50)
 
    total_loss /= len(range(0, e_N, batch_size))
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list), n100_list, recon_batch

  
        


if __name__=="__main__":
  DATA_DIR = 'C:/Users/SH420/vae_cf/recommendation/ml-20m'
  raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
  raw_data = raw_data[raw_data['rating'] > 3.5]
  
  raw_data, user_activity, item_popularity = filter_triplets(raw_data)
  
  # User Shuffle
  unique_uid = user_activity.index
  np.random.seed(98765)
  idx_perm = np.random.permutation(unique_uid.size)
  unique_uid = unique_uid[idx_perm]

  n_users = unique_uid.size #13��...
  n_heldout_users = 10000

  # Split Train/Validation/Test User Indices
  tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
  vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
  te_users = unique_uid[(n_users - n_heldout_users):]

  # show2id -> �������̵� 0���� ����
  # profile2id -> ������ ���̵� 0���� ����
  train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
  unique_sid = pd.unique(train_plays['movieId'])

  show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
  profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    
  pro_dir = os.path.join(DATA_DIR, 'pro_sg')

  if not os.path.exists(pro_dir):
      os.makedirs(pro_dir)
    
  with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
      for sid in unique_sid:
          f.write('%s\n' % sid)

  vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
  vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

  vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

  test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
  test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

  test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

  train_data = numerize(train_plays, profile2id, show2id)
  train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
  
  vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
  vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

  vad_data_te = numerize(vad_plays_te, profile2id, show2id)
  vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

  test_data_tr = numerize(test_plays_tr, profile2id, show2id)
  test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

  test_data_te = numerize(test_plays_te, profile2id, show2id)
  test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

  
  # tune it
  torch.manual_seed(1234)
  if torch.cuda.is_available():
      device = torch.device("cuda")
  else:
      device = torch.device("cpu")

  loader = DataLoader(DATA_DIR)

  n_items = loader.load_n_items()
  train_data = loader.load_data('train')
  vad_data_tr, vad_data_te = loader.load_data('validation')
  test_data_tr, test_data_te = loader.load_data('test')

  N = train_data.shape[0]
  idxlist = list(range(N))

  p_dims = [200, 600, n_items]
  model = MultiVAE(p_dims).to(device)

  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  criterion = loss_function

  epochs = 10
  best_n100 = -np.inf
  update_count = 0

# Ctrl + C to break out of training early.
  try:
      for epoch in range(1, epochs + 1):
          epoch_start_time = time.time()
          train()
          val_loss, n100, r20, r50, n100_list, train_recon = evaluate(vad_data_tr, vad_data_te)
          print('-' * 89)
          print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(epoch, time.time() - epoch_start_time, val_loss, n100, r20, r50))
          print('-' * 89)

          n_iter = epoch * len(range(0, N, batch_size))
          writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
          writer.add_scalar('data/n100', n100, n_iter)
          writer.add_scalar('data/r20', r20, n_iter)
          writer.add_scalar('data/r50', r50, n_iter)
  
        # Save the model if the n100 is the best we've seen so far.
          if n100 > best_n100:
              with open('model.pt', 'wb') as f:
                  torch.save(model, f)
              best_n100 = n100
  
  except KeyboardInterrupt:
      print('-' * 89)
      print('Keyboard Interrupted')

# Load the best saved model.
  with open('model.pt', 'rb') as f:
      model = torch.load(f)

# Run on test data.
  test_loss, n100, r20, r50, n100_list, test_recon = evaluate(test_data_tr[0:500], test_data_te[0:500])
  print('=' * 89)
  print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | '
        'r50 {:4.2f}'.format(test_loss, n100, r20, r50))
  print('=' * 89)

  test_recon = pd.DataFrame(test_recon)

  recon_dir = os.path.join(DATA_DIR, 'reconstructed')
  if not os.path.exists(recon_dir):
      os.makedirs(recon_dir)

  test_recon.to_csv(os.path.join(recon_dir, "recon.csv"))
