
import numpy as np
import tensorflow as tf
from utils import *
from distance import distance
import npdistance as nd
from predict import predict_cv
import copy
import time
from sklearn.preprocessing import normalize
import math

class SNEQ:

    def __init__(self, input,A, X, L,z, K=3, data_ = '',p_val=0.10, p_test=0.05, p_nodes=0.0, n_hidden=None,
                 max_iter=50001, tolerance=100,batch =100, p_recom=0.1, p_semi=0.3,scale=False, seed=0, verbose=True):
        self.data_name= data_
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.M = 8
        self.K = 16
        self.num_hops = K
        self.input = input
        self.label_ratio = p_semi
        X = X.astype(np.float32)
        self.GT = copy.deepcopy(A).toarray().astype(np.int8)
     
        p = np.arange(A.shape[0])
        np.random.shuffle(p)
        p = p[:300]
        self.recom_edges_id = p
        self.classification_ratios =  [0.02,0.10]
    
        if p_nodes > 0:
            A = self.__setup_inductive(A, X, p_nodes)
        else:
            self.X = X # tf.SparseTensor(*sparse_feeder(X))
            self.feed_dict = None
        self.labeled_nodes = self.sample_labeled_nodes(z,p_semi)
        self.N, self.D = X.shape
        self.L = L
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.scale = scale
        self.verbose = verbose
        self.batch_size = batch
        if n_hidden is None:
            n_hidden = [512]
        self.n_hidden = n_hidden

        if p_val + p_test > 0:
            train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(
                A=A, p_val=p_val, p_test=p_test, seed=seed, neg_mul=1, every_node=True, connected=False,
                undirected=False)

            A_train = edges_to_sparse(train_ones, self.N)
            hops = get_hops(A_train, K)
        else:
            hops = get_hops(A, K)
        self.hops = hops
        self.scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                           hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                       for h in hops}
        self.triplets = tf.placeholder(tf.int32, [None, None],'triplets')
        self.batch_input_tf = tf.sparse_placeholder(tf.float32,name='sb')
        self.neg_margin_tf = tf.placeholder(tf.float32,[None])
        self.pos_margin_tf = tf.placeholder(tf.float32,[None])
        self.lamb_weight = tf.placeholder(tf.float32)
        self.similarity = tf.placeholder(tf.float32,[None,None])
        self.mask = tf.placeholder(tf.float32,[None,None])
        self.__build()
        self.__dataset_generator(hops, scale_terms)
        self.__build_loss()

        if p_recom >= 0:
            self.recommend_edges = self.X[self.recom_edges_id,:]

        if p_val > 0:
            val_edges = np.row_stack((val_ones, val_zeros))  # N x 2
            print (val_edges.shape)
            self.left_val = self.X[val_edges[:, 0], :]
            self.right_val = self.X[val_edges[:, 1], :]
            self.val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1


        if p_test > 0:
            test_edges = np.row_stack((test_ones, test_zeros))  # N x 2
            self.left_test = self.X[test_edges[:, 0], :]
            self.right_test = self.X[test_edges[:, 1], :]
            self.test_ground_truth = A[test_edges[:, 0], test_edges[:, 1]].A1

        if p_nodes > 0:
            self.neg_ind_energy = -self.energy_kl(self.ind_pairs)

    def sample_labeled_nodes(self,z,p):
        len_ = self.X.shape[0]
        u = np.arange(len_)
        np.random.shuffle(u)
        keep_ = int(len_ * (1. - p))
        t = set()
        for i in u[:keep_]:
            z[i] = t
        # z[u[:keep_]] = -9999
        return z

    def __build(self):
        w_init = tf.contrib.layers.xavier_initializer
        sizes = [self.D] + self.n_hidden
        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                initializer=w_init())
            b = tf.get_variable(name='b{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())

            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(self.batch_input_tf, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.tanh(encoded)

        W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float32, initializer=w_init())
        mu_ = tf.matmul(encoded, W_mu) + b_mu
        self.mu = tf.nn.tanh(mu_)

        self.codebooks = tf.cast(tf.get_variable("codebook", [self.M * self.K, self.L]), tf.float32)
        logits = self.mu
        W_mu2 = tf.get_variable(name='W_mu2', shape=[self.L, self.L], dtype=tf.float32, initializer=w_init())
        b_mu2 = tf.get_variable(name='b_mu2', shape=[self.L], dtype=tf.float32, initializer=w_init())
        logits_a = tf.nn.tanh(tf.matmul(logits, W_mu2) + b_mu2)
        logits_a = tf.reshape(logits_a, [-1, self.M, self.K])
        logits_a = tf.nn.softmax(logits_a, dim=-1)
        self.atten_index = tf.cast(tf.argmax(logits_a, axis=-1), tf.int32)
        logits_a = tf.reshape(logits_a, [-1, self.M * self.K])

        logits = logits * logits_a
        logits = tf.reshape(logits, [-1, self.M, self.K], name="logits")
        # D = self._gumbel_softmax(logits, self._TAU, sampling=True)
        D = tf.nn.softmax(logits,-1)
        _output_ = tf.reshape(D, [-1, self.M * self.K])  # ~ (B, M * K)
        # self.maxp = tf.reduce_mean(tf.reduce_max(D, axis=2))
        y_hat = self._decode(_output_, self.codebooks)
        loss = 0.5 * tf.reduce_sum((y_hat - self.mu) ** 2, axis=1)
        self.loss_quatization = tf.reduce_mean(loss, name="loss")

        # recontruct rules
        self.max_index = max_index = tf.cast(tf.argmax(logits, axis=2), tf.int32)

        self.offset = offset = tf.range(self.M, dtype="int32") * self.K
        self.codes_with_offset = codes_with_offset = max_index + offset[None, :]
        selected_vectors = tf.gather(self.codebooks, codes_with_offset)  # ~ (B, M, H)
        self.reconstructed_embed = tf.reduce_sum(selected_vectors, axis=1)  # ~ (B, H)

    def _decode(self, gumbel_output, codebooks):
        return tf.matmul(gumbel_output, codebooks)

    def __build_loss(self):
        anc_vect = tf.gather(self.mu, self.triplets[:, 0])
        hop_pos =  tf.gather(self.mu, self.triplets[:, 1])
        hop_neg = tf.gather(self.mu, self.triplets[:,2])
        eng_pos = self.F2_distance(anc_vect,hop_pos)
        eng_neg = self.F2_distance(anc_vect,hop_neg)
        basic_loss = tf.maximum(eng_pos - eng_neg +    (self.neg_margin_tf-self.pos_margin_tf)*15. , 0.0)
        self.loss = tf.reduce_mean(basic_loss)
        lab_dis = distance(self.mu,  pair=True, dist_type='euclidean2')
        self.class_margin = tf.reduce_mean(tf.abs(lab_dis - self.similarity)*self.mask)

    def F2_distance(self, p1, p2):
        return distance(p1, p2, pair=False, dist_type='euclidean2')

    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return 0


    def construct_sim(self, label):
        a = np.ones((label.shape[0], label.shape[0])) * ( (self.num_hops+5)*15.0)
        b = np.ones((label.shape[0], label.shape[0]))

        for i in range(label.shape[0]):
            for j in range(label.shape[0]):
                if len(label[i]) <= 0 or len(label[j]) <= 0:
                    b[i][j] = 0
                    b[j][i] = 0
                    continue
                O =  len(label[i].intersection(label[j]))*1.0 / len(label[i].union(label[j]))
                if O != 0:
                    a[i][j] = 0
        return a, b

    def adaptation_factor(self, x):
        if x >= 1.0:
            return 1.0
        den = 1.0 + math.exp(-10 * x)
        lamb = 2.0 / den - 1.0
        return lamb

    def gen(self):
        while True:
            data,scal,neig_ty =  to_triplets(sample_all_hops(self.hops), self.scale_terms)
            num = data.shape[0]
            if num >= self.batch_size:
                its = int(num/self.batch_size)
            else:
                its = 1
                self.batch_size = num
            arr = np.arange(data.shape[0])
            np.random.shuffle(arr)
            np.random.shuffle(arr)
            for i in range(its):
                range_index = arr[(i*self.batch_size):(i+1)*self.batch_size]
                triplet_batch =data[range_index]
                scale_batch = scal[range_index]
                neig_batch = neig_ty[range_index]
                triplet_batch_ = triplet_batch.transpose().reshape(-1)
                triplet_batch1 = np.unique(triplet_batch_)

                c = np.array([np.where(triplet_batch1 == j)[0][0] for j in  triplet_batch_])
                c = c.reshape(3,self.batch_size).transpose()

                yield self.X[triplet_batch1,:],c ,neig_batch.astype(np.float32), \
                      np.array(self.labeled_nodes[triplet_batch1])

    def train(self, z, gpu_list='0'):

        
        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss +
                         self.lamb_weight*self.loss_quatization+
                                (2-self.lamb_weight)*self.class_margin)
   
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=gpu_list,
                                                                          allow_growth=True)))
        sess.run(tf.global_variables_initializer())
        iterator = self.gen()
        print ('start traning ...')
        for epoch in range(self.max_iter):
            data, trplits, neg_type,labes = iterator.__next__()
            print(neg_type)
            input()
            _s, Mask = self.construct_sim(labes)

            decay = self.adaptation_factor(epoch/5000.0)
            codebook, codes,q_loss,struct_loss,class_loss, _ = sess.run([self.codebooks,self.max_index,self.loss_quatization,self.loss, self.class_margin, train_op], {self.batch_input_tf: sparse_feeder(data),
                                                       self.triplets: trplits,
                                                       self.neg_margin_tf: neg_type[:, 2],
                                                       self.pos_margin_tf: neg_type[:, 1],
                                                       self.similarity: _s, self.mask: Mask,
                                                       self.lamb_weight:decay
                                                       })

            if epoch % 500 == 0:
                val_left = []
                val_right = []
      
                qus2 = []
                qus1 = []
                for i in range(0,self.right_val.shape[0] , 100):
                    mu1,qu1 = sess.run([self.mu,self.reconstructed_embed], {self.batch_input_tf: sparse_feeder(self.left_val[i: i + 100, :])})
                    mu2,qu2 = sess.run([self.mu, self.reconstructed_embed], {self.batch_input_tf: sparse_feeder(self.right_val[i: i + 100, :])})
                    val_left.append(mu1)
                    val_right.append(mu2)
                    qus1.append(qu1)
                    qus2.append(qu2)
                qus2 = np.row_stack(qus2)
                qus1 = np.row_stack(qus1)
                val_left = np.row_stack(val_left)
                val_right = np.row_stack(val_right)
                score = nd.distance(val_left, val_right, pair=False, dist_type='euclidean2')
                score2 = nd.distance(qus2, qus1, pair=False, dist_type='euclidean2')
                score =   - score
                score2 =  - score2
                val_auc1, val_ap1 = score_link_prediction(self.val_ground_truth, score)
                val_auc2, val_ap2 = score_link_prediction(self.val_ground_truth, score2)

                print('epoch: {:3d}, struct loss: {:.4f}, val_auc: {:.4f}, '
                      'val_ap: {:.4f}, c_loss: {:.4f}, quati:{:.5f}, qu_auc:{:.4f}, qu_ap:{:.4f},decay:{:.3f}'.format(epoch, struct_loss,
                                val_auc1, val_ap1,class_loss, q_loss,val_auc2,val_ap2,decay))

            if epoch % 500 == 0 :
                quantization_index = []
                querys=[]
                reconstructed_vect = []
                for i in range(0,self.X.shape[0],1000):
                    mu1, qu1, reco = sess.run([self.mu, self.max_index,self.reconstructed_embed],
                                        {self.batch_input_tf: sparse_feeder(self.X[i:i+1000, :])})
                    quantization_index.append(qu1)
                    querys.append(mu1)
                    reconstructed_vect.append(reco)
                reconstructed_vect = np.row_stack(  reconstructed_vect )
                querys = np.row_stack(np.array(querys))
                

                if epoch % 500 == -1:
                     np.savez('./output/'+self.data_name +'_'+str(epoch).zfill(6)+'.npz',
                              q=reconstructed_vect,e=querys,
                              codebook=codebook,nodes=self.recom_edges_id,
                              q_index=quantization_index)

           

                print('epoch: {:3d}, struct loss: {:.4f},  '
                      '  c_loss: {:.4f}, quati:{:.5f}, decay:{:.3f}'.format(
                    epoch, struct_loss, class_loss, q_loss,decay))

        return sess
