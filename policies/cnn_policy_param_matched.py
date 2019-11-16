import numpy as np
import tensorflow as tf
from baselines import logger
from utils import fc, conv, ortho_init
from stochastic_policy import StochasticPolicy
from tf_util import get_available_gpus
from mpi_util import RunningMeanStd


def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]: size *= shapel.value
    return tf.reshape(x, (-1, size))


def _fcnobias(x, scope, nh, *, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        return tf.matmul(x, w)


def _normalize(x):
    eps = 1e-5
    mean, var = tf.nn.moments(x, axes=(-1,), keepdims=True)
    return (x - mean) / tf.sqrt(var + eps)


def get_action_one_hot_list(n_actions, n_envs, n_steps):
    action_index = tf.reshape(tf.range(n_actions), [-1, 1])
    action_indexs = tf.reshape(tf.tile(action_index, [1, n_envs * n_steps]), [-1])
    action_one_hot = tf.one_hot(action_indexs, n_actions)

    action_one_hot_list = tf.split(action_one_hot, n_actions)

    return action_one_hot_list


def get_action_one_hot(n_actions, n_envs, n_steps):
    batch_size = n_envs * n_steps
    action_index = tf.range(n_actions)
    action_indexs = tf.tile(action_index, [batch_size])
    action_one_hot = tf.one_hot(action_indexs, n_actions)

    return action_one_hot


def get_action_encode_array(n_actions, n_envs, n_steps, shape, base_num=0.01, split_num=16):
    # logger.info("array_action_type:all_0.01")
    batch_size = n_envs * n_steps
    action_index = list(range(n_actions))
    encode_array = np.zeros((n_actions, *shape))
    row_num = shape[0] // n_actions

    for i in action_index:
        encode_array[i, i * row_num:(i + 1) * row_num] = base_num

    encode_array = tf.convert_to_tensor(encode_array, dtype=tf.float32)

    # each_len = batch_size // split_num
    # lengths = [each_len for _ in range(split_num)]
    # lengths[-1] += batch_size - each_len * split_num

    encode_array_batch = tf.tile(encode_array, (batch_size, 1, 1))
    # encode_array_batch = [tf.tile(encode_array, (x, 1, 1)) for x in lengths]

    return encode_array_batch
    # return encode_array, lengths


# def get_action_encode_array(n_actions, n_envs, n_steps, shape, base_num=1):
#     logger.info("array_action_type:col0:4_1")
#     batch_size = n_envs * n_steps
#     action_index = list(range(n_actions))
#     encode_array = np.zeros((n_actions, *shape))
#     row_num = shape[0] // n_actions
#
#     for i in action_index:
#         encode_array[i, i * row_num:(i + 1) * row_num, 0:4] = base_num
#
#     encode_array = tf.convert_to_tensor(encode_array, dtype=tf.float32)
#     encode_array_batch = tf.tile(encode_array, (batch_size, 1, 1))
#
#     return encode_array_batch


class CnnPolicy(StochasticPolicy):
    def __init__(self, scope, ob_space, ac_space,
                 policy_size='normal', maxpool=False, extrahid=True, hidsize=128, memsize=128, rec_gate_init=0.0,
                 update_ob_stats_independently_per_gpu=True,
                 proportion_of_exp_used_for_predictor_update=1.,
                 dynamics_bonus=False,
                 action_balance_coef=1., array_action=True
                 ):
        StochasticPolicy.__init__(self, scope, ob_space, ac_space)
        self.proportion_of_exp_used_for_predictor_update = proportion_of_exp_used_for_predictor_update
        self.action_balance_coef = action_balance_coef
        self.array_action = array_action

        self.enlargement = {
            'small': 1,
            'normal': 2,
            'large': 4
        }[policy_size]
        self.rep_size = 512
        self.ph_mean = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2]) + [1], name="obmean")
        self.ph_std = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2]) + [1], name="obstd")
        memsize *= self.enlargement
        hidsize *= self.enlargement
        self.convfeat = 16 * self.enlargement
        self.ob_rms = RunningMeanStd(shape=list(ob_space.shape[:2]) + [1],
                                     use_mpi=not update_ob_stats_independently_per_gpu)
        ph_istate = tf.placeholder(dtype=tf.float32, shape=(None, memsize), name='state')
        pdparamsize = self.pdtype.param_shape()[0]
        self.memsize = memsize

        # self.int_rew_ab = None
        # self.int_rew_ab_opt = None
        if self.action_balance_coef is not None:
            # self.action_one_hot_list_rollout = get_action_one_hot_list(self.ac_space.n, self.sy_nenvs, self.sy_nsteps)
            # self.action_one_hot_list_opt = get_action_one_hot_list(self.ac_space.n, self.sy_nenvs, self.sy_nsteps - 1)
            # with tf.device('/cpu:0'):
            self.action_one_hot_rollout = get_action_one_hot(self.ac_space.n, self.sy_nenvs, self.sy_nsteps)
            # self.action_one_hot_list_opt = get_action_one_hot(self.ac_space.n, self.sy_nenvs, self.sy_nsteps - 1)

            if self.array_action:
                # with tf.device('/cpu:0'):
                self.action_encode_array_rollout = get_action_encode_array(
                    self.ac_space.n, self.sy_nenvs, self.sy_nsteps, ob_space.shape[:2])
                # self.action_encode_array_rollout, self.split_lengths = get_action_encode_array(
                #     self.ac_space.n, self.sy_nenvs, self.sy_nsteps, ob_space.shape[:2])

            self.feat_var_ab, self.max_feat_ab, self.int_rew_ab, self.int_rew_ab_rollout, self.aux_loss_ab = \
                self.define_action_balance_rew(ph_ob=self.ph_ob[None],
                                               action_one_hot=self.action_one_hot_rollout,
                                               convfeat=self.convfeat,
                                               rep_size=self.rep_size, enlargement=self.enlargement,
                                               sy_nenvs=self.sy_nenvs,
                                               sy_nsteps=self.sy_nsteps,
                                               )
            # self.feat_var_ab_opt, self.max_feat_ab_opt, self.int_rew_ab_opt, self.aux_loss_ab = \
            #     self.define_action_balance_rew(ph_ob=self.ph_ob[None][:, :-1],
            #                                    action_one_hot=self.action_one_hot_list_opt,
            #                                    convfeat=self.convfeat,
            #                                    rep_size=self.rep_size, enlargement=self.enlargement,
            #                                    sy_nenvs=self.sy_nenvs,
            #                                    sy_nsteps=self.sy_nsteps - 1,
            #                                    )

            self.pd_ab = self.pdtype.pdfromflat(self.int_rew_ab)

        # Inputs to policy and value function will have different shapes depending on whether it is rollout
        # or optimization time, so we treat separately.
        self.pdparam_opt, self.vpred_int_opt, self.vpred_ext_opt, self.snext_opt, self.logits_raw_opt = \
            self.apply_policy(self.ph_ob[None][:, :-1],
                              reuse=False,
                              scope=scope,
                              hidsize=hidsize,
                              memsize=memsize,
                              extrahid=extrahid,
                              sy_nenvs=self.sy_nenvs,
                              sy_nsteps=self.sy_nsteps - 1,
                              pdparamsize=pdparamsize
                              )
        self.pdparam_rollout, self.vpred_int_rollout, self.vpred_ext_rollout, self.snext_rollout, _ = \
            self.apply_policy(self.ph_ob[None],
                              reuse=True,
                              scope=scope,
                              hidsize=hidsize,
                              memsize=memsize,
                              extrahid=extrahid,
                              sy_nenvs=self.sy_nenvs,
                              sy_nsteps=self.sy_nsteps,
                              pdparamsize=pdparamsize
                              )
        if dynamics_bonus:
            self.define_dynamics_prediction_rew(convfeat=self.convfeat, rep_size=self.rep_size,
                                                enlargement=self.enlargement)
        else:
            self.define_self_prediction_rew(convfeat=self.convfeat, rep_size=self.rep_size,
                                            enlargement=self.enlargement)

        pd = self.pdtype.pdfromflat(self.pdparam_rollout)
        self.a_samp = pd.sample()
        self.nlp_samp = pd.neglogp(self.a_samp)
        self.entropy_rollout = pd.entropy()
        self.pd_rollout = pd

        self.pd_opt = self.pdtype.pdfromflat(self.pdparam_opt)

        self.ph_istate = ph_istate

    def apply_policy(self, ph_ob, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize,
                     ):
        data_format = 'NHWC'
        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C
        logger.info("CnnPolicy: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        X = tf.cast(ph, tf.float32) / 255.
        X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))

        activ = tf.nn.relu
        yes_gpu = any(get_available_gpus())
        with tf.variable_scope(scope, reuse=reuse), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):
            X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
            X = to2d(X)
            mix_other_observations = [X]
            X = tf.concat(mix_other_observations, axis=1)
            X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
            additional_size = 448
            X = activ(fc(X, 'fc_additional', nh=additional_size, init_scale=np.sqrt(2)))
            snext = tf.zeros((sy_nenvs, memsize))
            mix_timeout = [X]

            Xtout = tf.concat(mix_timeout, axis=1)
            if extrahid:
                Xtout = X + activ(fc(Xtout, 'fc2val', nh=additional_size, init_scale=0.1))
                X = X + activ(fc(X, 'fc2act', nh=additional_size, init_scale=0.1))
            pdparam = fc(X, 'pd', nh=pdparamsize, init_scale=0.01)
            vpred_int = fc(Xtout, 'vf_int', nh=1, init_scale=0.01)
            vpred_ext = fc(Xtout, 'vf_ext', nh=1, init_scale=0.01)

            pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
            logits_raw = pdparam

            if self.action_balance_coef is not None:
                # self.define_action_balance_rew(convfeat=self.convfeat, rep_size=self.rep_size, enlargement=self.enlargement)
                pdparam = pdparam + tf.stop_gradient( self.int_rew_ab_rollout[:, :sy_nsteps] * self.action_balance_coef)
                # pdparam = pdparam + tf.stop_gradient(self.int_rew_ab_rollout * self.action_balance_coef)

            vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
            vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))
        return pdparam, vpred_int, vpred_ext, snext, logits_raw

    def define_action_balance_rew_old(self, ph_ob, action_one_hot_list, convfeat, rep_size, enlargement, sy_nenvs,
                                  sy_nsteps):
        logger.info("Using Action Balance BONUS ****************************************************")

        with tf.variable_scope('action_balance', reuse=tf.AUTO_REUSE):
            # Random target network.
            ph = ph_ob
            assert len(ph.shape.as_list()) == 5  # B,T,H,W,C

            logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
            xr = ph
            xr = tf.cast(xr, tf.float32)
            xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
            xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

            xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
            xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
            xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
            rgbr = to2d(xr)

            X_rs = []
            for ac_one_hot in action_one_hot_list:
                X_r = tf.nn.relu(fc(tf.concat([rgbr, ac_one_hot], 1), 'fc1r', nh=256, init_scale=np.sqrt(2)))
                X_r = fc(tf.concat([X_r, ac_one_hot], 1), 'fc2r', nh=rep_size, init_scale=np.sqrt(2))
                X_rs.append(X_r)
            X_r = tf.stack(X_rs, 1)

            # Predictor network.
            logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
            # xrp = ph[:, :-1]
            xrp = ph
            xrp = tf.cast(xrp, tf.float32)
            xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))
            # ph_mean, ph_std are 84x84x1, so we subtract the average of the last channel from all channels. Is this ok?
            xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

            xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
            xrp = tf.nn.leaky_relu(
                conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
            xrp = tf.nn.leaky_relu(
                conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
            rgbrp = to2d(xrp)

            X_r_hats = []
            for ac_one_hot in action_one_hot_list:
                # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(
                    fc(tf.concat([rgbrp, ac_one_hot], 1), 'fc1r_hat1_pred', nh=256 * enlargement,
                       init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(
                    fc(tf.concat([X_r_hat, ac_one_hot], 1), 'fc1r_hat2_pred', nh=256 * enlargement,
                       init_scale=np.sqrt(2)))
                X_r_hat = fc(tf.concat([X_r_hat, ac_one_hot], 1), 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))
                X_r_hats.append(X_r_hat)

            X_r_hat = tf.stack(X_r_hats, 1)

            feat_var_ab = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
            max_feat_ab = tf.reduce_max(tf.abs(X_r))
            int_rew_ab = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1)
            int_rew_ab = tf.reshape(int_rew_ab, (sy_nenvs, sy_nsteps, *int_rew_ab.shape.as_list()[1:]))

            # self.int_rew_ab = tf.reshape(self.int_rew_ab, (self.sy_nenvs, self.sy_nsteps - 1, self.ac_space.n))

            noisy_targets = tf.stop_gradient(X_r)
            # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
            aux_loss_ab = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), [-2, -1])
            mask = tf.random_uniform(shape=tf.shape(aux_loss_ab), minval=0., maxval=1., dtype=tf.float32)
            mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
            aux_loss_ab = tf.reduce_sum(mask * aux_loss_ab) / tf.maximum(tf.reduce_sum(mask), 1.)

        return feat_var_ab, max_feat_ab, int_rew_ab, aux_loss_ab

    def define_action_balance_rew(self, ph_ob, action_one_hot, convfeat, rep_size, enlargement, sy_nenvs, sy_nsteps,
                                  l2_normalize=True, sd_normalize=False):
        logger.info("Using Action Balance BONUS ****************************************************")

        with tf.variable_scope('action_balance', reuse=tf.AUTO_REUSE):
            # Random target network.
            ph = ph_ob
            assert len(ph.shape.as_list()) == 5  # B,T,H,W,C

            logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
            xr = ph
            xr = tf.cast(xr, tf.float32)
            xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
            xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

            def conv_layers(xr):
                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))

                return xr

            if self.array_action:
                # with tf.device('/cpu:0'):
                xr = tf.reshape(tf.tile(xr, [1, self.ac_space.n, 1, 1]), (-1, *xr.shape[1:]))
                xr = tf.concat([xr, self.action_encode_array_rollout[..., None]], axis=-1)
                xr = conv_layers(xr)

                # when n_env=128, the batch size is too big for GPU. Split inputs in order to use less memory.
                # xr_results = []
                # xr_list = tf.split(xr, num_or_size_splits=self.split_lengths)
                # state_shape = xr_list[0].shape[1:]

                # for i in range(len(xr_list)):
                #     action_array_tmp = tf.tile(self.action_encode_array_rollout, (self.split_lengths[i], 1, 1))
                #     xr = tf.reshape(tf.tile(xr_list[i], [1, self.ac_space.n, 1, 1]), (-1, *state_shape))
                #     # xr = tf.concat([xr, self.action_encode_array_list_rollout[i][..., None]], axis=-1)
                #     xr = tf.concat([xr, action_array_tmp[..., None]], axis=-1)
                #     xr = conv_layers(xr)
                #     xr_results.append(xr)
                # xr = tf.concat(xr_results, 0)
            else:
                xr = conv_layers(xr)
            rgbr = to2d(xr)

            if not self.array_action:
                # extend action dim
                rgbr_shape = rgbr.shape.as_list()
                rgbr = tf.reshape(tf.tile(rgbr, [1, self.ac_space.n]), (-1, rgbr_shape[1]))

            X_r = tf.nn.relu(fc(tf.concat([rgbr, action_one_hot], 1), 'fc1r', nh=256, init_scale=np.sqrt(2)))
            X_r = fc(tf.concat([X_r, action_one_hot], 1), 'fc2r', nh=rep_size, init_scale=np.sqrt(2))

            # Predictor network.
            logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
            # xrp = ph[:, :-1]
            xrp = ph
            xrp = tf.cast(xrp, tf.float32)
            xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))
            # ph_mean, ph_std are 84x84x1, so we subtract the average of the last channel from all channels. Is this ok?
            xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

            xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
            xrp = tf.nn.leaky_relu(
                conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
            xrp = tf.nn.leaky_relu(
                conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
            rgbrp = to2d(xrp)

            rgbrp_shape = rgbrp.shape.as_list()
            rgbrp = tf.reshape(tf.tile(rgbrp, [1, self.ac_space.n]), (-1, rgbrp_shape[1]))
            X_r_hat = tf.nn.relu(fc(tf.concat([rgbrp, action_one_hot], 1),
                                    'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
            X_r_hat = tf.nn.relu(fc(tf.concat([X_r_hat, action_one_hot], 1),
                                    'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
            X_r_hat = fc(tf.concat([X_r_hat, action_one_hot], 1),
                         'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

            X_r = tf.reshape(X_r, (sy_nenvs, sy_nsteps, self.ac_space.n, rep_size))
            X_r_hat = tf.reshape(X_r_hat, (sy_nenvs, sy_nsteps, self.ac_space.n, rep_size))
            int_rew_ab_rollout = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1)
            if l2_normalize:
                int_rew_ab_rollout = tf.math.l2_normalize(int_rew_ab_rollout, axis=-1)
            elif sd_normalize:
                mean_tmp, var_tmp = tf.nn.moments(int_rew_ab_rollout, axes=[-1], keep_dims=True)
                int_rew_ab_rollout = (int_rew_ab_rollout - mean_tmp) / tf.math.sqrt(var_tmp)

            X_r = X_r[:, :-1]
            X_r_hat = X_r_hat[:, :-1]
            feat_var_ab = tf.reduce_mean(tf.nn.moments(X_r, axes=[0, 1])[1])
            max_feat_ab = tf.reduce_max(tf.abs(X_r))
            int_rew_ab = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1)
            if l2_normalize:
                logger.info("Normalize logits:l2")
                int_rew_ab = tf.math.l2_normalize(int_rew_ab, axis=-1)
            elif sd_normalize:
                logger.info("Normalize logits:standard")
                mean_tmp, var_tmp = tf.nn.moments(int_rew_ab, axes=[-1], keep_dims=True)
                int_rew_ab = (int_rew_ab - mean_tmp) / tf.math.sqrt(var_tmp)

            # int_rew_ab = tf.reshape(int_rew_ab, (sy_nenvs, sy_nsteps, *int_rew_ab.shape.as_list()[1:]))
            # int_rew_ab = tf.reshape(int_rew_ab, (sy_nenvs, sy_nsteps, self.ac_space.n))

            # self.int_rew_ab = tf.reshape(self.int_rew_ab, (self.sy_nenvs, self.sy_nsteps - 1, self.ac_space.n))

            noisy_targets = tf.stop_gradient(X_r)
            # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
            aux_loss_ab = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), [-1])
            mask = tf.random_uniform(shape=tf.shape(aux_loss_ab), minval=0., maxval=1., dtype=tf.float32)
            mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
            aux_loss_ab = tf.reduce_sum(mask * aux_loss_ab) / tf.maximum(tf.reduce_sum(mask), 1.)

        return feat_var_ab, max_feat_ab, int_rew_ab, int_rew_ab_rollout, aux_loss_ab

    def define_self_prediction_rew(self, convfeat, rep_size, enlargement):
        logger.info("Using RND BONUS ****************************************************")

        # RND bonus.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:, 1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:, 1:]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)
                # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        targets = tf.stop_gradient(X_r)
        # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.aux_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    def define_dynamics_prediction_rew(self, convfeat, rep_size, enlargement):
        # Dynamics loss with random features.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:, 1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        ac_one_hot = tf.one_hot(self.ph_ac, self.ac_space.n, axis=2)
        assert ac_one_hot.get_shape().ndims == 3
        assert ac_one_hot.get_shape().as_list() == [None, None, self.ac_space.n], ac_one_hot.get_shape().as_list()
        ac_one_hot = tf.reshape(ac_one_hot, (-1, self.ac_space.n))

        def cond(x):
            return tf.concat([x, ac_one_hot], 1)

        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:, :-1]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))
                # ph_mean, ph_std are 84x84x1, so we subtract the average of the last channel from all channels. Is this ok?
                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)

                # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(cond(rgbrp), 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(cond(X_r_hat), 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(cond(X_r_hat), 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        noisy_targets = tf.stop_gradient(X_r)
        # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.aux_loss = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    def initial_state(self, n):
        return np.zeros((n, self.memsize), np.float32)

    def call(self, dict_obs, new, istate, update_obs_stats=False):
        for ob in dict_obs.values():
            if ob is not None:
                if update_obs_stats:
                    raise NotImplementedError
                    ob = ob.astype(np.float32)
                    ob = ob.reshape(-1, *self.ob_space.shape)
                    self.ob_rms.update(ob)
        # Note: if it fails here with ph vs observations inconsistency, check if you're loading agent from disk.
        # It will use whatever observation spaces saved to disk along with other ctor params.
        feed1 = {self.ph_ob[k]: dict_obs[k][:, None] for k in self.ph_ob_keys}
        feed2 = {self.ph_istate: istate, self.ph_new: new[:, None].astype(np.float32)}
        feed1.update({self.ph_mean: self.ob_rms.mean, self.ph_std: self.ob_rms.var ** 0.5})
        # for f in feed1:
        #     print(f)
        a, vpred_int, vpred_ext, nlp, newstate, ent = tf.get_default_session().run(
            [self.a_samp, self.vpred_int_rollout, self.vpred_ext_rollout, self.nlp_samp, self.snext_rollout,
             self.entropy_rollout],
            feed_dict={**feed1, **feed2})
        return a[:, 0], vpred_int[:, 0], vpred_ext[:, 0], nlp[:, 0], newstate, ent[:, 0]
