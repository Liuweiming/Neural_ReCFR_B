"""
A model for Neural ReCFR-B.
Adapted from open_spiel.python.algorithms.alpha_zero.model.
"""
import collections
import functools
import os
from typing import Sequence

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as tfk


def cascade(x, fns):
    for fn in fns:
        x = fn(x)
    return x


tfkl = tf.keras.layers

# ================================================================
# Flat vectors
# ================================================================


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm)
                 if grad is not None else grad for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


def set_flat(var_list, dtype=tf.float32):
    assigns = []
    shapes = list(map(var_shape, var_list))
    total_size = np.sum([intprod(shape) for shape in shapes])

    theta = tf.placeholder(dtype, [total_size], name="set_flat_input")
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = intprod(shape)
        assigns.append(tf.assign(v, tf.reshape(
            theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns, name="set_flat")
    return op, theta


def apply_gradients(var_list, opt, name="train", dtype=tf.float32):
    shapes = list(map(var_shape, var_list))
    total_size = np.sum([intprod(shape) for shape in shapes])

    theta = tf.placeholder(dtype, [total_size], name="grad_input")
    clip_norm = tf.placeholder(tf.float32, None, name="clip_norm")
    start = 0
    grads = []
    for (shape, v) in zip(shapes, var_list):
        size = intprod(shape)
        grads.append(tf.reshape(theta[start:start + size], shape))
        start += size
    # clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm)
    clipped_grads = [tf.clip_by_norm(grad, clip_norm) for grad in grads]
    train_op = opt.apply_gradients(zip(clipped_grads, var_list), name=name)
    return train_op


def get_flat(var_list):
    return tf.concat(
        axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list], name="get_flat")


def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])


class TrainInput(collections.namedtuple(
        "TrainInput", "observation masks target weight")):
    """Inputs for training the Model."""

    @staticmethod
    def stack(train_inputs):
        observation, masks, target, weight = zip(*train_inputs)
        return TrainInput(
            np.array(observation, dtype=np.float32),
            np.array(masks, dtype=np.int32),
            np.array(target),
            np.array(weight))


MAX_RANK = 13
MAX_SUIT = 4
MAX_CARD = 52


class EmbLayer(tfkl.Layer):
    def __init__(self, nn_width, name):
        super(EmbLayer, self).__init__(name=name)
        self.nn_width = nn_width

    def build(self, input_shape):
        self.rank_embs = tfkl.Embedding(
            MAX_RANK, self.nn_width, name="rank_emb", input_length=input_shape[-1])
        self.suit_embs = tfkl.Embedding(
            MAX_SUIT, self.nn_width, name="suit_emb", input_length=input_shape[-1])
        self.point_embs = tfkl.Embedding(
            MAX_CARD, self.nn_width, name="point_emb", input_length=input_shape[-1])

    def call(self, cards, **kwargs):
        # -1 means "no card"
        with tf.device("/cpu:0"):
            valid = tf.expand_dims(tf.cast(cards >= 0, dtype=tf.float32), -1)
            cards_clipped = tf.clip_by_value(cards, 0, MAX_CARD)
            cards_rank = cards_clipped // MAX_SUIT
            cards_suit = cards_clipped % MAX_SUIT

        card_emb = self.rank_embs(cards_rank) + \
            self.suit_embs(cards_suit) + \
            self.point_embs(cards_clipped)
        card_emb = tf.reduce_sum(input_tensor=card_emb * valid, axis=1)
        return card_emb

    def get_config(self):
        return {"nn_width": self.nn_width}


class CFRModel(object):
    """A model for Deep CFR algorithms.
       support models for q-value, value and policy networks.
    """
    valid_model_types = ["normal", "baseline", "dueling",
                         "softmax", "normal_fc", "softmax_fc"]

    def __init__(self, model_type, session, saver, path):
        """Init a model from build_model."""
        self.model_type = model_type
        self._session = session
        self._saver = saver
        self._path = path

        def get_var(name):
            return self._session.graph.get_tensor_by_name(name + ":0")

        def try_get_var(name):
            try:
                return get_var(name)
            except Exception:
                return None

        self._input = get_var("input")
        self._masks = get_var("masks")
        self._training = get_var("training")
        self._output = get_var("output")
        self._value = try_get_var("value")
        self._learning_rate = get_var("learning_rate")
        self._baseline = try_get_var("baseline")
        self._policy = try_get_var("policy")
        self._infer_list = [self._value, self._baseline, self._policy]
        self._infer_list = [i for i in self._infer_list if i is not None]
        self._loss = get_var("loss")
        self._targets = get_var("targets")
        self._baseline_targets = try_get_var("baseline_targets")
        self._weights = get_var("weights")
        self._gradients = get_var("grads")
        self._clip_norm = get_var("clip_norm")
        self._grad_input = get_var("grad_input")
        self._train = self._session.graph.get_operation_by_name("train")

    @classmethod
    def build_model(cls, model_type, input_shape, output_size,
                    nn_width, nn_depth, weight_decay, learning_rate, path, device):
        if model_type not in cls.valid_model_types:
            raise ValueError("Invalid model type")
        g = tf.Graph()
        with g.as_default():
            with tf.device(device):
                cls._define_graph(model_type, input_shape, output_size,
                                  nn_width, nn_depth, weight_decay, learning_rate)
                init = tf.variables_initializer(
                    tf.global_variables(), name="init_all_vars_op")
                get_flat_op = get_flat(tf.trainable_variables())
                set_flat_op, _ = set_flat(tf.trainable_variables())
                with tf.device("/cpu:0"):
                    saver = tf.train.Saver()
            session = tf.Session(
                graph=g, config=tf.ConfigProto(allow_soft_placement=True))
            session.__enter__()
            session.run(init)
        return cls(model_type, session, saver, path)

    @classmethod
    def from_checkpoint(cls, checkpoint, path=None):
        """Load a model from a checkpoint."""
        model = cls.from_graph(checkpoint, path)
        model.load_checkpoint(checkpoint)
        return model

    @classmethod
    def from_graph(cls, metagraph, path=None):
        """Load only the model from a graph or checkpoint."""
        if not os.path.exists(metagraph):
            metagraph += ".meta"
        if not path:
            path = os.path.dirname(metagraph)
        g = tf.Graph()  # Allow multiple independent models and graphs.
        with g.as_default():
            saver = tf.train.import_meta_graph(metagraph)
        session = tf.Session(graph=g)
        session.__enter__()
        session.run("init_all_vars_op")
        return cls("normal", session, saver, path)

    def __del__(self):
        if hasattr(self, "_session") and self._session:
            self._session.close()

    @staticmethod
    def _define_graph(model_type, input_shape, output_size,
                      nn_width, nn_depth, weight_decay, learning_rate):
        # NOTE: nn_depth has no effect here.
        # Inference inputs
        num_cards = input_shape[:-1]
        bets_size = input_shape[-1]
        num_cards = [nc for nc in num_cards if nc]
        input_size = int(np.sum(input_shape))
        print("input_shape = ", input_shape)
        print("num_cards = ", num_cards)
        print("bets_size = ", bets_size)
        observations = tf.placeholder(
            tf.float32, [None, input_size], name="input")
        obs_splits = tf.split(
            observations, num_cards + [bets_size], axis=-1)
        cards = obs_splits[:-1]
        bets = obs_splits[-1]
        masks = tf.placeholder(tf.bool, [None, output_size],
                               name="masks")
        legal_actions = tf.placeholder(tf.bool, [None, output_size],
                                       name="legal_actions")
        training = tf.placeholder(tf.bool, None, name="training")
        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")

        # target placeholder
        if model_type == "normal" or model_type == "softmax" or model_type == "normal_fc"or model_type == "softmax_fc":
            targets = tf.placeholder(dtype=tf.float32, shape=[
                                     None, output_size], name="targets")
        elif model_type == "baseline":
            targets = tf.placeholder(dtype=tf.float32, shape=[
                                     None, output_size + 1], name="targets")
            targets, baseline_targets = tf.split(
                targets, [output_size, 1], axis=-1)
        elif model_type == "dueling":
            targets = tf.placeholder(dtype=tf.float32, shape=[
                                     None, output_size + 1], name="targets")
            value_targets, baseline_targets = tf.split(
                targets, [output_size, 1], axis=-1)
        weights = tf.placeholder(dtype=tf.float32,
                                 shape=[None, 1], name="weights")

        card_embs = []
        if model_type == "normal_fc"or model_type == "softmax_fc":
            for i, card_group in enumerate(cards):
                card_embs.append(tf.reshape(
                    tf.one_hot(tf.cast(card_group, tf.int32), depth=MAX_CARD), [-1, int(MAX_CARD * num_cards[i])]))
        else:
            for i, card_group in enumerate(cards):
                card_embs.append(
                    EmbLayer(nn_width, name="emb_" + str(i))(card_group))

        card_embs = tf.concat(card_embs, axis=1)

        if model_type == "normal_fc"or model_type == "softmax_fc":
            bet_occurred = tf.cast(bets >= 0, tf.float32)
            bet_size = tf.clip_by_value(bets, 0, 1e6)
            bet_feats = tf.concat([bet_size, bet_occurred], axis=1)
            input_z = tf.concat([card_embs, bet_feats], axis=1)
            z = tf.nn.relu(tfkl.Dense(nn_width, name="comb_1")(input_z))
        else:
            x = tf.nn.relu(tfkl.Dense(
                nn_width * 3, name="card_1")(card_embs))
            x = tf.nn.relu(tfkl.Dense(nn_width * 3, name="card_2")(x))
            x = tf.nn.relu(tfkl.Dense(nn_width, name="card_3")(x))
            # -1 means didn"t reach yet.
            bet_occurred = tf.cast(bets >= 0, tf.float32)
            bet_size = tf.clip_by_value(bets, 0, 1e6)
            bet_feats = tf.concat([bet_size, bet_occurred], axis=1)
            y = tf.nn.relu(tfkl.Dense(nn_width, name="bet_1")(bet_feats))
            y = tf.nn.relu(tfkl.Dense(nn_width, name="bet_2")(y) + y)

            z = tf.concat([x, y], axis=1)

            z = tf.nn.relu(tfkl.Dense(nn_width, name="comb_1")(z))
            z = tf.nn.relu(tfkl.Dense(nn_width, name="comb_2")(z) + z)
            z = tf.nn.relu(tfkl.Dense(nn_width, name="comb_3")(z) + z)

        def _output_head(input_z, prefix=""):
            norm_z = tfkl.LayerNormalization()(input_z)
            return tfkl.Dense(output_size, name=prefix + "logit")(norm_z)

        def _baseline_output_head(input_z, prefix=""):
            norm_z = tfkl.LayerNormalization()(input_z)
            return tfkl.Dense(output_size, name=prefix + "logit")(norm_z), tfkl.Dense(1, name=prefix + "baseline")(norm_z)

        def _dueling_output_head(input_z, prefix=""):
            value_z = tf.nn.relu(tfkl.Dense(
                nn_width, name="value")(input_z))
            baseline_z = tf.nn.relu(tfkl.Dense(
                nn_width, name="baseline")(input_z))
            value_logit = tfkl.Dense(
                output_size, name=prefix + "value_logit")(value_z)
            baseline_logit = tfkl.Dense(
                1, name=prefix + "baseline_logit")(baseline_z)
            return value_logit, baseline_logit

        if model_type == "normal" or model_type == "normal_fc":
            logits = _output_head(z)
            logits = tf.where(legal_actions, logits, tf.zeros_like(logits))
            value = tf.identity(logits, name="value")
            output = tf.identity(logits, name="output")
        elif model_type == "baseline":
            logits, baseline = _baseline_output_head(z)
            logits = tf.where(legal_actions, logits, tf.zeros_like(logits))
            value = tf.identity(logits, name="value")
            output = tf.concat([value, baseline], axis=-1, name="output")
        elif model_type == "softmax" or model_type == "softmax_fc":
            logits = _output_head(z)
            logits = tf.where(legal_actions, logits, -
                              1e32 * tf.ones_like(logits))
            policy = tf.nn.softmax(logits, name="policy")
            output = tf.identity(policy, name="output")
        elif model_type == "dueling":
            value_logits, baseline_logits = _dueling_output_head(
                z, "dueling_")
            value_logits = tf.where(
                legal_actions, value_logits, tf.zeros_like(value_logits))
            value = tf.identity(value_logits, name="value")
            baseline = tf.identity(baseline_logits, name="baseline")
            output = tf.concat([value, baseline], axis=-1, name="output")

            # losses
        if model_type == "normal" or model_type == "baseline" or model_type == "normal_fc":
            value_loss = 0.5 * tf.reduce_mean(tf.boolean_mask(
                weights * tf.squared_difference(value, targets), masks))
            value_loss = tf.identity(value_loss, name="value_loss")
        if model_type == "normal" or model_type == "normal_fc":
            loss = tf.identity(value_loss, name="loss")
        elif model_type == "baseline":
            baseline_loss = 0.5 * tf.reduce_mean(
                weights * tf.squared_difference(baseline, baseline_targets))
            baseline_loss = tf.identity(baseline_loss, name="baseline_loss")
            loss = tf.identity(value_loss + baseline_loss, name="loss")
        elif model_type == "softmax" or model_type == "softmax_fc":
            policy_loss = tf.reduce_mean(
                weights * tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=targets, logits=logits))
            loss = tf.identity(policy_loss, name="loss")
        elif model_type == "dueling":
            value_loss = 0.5 * tf.reduce_mean(tf.boolean_mask(
                weights * tf.squared_difference(value, value_targets), masks))
            values_masked = tf.where(
                legal_actions, value_logits, -np.infty * tf.ones_like(value_logits))
            max_value = tf.reduce_max(values_masked, axis=-1, keepdims=True)
            thres = tf.square(tf.nn.relu(
                tf.stop_gradient(values_masked) - baseline))
            thres_masked = tf.reduce_sum(thres, axis=-1, keepdims=True)
            thres_masked -= tf.nn.relu(baseline - tf.stop_gradient(max_value))
            thres_loss = 0.5 * tf.reduce_mean(
                weights * tf.squared_difference(thres_masked, baseline_targets))
            loss = tf.math.add(value_loss, thres_loss, name="loss")

        # optimzer
        if model_type == "normal_fc"or model_type == "softmax_fc":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate)
        tvs = tf.trainable_variables()
        grads = flatgrad(loss, tvs, clip_norm=None)
        grads = tf.identity(grads, name="grads")
        train_op = apply_gradients(tvs, optimizer, name="train")
        grads = tf.gradients(loss, tvs)
        clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
        sim_train_op = optimizer.apply_gradients(
            zip(clipped_grads, tvs), name="sim_train")

    def inference(self, obs, masks):
        return self._session.run(self._output,
                                 feed_dict={self._input: obs, self._masks: masks})

    def update(self, train_inputs, lr):
        batch = TrainInput.stack(train_inputs)
        for i in range(1):
            _, loss = self._session.run(["sim_train", self._loss],
                                        feed_dict={self._input: batch.observation,
                                                   self._masks: batch.masks,
                                                   self._weights: batch.weight,
                                                   self._targets: batch.target,
                                                   self._learning_rate: lr})
            # self._session.run(self._train,
            # feed_dict={self._grad_input: grad,
            #     self._clip_norm: 1.0,
            #     self._learning_rate: lr})
        return loss

    @property
    def num_trainable_variables(self):
        return sum(np.prod(v.shape) for v in tf.trainable_variables())

    def print_trainable_variables(self):
        for v in tf.trainable_variables():
            print("{}: {}".format(v.name, v.shape))

    def write_graph(self, filename):
        full_path = os.path.join(self._path, filename)
        tf.train.export_meta_graph(
            graph_def=self._session.graph_def, saver_def=self._saver.saver_def,
            filename=full_path, as_text=False)
        # tf.io.write_graph(self._session.graph_def, self._path, filename, as_text=False)
        return full_path

    def save_checkpoint(self, step):
        return self._saver.save(
            self._session,
            os.path.join(self._path, "checkpoint"),
            global_step=step)

    def load_checkpoint(self, path):
        return self._saver.restore(self._session, path)
