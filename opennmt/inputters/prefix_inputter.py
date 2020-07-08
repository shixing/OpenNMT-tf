
import abc
import collections
import os
import six

import numpy as np
import tensorflow as tf

from tensorboard.plugins import projector
from google.protobuf import text_format

from opennmt import constants, tokenizers
from opennmt.data import text
from opennmt.data.vocab import Vocab
from opennmt.inputters.inputter import Inputter
from opennmt.layers import common
from opennmt.utils import misc
from opennmt.data import dataset as dataset_util

from collections import deque

class TrieNode:
  def __init__(self):
    self.children = {}
    self.vocab_id = -1
    self.all_children_vocab_ids = None

  def get_all_children_vocab_ids(self):
    if self.all_children_vocab_ids is not None:
      return self.all_children_vocab_ids
    ids = []
    if self.vocab_id != -1:
      ids.append(self.vocab_id)
    for c in self.children:
      child_node = self.children[c]
      ids += child_node.get_all_children_vocab_ids()
    self.all_children_vocab_ids = ids
    return ids

  def add(self, suffix, vocab_id):
    if suffix == '':
      self.vocab_id = vocab_id
    else:
      c = suffix[0]
      if c not in self.children:
        self.children[c] = TrieNode()
      self.children[c].add(suffix[1:], vocab_id)

  def search(self, suffix):
    if suffix == "":
      return self
    c = suffix[0]
    if c in self.children:
      return self.children[c].search(suffix[1:])


def load_vocab(fn_vocab):
  vocab_to_id = {}
  id_to_vocab = {}
  with open(fn_vocab) as f:
    for idx, line in enumerate(f):
      line = line[:-1]
      vocab_to_id[line] = idx
      id_to_vocab[idx] = line

  return vocab_to_id, id_to_vocab

def build_trie(vocab_to_id):
  trie = TrieNode()
  for vocab in vocab_to_id:
    vocab_id = vocab_to_id[vocab]
    trie.add(vocab, vocab_id)
  return trie

def load_vocab_trie(fn_vocab):
  vocab_to_id, id_to_vocab = load_vocab(fn_vocab)
  trie = build_trie(vocab_to_id)
  return trie, vocab_to_id, id_to_vocab


def build_allowed_ids(orig_prefix, trie, vocab_to_id, allowed_ids_dict, overwrite_key=None):
  # {prefix: (allowed_ids, suffix)}
  allowed_ids = {}
  for end in range(1, len(orig_prefix)):
    prefix = orig_prefix[:end]
    suffix = orig_prefix[end:]
    if prefix in vocab_to_id:
      vocab_id = vocab_to_id[prefix]
      if suffix not in allowed_ids_dict:
        build_allowed_ids(suffix, trie, vocab_to_id, allowed_ids_dict)
      if suffix in allowed_ids_dict and len(allowed_ids_dict[suffix]) > 0:
        allowed_ids[vocab_id] = suffix

  node = trie.search(orig_prefix)
  if node is not None:
    ids = node.get_all_children_vocab_ids()
    for id in ids:
      allowed_ids[id] = "ALL"

  if overwrite_key is not None:
    allowed_ids_dict[overwrite_key] = allowed_ids
  else:
    allowed_ids_dict[orig_prefix] = allowed_ids

def allowed_ids_dict_to_matrix(allowed_id_dict):


  _emission_matrix = []
  _transition_matrix = []
  length_matrix = []

  state_idx = 0
  start_key = "[BOS]"
  suffix_to_idx = {}

  # breath-first search

  q = deque()
  if start_key in allowed_id_dict and len(allowed_id_dict[start_key]) > 0:
    q.append((start_key, 0))
  max_length = 0

  while len(q) > 0:
    key, idx = q.popleft()
    vocab_ids = []
    next_states = []
    for vocab_id in allowed_id_dict[key]:
      suffix = allowed_id_dict[key][vocab_id]
      if suffix == "ALL":
        next_state_id = -1
      else:
        if suffix not in suffix_to_idx:
          state_idx += 1
          suffix_to_idx[suffix] = state_idx
          q.append((suffix, state_idx))
        next_state_id = suffix_to_idx[suffix]
      vocab_ids.append(vocab_id)
      next_states.append(next_state_id)
    _emission_matrix.append(vocab_ids)
    _transition_matrix.append(next_states)
    length_matrix.append(len(vocab_ids))
    max_length = max(max_length, len(vocab_ids))

  n_state = len(_emission_matrix)
  emission_matrix = np.ones((n_state, max_length), dtype=np.int32) * -2
  transition_matrix = np.ones((n_state, max_length), dtype=np.int32) * -2
  for row in range(n_state):
    for col in range(len(_emission_matrix[row])):
      emission_matrix[row][col] = _emission_matrix[row][col]
      transition_matrix[row][col] = _transition_matrix[row][col]

  init_state = 0
  if len(length_matrix) == 0:
    init_state = -2

  length_matrix = np.array(length_matrix)

  return emission_matrix, transition_matrix, length_matrix, init_state

def print_allowed_ids_dict(allowed_ids_dict, id_to_vocab):
  for prefix in allowed_ids_dict:
    allowed_ids = allowed_ids_dict[prefix]
    print(prefix)
    for idx in allowed_ids:
      suffix = allowed_ids[idx]
      vocab = id_to_vocab[idx]
      print(f'\t{idx}-{vocab}\t\t{suffix}')

def get_emission_transition_matrix(prefix, trie, vocab_to_id, id_to_vocab):
  allowed_ids_dict = {}
  build_allowed_ids(prefix, trie, vocab_to_id, allowed_ids_dict, overwrite_key="[BOS]")
  #print_allowed_ids_dict(allowed_ids_dict, id_to_vocab)
  emission_matrix, transition_matrix, length_matrix, init_state = allowed_ids_dict_to_matrix(allowed_ids_dict)
  return emission_matrix, transition_matrix, length_matrix, init_state

@six.add_metaclass(abc.ABCMeta)
class PrefixInputter(Inputter):
  """Input Class to process prefix."""
  def __init__(self, fn_vocab, **kwargs):
    super(PrefixInputter, self).__init__(**kwargs)
    self.asset_prefix = None

    vocab_to_id, id_to_vocab = load_vocab(fn_vocab)
    trie = build_trie(vocab_to_id)
    self.vocab_to_id = vocab_to_id
    self.id_to_vocab = id_to_vocab
    self.trie = trie

  def make_dataset(self, data_file, training=None):
    """Creates the base dataset required by this inputter.

    Args:
      data_file: The data file.
      training: Run in training mode.

    Returns:
      A non transformed ``tf.data.Dataset`` instance.
    """
    ems = []
    tms = []
    lms = []
    iss = []
    with open(data_file) as f:
      for line in f:
        prefix = line.strip()
        emission_matrix, transition_matrix, length_matrix, init_state = get_emission_transition_matrix(prefix, self.trie, self.vocab_to_id, self.id_to_vocab)
        if len(length_matrix) == 0:
          print(prefix)
        ems.append(emission_matrix)
        tms.append(transition_matrix)
        lms.append(length_matrix)
        iss.append(init_state)

    d1 = tf.data.Dataset.from_generator(ems.__iter__, tf.int32, tf.TensorShape([None, None]))
    d2 = tf.data.Dataset.from_generator(tms.__iter__, tf.int32, tf.TensorShape([None, None]))
    d3 = tf.data.Dataset.from_generator(lms.__iter__, tf.int32, tf.TensorShape([None]))
    d4 = tf.data.Dataset.from_generator(iss.__iter__, tf.int32, tf.TensorShape([]))

    return tf.data.Dataset.zip((d1,d2,d3, d4))

  def make_inference_dataset(self,
                             features_file,
                             features_inputter,
                             prefix_file,
                             batch_size,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    """Builds a dataset to be used for inference.

    For evaluation and training datasets, see
    :class:`opennmt.inputters.ExampleInputter`.

    Args:
      features_file: The test file.
      batch_size: The batch size to use.
      length_bucket_width: The width of the length buckets to select batch
        candidates from (for efficiency). Set ``None`` to not constrain batch
        formation.
      num_threads: The number of elements processed in parallel.
      prefetch_buffer_size: The number of batches to prefetch asynchronously. If
        ``None``, use an automatically tuned value.

    Returns:
      A ``tf.data.Dataset``.

    See Also:
      :func:`opennmt.data.inference_pipeline`
    """
    features_dataset = features_inputter.make_dataset(features_file)
    prefix_dataset = self.make_dataset(prefix_file)
    dataset = tf.data.Dataset.zip((features_dataset, prefix_dataset))

    def map_func(source, prefix):
      #source, prefix = item
      feature = features_inputter.make_features(source)
      prefix_feature = self.make_features(prefix)
      feature.update(prefix_feature)
      return feature

    dataset = dataset.map(map_func, num_parallel_calls=num_threads)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset

  def input_signature(self, features_inputter):
    """Returns the input signature of this inputter."""
    source_input_signature = features_inputter.input_signature()
    prefix_input_signature = {
      "emission_matrix": tf.TensorSpec([None, None, None], tf.int32),
      "transition_matrix": tf.TensorSpec([None, None, None], tf.int32),
      "length_matrix": tf.TensorSpec([None, None], tf.int32),
      "init_state": tf.TensorSpec([None], tf.int32)

    }
    prefix_input_signature.update(source_input_signature)
    return prefix_input_signature

  def make_features(self, element=None, features=None, training=None):
    """Creates features from data.

    This is typically called in a data pipeline (such as ``Dataset.map``).
    Common transformation includes tokenization, parsing, vocabulary lookup,
    etc.

    This method accept both a single :obj:`element` from the dataset or a
    partially built dictionary of :obj:`features`.

    Args:
      element: An element from the dataset returned by
        :meth:`opennmt.inputters.Inputter.make_dataset`.
      features: An optional and possibly partial dictionary of features to
        augment.
      training: Run in training mode.

    Returns:
      A dictionary of ``tf.Tensor``.
    """
    if features is None:
      features = {}

    features['emission_matrix'] = element[0]
    features['transition_matrix'] = element[1]
    features['length_matrix'] = element[2]
    features['init_state'] = element[3]
    return features



