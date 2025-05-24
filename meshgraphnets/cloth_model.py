# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model for FlagSimple."""

import sonnet as snt
import tensorflow.compat.v1 as tf

from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization


class Model(snt.AbstractModule):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, name='Model'):
    super(Model, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = normalization.Normalizer(
          size=3, name='output_normalizer')
      self._node_normalizer = normalization.Normalizer(
          size=3+common.NodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = normalization.Normalizer(
          size=7, name='edge_normalizer')  # 2D coord + 3D coord + 2*length = 7

  def _build_graph(self, inputs, is_training):
    """
    Builds input graph.
    
    The inputs of the model is a dict:
    {
    'cells'             : [3028, 3] int32
    'mesh_pos'          : [1579, 2] float32
    'node_type'         : [1579, 1] int32
    "world_pos"         : [1579, 3] float32, ← current
    "prev|world_pos"    : [1579, 3] float32, ← (optional history)
    "target|world_pos"  : [1579, 3] float32, ← next-step label
    }
    """
    # construct graph nodes
    # (1579, 3)
    velocity = inputs['world_pos'] - inputs['prev|world_pos']
    # (1579, 9)
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    # (1579, 12)
    node_features = tf.concat([velocity, node_type], axis=-1)

    # construct graph edges
    # (max 9084,), the real shape is unknown until runtime because we need to remove dups
    senders, receivers = common.triangles_to_edges(inputs['cells'])
    # (max 9084, 3)
    relative_world_pos = (tf.gather(inputs['world_pos'], senders) -
                          tf.gather(inputs['world_pos'], receivers))
    # (max 9084, 2)
    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))
    # (max 9084, 7)
    edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])

  def _build(self, inputs):
    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output)

  @snt.reuse_variables
  def loss(self, inputs):
    """L2 loss on position."""
    graph = self._build_graph(inputs, is_training=True)
    network_output = self._learned_model(graph)

    # build target acceleration
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    target_position = inputs['target|world_pos']
    target_acceleration = target_position - 2*cur_position + prev_position
    target_normalized = self._output_normalizer(target_acceleration)

    # build loss
    loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
    loss = tf.reduce_mean(error[loss_mask])
    return loss

  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs."""
    acceleration = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    position = 2*cur_position + acceleration - prev_position
    return position
