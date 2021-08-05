#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import argparse
import logging

import tensorflow as tf
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from onnx import shape_inference
from tf2onnx import tfonnx, optimizer, tf_loader
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

import onnx_utils

logging.basicConfig(level=logging.INFO)
logging.getLogger("ModelHelper").setLevel(logging.INFO)
log = logging.getLogger("ModelHelper")


class ModelGraphSurgeon:
    def __init__(self, saved_model_path, pipeline_config_path):
        """
        Constructor of the Model Graph Surgeon object, to do the conversion of an TFOD saved model
        to an ONNX-TensorRT parsable model.
        :param saved_model_path: The path pointing to the TensorFlow saved model to load.
        :param pipeline_config_path: The path pointing to the TensorFlow pipeline.config to load.
        """

        saved_model_path = os.path.realpath(saved_model_path)
        assert os.path.exists(saved_model_path)

        # Use tf2onnx to convert saved model to an initial ONNX graph.
        graph_def, inputs, outputs = tf_loader.from_saved_model(saved_model_path, None, None, "serve",
                                                                ["serving_default"])
        log.info("Loaded saved model from {}".format(saved_model_path))
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name="")
        with tf_loader.tf_session(graph=tf_graph):
            onnx_graph = tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs, opset=11)
        onnx_model = optimizer.optimize_graph(onnx_graph).make_model("Converted from {}".format(saved_model_path))
        self.graph = gs.import_onnx(onnx_model)
        assert self.graph
        log.info("TF2ONNX graph created successfully")

        # Fold constants via ONNX-GS that TF2ONNX may have missed.
        self.graph.fold_constants()
        
        # Pipeline config parsing.
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)

        # If your model is SSD, get characteristics accordingly from pipeline.config file.
        if pipeline_config.model.HasField("ssd"):
            # Getting model characteristics.
            self.model = str(pipeline_config.model.ssd.feature_extractor.type)
            self.height = int(pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height)
            self.width = int(pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width)
            self.first_stage_nms_score_threshold = float(pipeline_config.model.ssd.post_processing.batch_non_max_suppression.score_threshold)
            self.first_stage_nms_iou_threshold = float(pipeline_config.model.ssd.post_processing.batch_non_max_suppression.iou_threshold)
            self.first_stage_max_proposals = int(pipeline_config.model.ssd.post_processing.batch_non_max_suppression.max_detections_per_class)

        # If your model is Faster R-CNN get it's characteristics from pipeline.config file.
        elif pipeline_config.model.HasField("faster_rcnn"):  
            self.model = str(pipeline_config.model.faster_rcnn.feature_extractor.type) 
            # There are two types of image_resizers, select accordingly from pipeline.config file.
            if pipeline_config.model.faster_rcnn.image_resizer.HasField("fixed_shape_resizer"):
                self.height = int(pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height)
                self.width = int(pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width)
            elif pipeline_config.model.faster_rcnn.image_resizer.HasField("keep_aspect_ratio_resizer"): 
                self.height = int(pipeline_config.model.faster_rcnn.image_resizer.keep_aspect_ratio_resizer.max_dimension)
                self.width = self.height
            else:
                log.info("Image resizer config is not supported")
                sys.exit(1)

            # Getting model characteristics
            self.first_stage_nms_score_threshold = float(pipeline_config.model.faster_rcnn.first_stage_nms_score_threshold)       
            self.first_stage_nms_iou_threshold = float(pipeline_config.model.faster_rcnn.first_stage_nms_iou_threshold)
            self.first_stage_max_proposals = int(pipeline_config.model.faster_rcnn.first_stage_max_proposals)
            self.initial_crop_size = int(pipeline_config.model.faster_rcnn.initial_crop_size)
            self.second_score_threshold = float(pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.score_threshold)
            self.second_iou_threshold = float(pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.iou_threshold)

        else: 
            log.info("Given pipeline.config file is not supported")
            sys.exit(1)

        #print(self.model)
        #print(self.height)
        #print(self.width)
        #print(self.first_stage_nms_score_threshold)
        #print(self.first_stage_nms_iou_threshold)
        #print(self.first_stage_max_proposals)
        #print(self.initial_crop_size)
        #print(self.second_score_threshold)
        #print(self.second_iou_threshold)
        #print(self.first_stage_max_proposals)

        self.batch_size = None

    def infer(self):
        """
        Sanitize the graph by cleaning any unconnected nodes, do a topological resort, and fold constant inputs values.
        When possible, run shape inference on the ONNX graph to determine tensor shapes.
        """
        for i in range(6):
            count_before = len(self.graph.nodes)

            self.graph.cleanup().toposort()
            try:
                for node in self.graph.nodes:
                    for o in node.outputs:
                        o.shape = None
                model = gs.export_onnx(self.graph)
                model = shape_inference.infer_shapes(model)
                self.graph = gs.import_onnx(model)
            except Exception as e:
                log.info("Shape inference could not be performed at this time:\n{}".format(e))
            try:
                self.graph.fold_constants(fold_shapes=True)
            except TypeError as e:
                log.error("This version of ONNX GraphSurgeon does not support folding shapes, please upgrade your "
                          "onnx_graphsurgeon module. Error:\n{}".format(e))
                raise

            count_after = len(self.graph.nodes)
            if count_before == count_after:
                # No new folding occurred in this iteration, so we can stop for now.
                break

    def save(self, output_path):
        """
        Save the ONNX model to the given location.
        :param output_path: Path pointing to the location where to write out the updated ONNX model.
        """
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        output_path = os.path.realpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save(model, output_path)
        log.info("Saved ONNX model to {}".format(output_path))

    def add_debug_output(self, debug):
        """
        Add a debug output to a given node. 
        :param debug: Name of the output you would like to debug.
        """
        tensors = self.graph.tensors()
        for n, name in enumerate(debug):
            if name not in tensors:
                log.warning("Could not find tensor '{}'".format(name))
            debug_tensor = gs.Variable(name="debug:{}".format(n), dtype=tensors[name].dtype)
            debug_node = gs.Node(op="Identity", name="debug_{}".format(n), inputs=[tensors[name]], outputs=[debug_tensor])
            self.graph.nodes.append(debug_node)
            self.graph.outputs.append(debug_tensor)
            log.info("Adding debug output '{}' for graph tensor '{}'".format(debug_tensor.name, name))

    def update_preprocessor(self, batch_size):
        """
        Remove all the pre-processing nodes in the ONNX graph and leave only the image normalization essentials.
        :param batch_size: The batch size to use for the ONNX graph.
        """
        # Update batch size.
        self.batch_size = batch_size

        # Set input tensor shape (NHWC).
        input_shape = [None] * 4
        input_shape[0] = self.batch_size
        input_shape[1] = self.height
        input_shape[2] = self.width
        input_shape[3] = 3
        
        assert len(input_shape) == 4
        for i in range(len(input_shape)):
            assert input_shape[i] >= 1
        input_format = None
        if input_shape[1] == 3:
            input_format = "NCHW"
        if input_shape[3] == 3:
            input_format = "NHWC"
        assert input_format in ["NCHW", "NHWC"]

        self.graph.inputs[0].shape = input_shape
        self.graph.inputs[0].dtype = np.float32

        self.infer()
        log.info("ONNX graph input shape: {} [NCHW format set]".format(self.graph.inputs[0].shape))

        # Find the initial nodes of the graph, whatever the input is first connected to, and disconnect them.
        for node in [node for node in self.graph.nodes if self.graph.inputs[0] in node.inputs]:
            node.inputs.clear()

        # Get input tensor.
        # Convert to NCHW format if needed.
        input_tensor = self.graph.inputs[0]
        if input_format == "NHWC":
            input_tensor = self.graph.transpose("preprocessor/transpose", input_tensor, [0, 3, 1, 2])

        # Mobilenets' and inception's backbones preprocessor.
        if (self.model == 'ssd_mobilenet_v2_keras' or self.model == 'ssd_mobilenet_v1_fpn_keras' 
            or self.model == 'ssd_mobilenet_v2_fpn_keras' or self.model == "faster_rcnn_inception_resnet_v2_keras"):
            mul_const = np.expand_dims(np.asarray([0.007843137718737125, 0.007843137718737125, 0.007843137718737125], dtype=np.float32), axis=(0, 2, 3))
            sub_const = np.expand_dims(np.asarray([1, 1, 1], dtype=np.float32), axis=(0, 2, 3))
            mul_out = self.graph.elt_const("Mul", "preprocessor/scale", input_tensor, mul_const)
            sub_out = self.graph.elt_const("Sub", "preprocessor/mean", mul_out, sub_const)

        # Resnet backbones' preprocessor.
        elif (self.model == 'ssd_resnet50_v1_fpn_keras' or self.model == 'ssd_resnet101_v1_fpn_keras' 
            or self.model == 'ssd_resnet152_v1_fpn_keras' or self.model == 'faster_rcnn_resnet50_keras' 
            or self.model == 'faster_rcnn_resnet101_keras' or self.model == 'faster_rcnn_resnet152_keras'):
            sub_const = np.expand_dims(np.asarray([123.68000030517578, 116.77899932861328, 103.93900299072266], dtype=np.float32), axis=(0, 2, 3))
            sub_out = self.graph.elt_const("Sub", "preprocessor/mean", input_tensor, sub_const)
        
        # Model is not supported.
        else:
            log.info("This model: {} is not supported".format(self.model))
            sys.exit(1)

        # Find first Conv node and connect preprocessor directly to it.
        stem_name = "StatefulPartitionedCall/"
        stem = [node for node in self.graph.nodes if node.op == "Conv" and stem_name in node.name][0]
        log.info("Found {} node '{}' as stem entry".format(stem.op, stem.name))
        stem.inputs[0] = sub_out[0]

        # Get rid of the last node in one of the preprocessing branches with first TensorListStack parent node
        concat_name = "StatefulPartitionedCall/"
        concat_node = [node for node in self.graph.nodes if node.op == "Concat" and concat_name in node.name][0]
        concat_node.outputs = []

        # Get rid of the last node in second preprocessing branch with parent second TensorListStack node:
        cast_name = "StatefulPartitionedCall/"
        cast_node = [node for node in self.graph.nodes if node.op == "Tile" and cast_name in node.name][0]
        cast_node.outputs = []

        # Reshape nodes tend to update the batch dimension to a fixed value of 1, they should use the batch size instead.
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            if type(node.inputs[1]) == gs.Constant and node.inputs[1].values[0] == 1:
                node.inputs[1].values[0] = self.batch_size

        self.infer()


    def process_graph(self, threshold=None):
        """
        Processes the graph to replace the NMS operations by BatchedNMS_TRT TensorRT plugin nodes and
        cropAndResize operations by CropAndResize plugin node.
        :param threshold: Override the score threshold value. If set to None, use the value in the graph.
        """

        def find_head_end(head_name, descendant, end_op):
            # This helper function finds ends of Class Net and Box Net, based on a model type. 
            # :param head_name: This is a common name that nodes in either Class or Box Nets start with.
            # :param descendant: Descendant of head_name, identified by operation (Transpose, MatMul, etc.).
            # :param end_op: Operation of a node you would like to get in the end of each Net.
            # These end_op nodes bring together prediction data based on type of model.
            # The Class Net end node will have shape [batch_size, num_anchors, num_classes],
            # and the Box Net end node has the shape [batch_size, num_anchors, 4].
            # These end nodes can be be found by searching for all end_op's operation nodes and checking if the node two
            # steps above in the graph has a name that begins with one of head_names for Class Net and Box Net respectively.
            for node in [node for node in self.graph.nodes if node.op == descendant and head_name in node.name]:
                target_node = self.graph.find_descendant_by_op(node, end_op)
                log.info("Found {} node '{}' as the tip of {}".format(target_node.op, target_node.name, head_name))
                return target_node

        def extract_anchors_tensor(split):
            # This will find the anchors that have been hardcoded somewhere within the ONNX graph.
            # The function will return a gs.Constant that can be directly used as an input to the NMS plugin.
            # The anchor tensor shape will be [1, num_anchors, 4]. Note that '1' is kept as first dim, regardless of
            # batch size, as it's not necessary to replicate the anchors for all images in the batch.

            # The anchors are available (one per coordinate) hardcoded as constants within certain box decoder nodes.
            # Each of these four constants have shape [1, num_anchors], so some numpy operations are used to expand the
            # dims and concatenate them as needed.

            # These constants can be found by starting from the Box Net's split operation , and for each coordinate,
            # walking down in the graph until either an Add or specific Mul node is found. The second input on this nodes will
            # be the anchor data required.

            # Get Add anchor nodes
            def get_anchor_add(output_idx, op):
                node = self.graph.find_descendant_by_op(split.o(0, output_idx), op)
                assert node
                val = np.squeeze(node.inputs[1].values)
                return np.expand_dims(val.flatten(), axis=(0, 2))

            # Get Mul anchor nodes
            def get_anchor_mul(name, op):
                node = [node for node in self.graph.nodes if node.op == op and name == node.name][0]
                assert node
                val = np.squeeze(node.inputs[1].values)
                return np.expand_dims(val.flatten(), axis=(0, 2))

            
            anchors_y = get_anchor_add(0, "Add")
            anchors_x = get_anchor_add(1, "Add")
            anchors_h = None
            anchors_w = None

            # Based on a model type, naming of Mul nodes is slightly different, this will be improved in future to exclude branching.
            if "ssd" in self.model:
                anchors_h = get_anchor_mul("StatefulPartitionedCall/Postprocessor/Decode/mul_1","Mul")
                anchors_w = get_anchor_mul("StatefulPartitionedCall/Postprocessor/Decode/mul","Mul")
            elif "faster_rcnn" in self.model:
                anchors_h = get_anchor_mul("StatefulPartitionedCall/Decode/mul_1","Mul")
                anchors_w = get_anchor_mul("StatefulPartitionedCall/Decode/mul","Mul") 

            batched_anchors = np.concatenate([anchors_y, anchors_x, anchors_h, anchors_w], axis=2)
            # Identify num of anchors without repetitions.
            num_anchors = int(batched_anchors.shape[1]/self.batch_size)
            # Trim total number of anchors in order to not have copies introduced by growing number of batch_size.
            anchors = batched_anchors[0:num_anchors,0:num_anchors]
            return gs.Constant(name="nms/anchors:0", values=anchors)

        self.infer()

        def first_nms(background_class, score_activation, threshold):
            """
            Updates the graph to replace the NMS op by BatchedNMS_TRT TensorRT plugin node.
            :param background_class: Set EfficientNMS_TRT's background_class atribute. 
            :param score_activation: Set EfficientNMS_TRT's score_activation atribute. 
            """
            # Identify Class Net and Box Net head names based on model type.
            if self.model == 'ssd_mobilenet_v2_keras':
                head_names = ['StatefulPartitionedCall/BoxPredictor/ConvolutionalClassHead_', 
                'StatefulPartitionedCall/BoxPredictor/ConvolutionalBoxHead_']
            elif (self.model == 'ssd_mobilenet_v1_fpn_keras' or self.model == 'ssd_mobilenet_v2_fpn_keras' or 
                self.model == 'ssd_resnet50_v1_fpn_keras' or self.model == 'ssd_resnet101_v1_fpn_keras' or 
                self.model == 'ssd_resnet152_v1_fpn_keras'):
                head_names = ['StatefulPartitionedCall/WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead',
                'StatefulPartitionedCall/WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead']
            elif (self.model == 'faster_rcnn_resnet50_keras' or self.model == 'faster_rcnn_resnet101_keras' or 
                self.model == 'faster_rcnn_resnet152_keras' or self.model == "faster_rcnn_inception_resnet_v2_keras"):
                head_names = ['StatefulPartitionedCall/FirstStageBoxPredictor/ConvolutionalClassHead_0/ClassPredictor',
            'StatefulPartitionedCall/FirstStageBoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor']

            class_net = None
            box_net = None

            # Getting SSD's Class and Box Nets final tensors.
            if "ssd" in self.model:
                # Find the concat node at the end of the class net (multi-scale class predictor).
                class_net = find_head_end(head_names[0], "Transpose", "Concat")

                # SSD's slice operation to adjust third dimension of Class Net's last node tensor (adjusting class values).
                slice_start = np.asarray([1], dtype=np.int64)
                slice_end = np.asarray([91], dtype=np.int64)
                # Second list element but third tensor dimension.
                slice_axes = np.asarray([2], dtype=np.int64)
                slice_out = self.graph.elt_const_slice("Slice", head_names[0]+"/slicer", class_net.outputs[0], slice_start, slice_end, slice_axes)

                # Final Class Net tensor.
                class_net_tensor = slice_out[0]

                if self.model == 'ssd_mobilenet_v2_keras':
                    # Find the squeeze node at the end of the box net (multi-scale localization predictor).
                    box_net = find_head_end(head_names[1], "Transpose", "Concat")
                    box_net_squeeze = self.graph.find_descendant_by_op(box_net, "Squeeze")
                    box_net_output = box_net_squeeze.outputs[0]

                elif (self.model == 'ssd_mobilenet_v1_fpn_keras' or self.model == 'ssd_mobilenet_v2_fpn_keras' or 
                    self.model == 'ssd_resnet50_v1_fpn_keras' or self.model == 'ssd_resnet101_v1_fpn_keras' or 
                    self.model == 'ssd_resnet152_v1_fpn_keras'):
                    # Find the concat node at the end of the box net (multi-scale localization predictor).
                    box_net = find_head_end(head_names[1], "Transpose", "Concat")
                    box_net_output = box_net.outputs[0]

                # 0.1, 0.1, 0.2, 0.2 are localization head variance numbers, they scale box_net_output in order to get accurate coordinates.
                scale_adj = np.expand_dims(np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32), axis=(0, 1))
                scale_out = self.graph.elt_const("Mul", head_names[1]+"/scale", box_net_output, scale_adj)

                # Final Box Net tensor.
                box_net_tensor = scale_out[0]
    
            # Getting Faster R-CNN's 1st Class and Box Nets tensors.
            elif "faster_rcnn" in self.model:
                if (self.model == 'faster_rcnn_resnet50_keras' or self.model == 'faster_rcnn_resnet101_keras' or 
                    self.model == 'faster_rcnn_resnet152_keras' or self.model == "faster_rcnn_inception_resnet_v2_keras"):
                    # Find the softmax node at the end of the class net (multi-scale class predictor).
                    class_net = find_head_end(head_names[0], "Transpose", "Softmax")

                    # Final Class Net tensor
                    class_net_tensor = class_net.outputs[0] 

                    # Find the reshape node at the end of the box net (multi-scale localization predictor).
                    box_net = find_head_end(head_names[1], "Transpose", "Reshape")
                    # Final Box Net tensor.
                    box_net_output = box_net.outputs[0]

                    #Insert a squeeze node
                    squeeze_node = self.graph.squeeze(head_names[1]+"/squeeze", box_net_output)
                    # 0.1, 0.1, 0.2, 0.2 are localization head variance numbers, they scale box_net_output, in order to get accurate coordinates.
                    scale_adj = np.expand_dims(np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32), axis=(0, 1))
                    scale_out = self.graph.elt_const("Mul", head_names[1]+"/scale", squeeze_node, scale_adj)

                    # Final Box Net tensor.
                    box_net_tensor = scale_out[0]


            # 3. Find the split node that separates the box net coordinates and feeds them into the box decoder.
            box_net_split = self.graph.find_descendant_by_op(box_net, "Split")
            assert box_net_split and len(box_net_split.outputs) == 4

            # Set score threshold
            score_threshold = self.first_stage_nms_score_threshold if threshold is None else threshold

            # NMS Inputs and Attributes
            # NMS expects these shapes for its input tensors:
            # box_net: [batch_size, number_boxes, 4]
            # class_net: [batch_size, number_boxes, number_classes]
            # anchors: [1, number_boxes, 4] (if used)
            nms_op = None
            nms_attrs = None
            nms_inputs = None

            # EfficientNMS TensorRT Plugin is suitable for our use case.
            # Fusing the decoder will always be faster, so this is the default NMS method supported. In this case,
            # three inputs are given to the NMS TensorRT node:
            # - The box predictions (from the Box Net node found above)
            # - The class predictions (from the Class Net node found above)
            # - The default anchor coordinates (from the extracted anchor constants)
            # As the original tensors from given model will be used, the NMS code type is set to 1 (Center+Size),
            # because this is the internal box coding format used by the network.
            anchors_tensor = extract_anchors_tensor(box_net_split)
            nms_inputs = [box_net_tensor, class_net_tensor, anchors_tensor]
            nms_op = "EfficientNMS_TRT"
            nms_attrs = {
                'plugin_version': "1",
                'background_class': background_class,
                'max_output_boxes': self.first_stage_max_proposals,
                'score_threshold': max(0.01, score_threshold),
                'iou_threshold': self.first_stage_nms_iou_threshold,
                'score_activation': score_activation,
                'box_coding': 1,
            }
            nms_output_classes_dtype = np.int32

            # NMS Outputs.
            nms_output_num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=[self.batch_size, 1])
            nms_output_boxes = gs.Variable(name="detection_boxes", dtype=np.float32,
                                           shape=[self.batch_size, self.first_stage_max_proposals, 4])
            nms_output_scores = gs.Variable(name="detection_scores", dtype=np.float32,
                                            shape=[self.batch_size, self.first_stage_max_proposals])
            nms_output_classes = gs.Variable(name="detection_classes", dtype=nms_output_classes_dtype,
                                             shape=[self.batch_size, self.first_stage_max_proposals])

            nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

            # Create the NMS Plugin node with the selected inputs. 
            self.graph.plugin(
                op=nms_op,
                name="nms/non_maximum_suppression_first",
                inputs=nms_inputs,
                outputs=nms_outputs,
                attrs=nms_attrs)
            log.info("Created NMS plugin '{}' with attributes: {}".format(nms_op, nms_attrs))

            # If model type is SSD, then you are done with conversion and optimizations.
            if "ssd" in self.model:
                self.graph.outputs = nms_outputs
                self.infer()
                return None
            # If model is Faster R-CNN, then you continue with conversion and optimizations,
            # next step is CropAndResize.     
            elif "faster_rcnn" in self.model:
                return nms_outputs[1]

        def crop_and_resize(input):
            """
            Updates the graph to replace the cropAndResize op by CropAndResize TensorRT plugin node.
            :param input: Input tensor is the output from previous first_nms() step. 
            """

            # Locate the last Relu node of the first backbone (pre 1st NMS). Relu node contains feature maps
            # necessary for CropAndResize plugin.
            relu_name = "StatefulPartitionedCall/model/"
            relu_node = [node for node in self.graph.nodes if node.op == "Relu" and relu_name in node.name][-1]

            # Before passing 1st NMS's detection boxes (rois) to CropAndResize, we need to clip and normalize them.
            # Clipping happens for coordinates that are less than 0 and more than self.height.
            # Normalization is just divison of every coordinate by self.height.
            clip_min = np.asarray([0], dtype=np.float32)
            clip_max = np.asarray([self.height], dtype=np.float32)
            clip_out = self.graph.elt_const_clip("Clip", "FirstNMS/detection_boxes_clipper", input, clip_min, clip_max)
            div_const = np.expand_dims(np.asarray([self.height, self.height, self.height, self.height], dtype=np.float32), axis=(0, 1))
            div_out = self.graph.elt_const("Div", "FirstNMS/detection_boxes_normalizer", clip_out[0], div_const)

            # Linear transformation to convert box coordinates from (TopLeft, BottomRight) Corner encoding
            # to CenterSize encoding.
            matmul_const = np.matrix('0.5 0 -1 0; 0 0.5 0 -1; 0.5 0 1 0; 0 0.5 0 1', dtype=np.float32)
            matmul_out = self.graph.elt_const("MatMul", "FirstNMS/detection_boxes_conversion", div_out[0], matmul_const)

            # Additionally CropAndResizePlugin requires 4th dimension of 1: [N, B, 4, 1], so
            # we need to add unsqeeze node to make tensor 4 dimensional. 
            unsqueeze_node = self.graph.unsqueeze( "FirstNMS/detection_boxes_unsqueeze", div_out)

            # CropAndResizePlugin's inputs 
            feature_maps = relu_node.outputs[0]
            rois = unsqueeze_node[0]

            # CropAndResize TensorRT Plugin.
            # Two inputs are given to the CropAndResize TensorRT node:
            # - The feature_maps (from the Relu node found above): [batch_size, channel_num, height, width]
            # - The rois (in other words clipped and normalized detection boxes resulting fromm 1st NMS): [batch_size, featuremap, 4, 1]
            cnr_inputs = [feature_maps, rois]
            cnr_op = "CropAndResize"
            cnr_attrs = {
                'crop_width': self.initial_crop_size,
                'crop_height': self.initial_crop_size,
            }

            # CropAndResize Outputs.
            cnr_pfmap = gs.Variable(name="pfmap", dtype=np.float32,
                                           shape=[self.batch_size, self.first_stage_max_proposals, feature_maps.shape[1], self.initial_crop_size, self.initial_crop_size])
            cnr_outputs = [cnr_pfmap]

            # Create the CropandResize Plugin node with the selected inputs. 
            self.graph.plugin(
                op=cnr_op,
                name="cnr/crop_and_resize",
                inputs=cnr_inputs,
                outputs=cnr_outputs,
                attrs=cnr_attrs)
            log.info("Created CropAndResize plugin '{}' with attributes: {}".format(cnr_op, cnr_attrs))

            # Reshape node that is preparing CropAndResize's pfmap output shape for MaxPool node that comes next,
            # after that is 2nd backbone that leads us to final 2nd NMS.
            reshape_shape = np.asarray([self.first_stage_max_proposals*self.batch_size, feature_maps.shape[1], self.initial_crop_size, self.initial_crop_size], dtype=np.int64)
            reshape_node = self.graph.elt_const("Reshape", "StatefulPartitionedCall/CropandResize/reshape", cnr_outputs[0], reshape_shape)
            maxpl_name = "StatefulPartitionedCall/MaxPool2D/MaxPool"
            maxpool_node = [node for node in self.graph.nodes if node.op == "MaxPool" and maxpl_name == node.name][0]
            maxpool_node.inputs[0] = reshape_node[0]

            # Return linear transformation node, it will be located between 1st and 2nd NMS, 
            # so we need to pass and connect it to 2nd NMS.
            return matmul_out[0]

        def second_nms(input, threshold):
            """
            Updates the graph to replace the 2nd NMS op by BatchedNMS_TRT TensorRT plugin node.
            :param input: MatMul node that sits between 1st and 2nd NMS nodes.
            """

            # Identify Class Net and Box Net head names.
            second_head_names = ['StatefulPartitionedCall/mask_rcnn_keras_box_predictor/mask_rcnn_class_head/ClassPredictor_dense',
                'StatefulPartitionedCall/mask_rcnn_keras_box_predictor/mask_rcnn_box_head/BoxEncodingPredictor_dense']

            # Find the softmax node at the end of the 2nd class net (multi-scale class predictor).
            second_class_net = find_head_end(second_head_names[0], "MatMul", "Softmax")

            # Faster R-CNN's slice operation to adjust third dimension of Class Net's last node tensor (adjusting class values).
            slice_start = np.asarray([1], dtype=np.int64)
            slice_end = np.asarray([91], dtype=np.int64)
            # Second list element but third tensor dimension.
            slice_axes = np.asarray([2], dtype=np.int64)
            slice_out = self.graph.elt_const_slice("Slice", second_head_names[0]+"/slicer", second_class_net.outputs[0], slice_start, slice_end, slice_axes)

            # Final Class Net tensor.
            second_class_net_tensor = slice_out[0]
        
            # Find the add node at the end of the box net (multi-scale localization predictor).
            second_box_net = find_head_end(second_head_names[1], "MatMul", "Add")
            # Final Box Net tensor.
            second_box_net_output = second_box_net.outputs[0]

            # Reshape node that is preparing second_box_net_output's output shape for Mul scaling node that comes next.
            reshape_shape_second = np.asarray([self.batch_size, self.first_stage_max_proposals, second_box_net.outputs[0].shape[1]], dtype=np.int64)
            reshape_node_second = self.graph.elt_const("Reshape", second_head_names[1]+"/reshape", second_box_net_output, reshape_shape_second)
            # 0.1, 0.1, 0.2, 0.2 are localization head variance numbers, they scale second_box_net_output, in order to get accurate coordinates.
            second_scale_adj = np.expand_dims(np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32), axis=(0, 1))
            second_scale_out = self.graph.elt_const("Mul", second_head_names[1]+"/scale_second", reshape_node_second[0], second_scale_adj)

            # Final Box Net tensor.
            second_box_net_tensor = second_scale_out[0]

            # Set score threshold
            score_threshold = self.second_score_threshold if threshold is None else threshold

            # NMS Inputs and Attributes
            # NMS expects these shapes for its input tensors:
            # box_net: [batch_size, number_boxes, 4]
            # class_net: [batch_size, number_boxes, number_classes]
            # anchors: [1, number_boxes, 4] (if used)
            second_nms_op = None
            second_nms_attrs = None
            second_nms_inputs = None

            # EfficientNMS TensorRT Plugin is suitable for our use case.
            # Fusing the decoder will always be faster, so this is the default NMS method supported. In this case,
            # three inputs are given to the NMS TensorRT node:
            # - The box predictions (from the Box Net node found above)
            # - The class predictions (from the Class Net node found above)
            # - The default anchor coordinates (from the extracted anchor constants)
            # As the original tensors from given model will be used, the NMS code type is set to 1 (Center+Size),
            # because this is the internal box coding format used by the network.
            second_nms_inputs = [second_box_net_tensor, second_class_net_tensor, input]
            second_nms_op = "EfficientNMS_TRT"
            second_nms_attrs = {
                'plugin_version': "1",
                'background_class': -1,
                'max_output_boxes': self.first_stage_max_proposals,
                'score_threshold': max(0.01, score_threshold),
                'iou_threshold': self.second_iou_threshold,
                'score_activation': False,
                'box_coding': 1,
            }
            second_nms_output_classes_dtype = np.int32

            # NMS Outputs.
            second_nms_output_num_detections = gs.Variable(name="second_num_detections", dtype=np.int32, shape=[self.batch_size, 1])
            second_nms_output_boxes = gs.Variable(name="second_detection_boxes", dtype=np.float32,
                                           shape=[self.batch_size, self.first_stage_max_proposals, 4])
            second_nms_output_scores = gs.Variable(name="second_detection_scores", dtype=np.float32,
                                            shape=[self.batch_size, self.first_stage_max_proposals])
            second_nms_output_classes = gs.Variable(name="second_detection_classes", dtype=second_nms_output_classes_dtype,
                                             shape=[self.batch_size, self.first_stage_max_proposals])

            second_nms_outputs = [second_nms_output_num_detections, second_nms_output_boxes, second_nms_output_scores, second_nms_output_classes]

            # Create the NMS Plugin node with the selected inputs. 
            self.graph.plugin(
                op=second_nms_op,
                name="nms/non_maximum_suppression_second",
                inputs=second_nms_inputs,
                outputs=second_nms_outputs,
                attrs=second_nms_attrs)
            log.info("Created NMS plugin '{}' with attributes: {}".format(second_nms_op, second_nms_attrs))
            
            # Set graph outputs.
            self.graph.outputs = second_nms_outputs

            self.infer()

        # If you model is SSD, you need only one NMS and nothin else.
        if "ssd" in self.model:
            first_nms_output = first_nms(-1, True, threshold)
        # If your model is Faster R-CNN, you will need 2 NMS nodes with CropAndResize in between.
        elif "faster_rcnn" in self.model:
            first_nms_output = first_nms(0, False, threshold)
            cnr_output = crop_and_resize(first_nms_output)
            second_nms(cnr_output, threshold)


def main(args):
    effdet_gs = ModelGraphSurgeon(args.saved_model, args.pipeline_config)
    if args.tf2onnx:
        effdet_gs.save(args.tf2onnx)
    effdet_gs.update_preprocessor(args.batch_size)
    effdet_gs.process_graph(args.nms_threshold)
    if args.debug:
        effdet_gs.add_debug_output(args.debug)
    effdet_gs.save(args.onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pipeline_config", help="Pipeline configuration file to load", type=str)
    parser.add_argument("-m", "--saved_model", help="The TensorFlow saved model directory to load", type=str)
    parser.add_argument("-o", "--onnx", help="The output ONNX model file to write", type=str)
    parser.add_argument("-b", "--batch_size", help="Batch size for the model", type=int, default=1)
    parser.add_argument("-t", "--nms_threshold", help="Override the score threshold for the NMS operation", type=float)
    parser.add_argument("-d", "--debug", action='append', help="Add an extra output to debug a particular node")
    parser.add_argument("--tf2onnx", help="The path where to save the intermediate ONNX graph generated by tf2onnx, "
                                          "useful for debugging purposes, default: not saved", type=str)
    args = parser.parse_args()
    if not all([args.pipeline_config, args.saved_model, args.onnx]):
        parser.print_help()
        print("\nThese arguments are required: --pipeline_config, --saved_model and --onnx")
        sys.exit(1)
    main(args)

