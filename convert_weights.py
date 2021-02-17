import argparse

import numpy as np
import os

import tensorflow as tf
from AnimeGANv2.net import generator as tf_generator

import torch
from model import Generator


def load_tf_weights(tf_path):
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')
    with tf.variable_scope("generator", reuse=False):
        test_generated = tf_generator.G_net(test_real).fake

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})) as sess:
        ckpt = tf.train.get_checkpoint_state(tf_path)

        assert ckpt is not None and ckpt.model_checkpoint_path is not None, f"Failed to load checkpoint {tf_path}"

        saver.restore(sess, ckpt.model_checkpoint_path)
        print(f"Tensorflow model checkpoint {ckpt.model_checkpoint_path} loaded")

        tf_weights = {}
        for v in tf.trainable_variables():
            tf_weights[v.name] = v.eval()
    
    return tf_weights

            
def convert_keys(k):

    # 1. divide tf weight name in three parts [block_idx, layer_idx, weight/bias]
    # 2. handle each part & merge into a pytorch model keys
    
    k = k.replace("Conv/", "Conv_0/").replace("LayerNorm/", "LayerNorm_0/")    
    keys = k.split("/")[2:]
    
    is_dconv = False

    # handle C block..
    if keys[0] == "C":
        if keys[1] in ["Conv_1", "LayerNorm_1"]:
            keys[1] = keys[1].replace("1", "5")
        
        if len(keys) == 4:
            assert "r" in keys[1]

            if keys[1] == keys[2]:
                is_dconv = True
                keys[2] = "1.1"
            
            block_c_maps = {
                "1":  "1.2",
                "Conv_1":  "2",
                "2":  "3",
            }
            if keys[2] in block_c_maps:
                keys[2] = block_c_maps[keys[2]]

            keys[1] = keys[1].replace("r", "") + ".layers." + keys[2]
            keys[2] = keys[3]
            keys.pop(-1)
    assert len(keys) == 3

    # handle output block
    if "out" in keys[0]:
        keys[1] = "0"
    
    # first part
    if keys[0] in ["A", "B", "C", "D", "E"]:
        keys[0] = "block_" + keys[0].lower()        
        
    # second part
    if "LayerNorm_" in keys[1]:
        keys[1] = keys[1].replace("LayerNorm_", "") + ".2"
    if "Conv_" in keys[1]:
        keys[1] = keys[1].replace("Conv_", "") + ".1"
        
    # third part
    keys[2] = {
        "weights:0": "weight",
        "w:0": "weight",
        "bias:0": "bias",
        "gamma:0": "weight",
        "beta:0": "bias",
    }[keys[2]]
        
    return ".".join(keys), is_dconv


def convert_and_save(tf_checkpoint_path, save_name):

    tf_weights = load_tf_weights(tf_checkpoint_path)
    
    torch_net = Generator()
    torch_weights = torch_net.state_dict()

    torch_converted_weights = {}
    for k, v in tf_weights.items():
        torch_k, is_dconv = convert_keys(k)
        assert torch_k in torch_weights, f"weight name mismatch: {k}"

        converted_weight = torch.from_numpy(v)
        if len(converted_weight.shape) == 4:
            if is_dconv:
                converted_weight = converted_weight.permute(2, 3, 0, 1)
            else:
                converted_weight = converted_weight.permute(3, 2, 0, 1)

        assert torch_weights[torch_k].shape == converted_weight.shape, f"shape mismatch: {k}"

        torch_converted_weights[torch_k] = converted_weight

    assert sorted(list(torch_converted_weights)) == sorted(list(torch_weights)), f"some weights are missing"
    torch_net.load_state_dict(torch_converted_weights)    
    torch.save(torch_net.state_dict(), save_name)
    print(f"PyTorch model saved at {save_name}")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tf_checkpoint_path',
        type=str,
        default='AnimeGANv2/checkpoint/generator_Paprika_weight',
    )
    parser.add_argument(
        '--save_name', 
        type=str, 
        default='pytorch_generator_Paprika.pt',
    )
    args = parser.parse_args()
    
    convert_and_save(args.tf_checkpoint_path, args.save_name)