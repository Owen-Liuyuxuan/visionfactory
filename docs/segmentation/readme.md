# Vision Base - Segmentation

This repo use vision_base library extracts from [monodepth](http://gitlab.ram-lab.com/yuxuan/monodepth) to accelerate the development of segmentation related research.

The vision_base library contains generic codes for:

1. Config-based model/pipeline construction.
2. Common tools for pipeline organization, experiment logs.
3. Common mathematical, pytorch, numerical tools.

## Recommended Practice

1. Write code for dataset fetching, dataset evaluator, and network model.
2. Adapt config files based on examples to launch dataset/network/training with your own model. 
3. Use existing scripts and launchers to start experiments.
4. Minimize the modification in vision_base. Just write new classes in another folders if needed.
