# Vision Factory

This is a general repo for training and testing vision tasks (monocular 3D detection, segmentation, monodepth and more). 

The general starting points/runtime backbone will be in the "scripts", common modules, runtime plugins and helper functions will be in "vision_base". 

For different tasks, we may meed to overwrite parts of the dataloader / meta model / evaluator for different tasks. Checkout [visual3D], [segmentation], [monodepth] for more.


[visual3Ds]:docs/mono3d/readme.md
[segmentation]:docs/segmentation/readme.md
[monodepth]:docs/monodepth/readme.md