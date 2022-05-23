# SplineDist
Pytorch implementation of the Spline dist cell segmentation method

# Configuration
Fellow the instruction to install singularity
https://singularity-tutorial.github.io/01-installation/

```bash
sudo singularity build container/torch.sif container/torch.def
```

```bash
singularity shell --nv container/torch.sif
```


## Train
Use notebooks found in src/Experiments/SplineDist


## Tensorboard
```bash
tensorboard --logsdir=src/Experiments/pl_logs_final2
```

## Inference
Use the inference notebook found in src, prove all the images that you want to infer in the inderenceDSB or inferenceCISD folder and run the code.
