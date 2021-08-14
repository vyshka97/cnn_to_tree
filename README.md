## CNN to Tree

The realization of paper: [Interpreting CNNs via Decision Trees](https://arxiv.org/abs/1802.00121)
on Pytorch 1.7.0

And also of [Interpretable Convolutional Neural Networks](https://arxiv.org/abs/1710.00935)

### Project structure
- `xai` - Python module with training stuff (datasets, losses, algorithms, models, snapshot saving, etc)
- `experiments` - directory containing all experiments
- `utils` - directory with Python scripts for source data transformation (from jpg to hdf5)
- `build_docker_image.sh` - Bash script for building Docker image, described in this `Dockerfile`
- `run_container.sh` - Bash script for Docker container running

### Requirements
- Pytorch 1.7.0
- torchvision 
- Everything else in `requirements.txt`
