import os

# Implemented verions of the model
# Default is always latest (at the moment 3)
version_array = [1, 2, 3]

graphics_dir = "./graphics/"
data_dir = "./data/"
data_test_dir = "./data/test/"

for version in version_array:
    os.makedirs(graphics_dir + f'v{version}/', exist_ok=True)
    os.makedirs(data_dir + f'v{version}/', exist_ok=True)

marker_list = ['o', '^', 's', 'h', 'v', 'D',]
