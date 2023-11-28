import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f'{dir_path}/../')
sys.path.append(f'{dir_path}/../../')
sys.path.append(f'{dir_path}/src')
sys.path.append(f'{dir_path}/../../utils/experiments_utils')
sys.path.append(f'{dir_path}/../utils')