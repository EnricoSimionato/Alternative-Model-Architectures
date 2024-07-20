import os
import shutil

# Directories are defined starting inside src
experiments_directory = os.path.join(os.getcwd(), 'experiments', 'performed_experiments')
destination_directory = os.path.join(experiments_directory, 'all_logs')

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Iterating through all items in the current directory
for item in os.listdir(experiments_directory):
    item_path = os.path.join(experiments_directory, item)
    logs_path = os.path.join(item_path, 'logs')

    # Checking if the item is a directory and contains a 'logs' directory
    if os.path.isdir(item_path) and os.path.exists(logs_path) and os.path.isdir(logs_path):
        # Creating the corresponding directory in the destination directory
        new_dir_path = os.path.join(destination_directory, item)
        os.makedirs(new_dir_path, exist_ok=True)

        # Moving the 'logs' directory to the new directory
        shutil.copytree(logs_path, os.path.join(new_dir_path, 'logs'), dirs_exist_ok=True)
        print(f"Copied {logs_path} to {os.path.join(new_dir_path, 'logs')}")
