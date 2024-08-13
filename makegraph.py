import os
import re
import yaml

def access_specific_accuracy_in_log_files(base_folder, target_file_name, round_number=10):
    folder_list = sorted(os.listdir(base_folder), reverse=True)

    for folder_name in folder_list:
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            target_file_path = os.path.join(folder_path, target_file_name)
            hydrapath = folder_path + "/.hydra" + "/config.yaml"

            with open(hydrapath, 'r') as file:
                config = yaml.safe_load(file)
            if os.path.isfile(target_file_path):
                with open(target_file_path, 'r') as file:
                    content = file.read()

                    # Search for the accuracy of the specified round
                    pattern = fr'\({round_number}, ([\d\.]+)\)'
                    match = re.search(pattern, content)
                    if match:
                        accuracy = match.group(1)
                        maxattackval = int(config['max_attack_ratio'] * 10)
                        labelval = int(config['label_attack_ratio'] * 100)

                        print(f"({maxattackval}, {labelval}, {accuracy})")
                    else:
                        print(f"Accuracy for round {round_number} not found in {target_file_path}")
            else:
                print(f"{target_file_name} does not exist in {folder_path}")

# Example usage
base_folder = 'C:/Users/ranar/Coding/federated-ml/3d_100_outputs/2024-08-13'
target_file_name = 'main.log'
access_specific_accuracy_in_log_files(base_folder, target_file_name, round_number=10)
