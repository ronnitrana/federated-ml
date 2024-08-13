import yaml
import subprocess

def update_yaml(file_path, max_attack_ratio, label_attack_ratio):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['max_attack_ratio'] = max_attack_ratio
    config['label_attack_ratio'] = label_attack_ratio

    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

yaml_file = 'C:/Users/ranar/Coding/federated-ml/conf/base.yaml'

for it_client_attack_ratio in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    print("client_attack_ratio:", it_client_attack_ratio)
    for it_label_attack_ratio in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:

        max_attack_ratio = it_client_attack_ratio 
        label_attack_ratio = it_label_attack_ratio 

        update_yaml(yaml_file, max_attack_ratio, label_attack_ratio)

        result = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
        
        print(result.stdout)