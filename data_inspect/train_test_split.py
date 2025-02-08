import os
import json
import random
from tqdm import tqdm

if __name__=='__main__':
    
    random.seed(123)
    
    train_ratio = 0.8
    val_ratio = 0.2
    test_ratio = 0.0
    
    json_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/all_data.json'
    out_dir = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/'
    datas = json.load(open(json_path))
    
    count_dict = {}
    for data in tqdm(datas, total=len(datas)):
        count_dict[data['label']] = count_dict.get(data['label'], 0) + 1
        
        if data['original_json_file'] not in count_dict.get(f"have_{data['label']}", []):
            count_dict[f"have_{data['label']}"] = count_dict.get(f"have_{data['label']}", []) + [data['original_json_file']]
            
    have_mitosis_paths = count_dict['have_Mitosis'].copy()
    random.shuffle(have_mitosis_paths)
    have_non_mitosis_paths = count_dict['have_Non-mitosis'].copy()
    random.shuffle(have_non_mitosis_paths)
    
    train_mitosis = have_mitosis_paths[:int(len(have_mitosis_paths) * train_ratio)]
    valid_mitosis = have_mitosis_paths[int(len(have_mitosis_paths) * train_ratio):int(len(have_mitosis_paths) * (train_ratio + val_ratio))]
    test_mitosis = have_mitosis_paths[int(len(have_mitosis_paths) * (train_ratio + val_ratio)):]
    
    train_no_mitosis = have_non_mitosis_paths[:int(len(have_non_mitosis_paths) * train_ratio)]
    valid_no_mitosis = have_non_mitosis_paths[int(len(have_non_mitosis_paths) * train_ratio):int(len(have_non_mitosis_paths) * (train_ratio + val_ratio))]
    test_no_mitosis = have_non_mitosis_paths[int(len(have_non_mitosis_paths) * (train_ratio + val_ratio)):]
    
    train_original_paths = train_mitosis + train_no_mitosis
    valid_original_paths = valid_mitosis + valid_no_mitosis
    test_original_paths = test_mitosis + test_no_mitosis
    
    train_patches, valid_patches, test_patches = [], [], []
    for data in tqdm(datas, total=len(datas)):
        if data['original_json_file'] in train_original_paths:
            train_patches.append(data)
        elif data['original_json_file'] in valid_original_paths:
            valid_patches.append(data)
        elif data['original_json_file'] in test_original_paths:
            test_patches.append(data)
        else: assert(0)
        
    train_save_path = os.path.join(out_dir, 'training_data.json')
    valid_save_path = os.path.join(out_dir, 'valid_data.json')
    test_save_path = os.path.join(out_dir, 'testing_data.json')
    
    with open(train_save_path, 'w') as f:
        json.dump(train_patches, f, indent=4)
    with open(valid_save_path, 'w') as f:
        json.dump(valid_patches, f, indent=4)
    with open(test_save_path, 'w') as f:
        json.dump(test_patches, f, indent=4)
        
    print(f"{len(train_patches)} train patches, {len(valid_patches)} valid patches, {len(test_patches)} test patches.")