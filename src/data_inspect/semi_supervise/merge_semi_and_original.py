import os
import json

def load_json(json_path):
    return json.load(open(json_path))

def merge_datas(data1, data2):
    return data1 + data2

def main():
    json_path_original = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/by_patches/testing_data.json'
    json_path_new = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/testing_data.json'
    merged_path = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/processed_format/merged_testing_data.json'
    
    data_json_original = load_json(json_path_original)
    data_json_new = load_json(json_path_new)
    
    merged_data = merge_datas(data_json_original, data_json_new)
    
    with open(merged_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
        
if __name__=='__main__':
    main()