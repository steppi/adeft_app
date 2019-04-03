import os
import boto3
import argparse

from deft_app.locations import DATA_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload model to S3')
    parser.add_argument('shortform')
    args = parser.parse_args()
    shortform = args.shortform
    local_models_path = os.path.join(DATA_PATH, 'models', shortform)
    s3_models_path = os.path.join('Models', shortform)

    file_names = [f'{shortform}_{end}' for end in
                  ('model.gz', 'grounding_map.json', 'names.json')]

    client = boto3.client('s3')
    for file_name in file_names:
        client.upload_file(os.path.join(local_models_path,
                                        file_name), 'deft-models',
                           os.path.join(s3_models_path, file_name))
