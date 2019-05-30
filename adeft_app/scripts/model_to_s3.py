import os
import json
import boto3
import argparse
import tempfile

from adeft.download import get_s3_models

from adeft_app.locations import DATA_PATH, S3_BUCKET
from adeft_app.filenames import escape_filename


def model_to_s3(model_name):
    model_name = escape_filename(model_name)
    local_models_path = os.path.join(DATA_PATH, 'models', model_name)
    with open(os.path.join(local_models_path,
                           f'{model_name}_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    model_map = {key: model_name for key in grounding_dict}
    s3_models = get_s3_models()
    s3_models.update(model_map)

    client = boto3.client('s3')
    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            json.dump(s3_models, f)
        client.upload_file(temp.name, S3_BUCKET, 's3_models.json')

    file_names = [f'{model_name}_{end}' for end in
                  ('model.gz', 'grounding_dict.json', 'names.json')]

    for file_name in file_names:
        client.upload_file(os.path.join(local_models_path,
                                        file_name), S3_BUCKET,
                           os.path.join(model_name, file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload model to S3')
    parser.add_argument('model_name')
    args = parser.parse_args()
    model_name = args.model_name
    model_to_s3(model_name)
