from frink.configs.base import PathConfig


local_paths_config = PathConfig(dict(
    bucket='aiiku-frink-dev',
    athena_output='s3://aws-athena-query-results-586228983293-eu-west-1/',
    output='output',
))


dev_paths_config = PathConfig(dict(
    bucket='aiiku-frink-dev',
    athena_output='s3://aws-athena-query-results-586228983293-eu-west-1/',
    output='output',
))


prod_paths_config = PathConfig(dict(
    bucket='aiiku-frink-prod',
    athena_output='s3://aws-athena-query-results-750076298197-eu-west-1/',
    output='output',
))
