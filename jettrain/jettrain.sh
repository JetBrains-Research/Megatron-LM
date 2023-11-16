source jettrain/secrets.sh
jettrain \
--config-dir /home/galimzyanov/Megatron-LM/jettrain \
teamcity.token=$TEAMCITY_TOKEN \
storages.0.credentials.access_key_id=$AWS_ACCESS_KEY_ID \
storages.0.credentials.secret_access_key=$AWS_SECRET_ACCESS_KEY \
env.secrets.variables.WANDB_API_KEY=$WANDB_API_KEY