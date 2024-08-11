unset PYTHONPATH

DIR=`pwd`
export PYTHONPATH=$DIR:$PYTHONPATH
export PYTHONPATH=$DIR"/puzzlegen":$PYTHONPATH

export RL_ENV_DATA="../../../rl_env_data/"
#export RL_ENV_DATA=$DIR"/data"
