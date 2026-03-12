export TQDM_DISABLE=1
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
bash /data1/mamingrui/LLM-DynDrive-pangu/scripts_all/extract_hidden_pangu_dist.sh > $PROJECT_DIR/outlog/extract_hidden_pangu_dist.log 2>&1 && \
bash /data1/mamingrui/LLM-DynDrive-pangu/scripts_all/hidden_analysis_pangu.sh > $PROJECT_DIR/outlog/hidden_analysis_pangu.log 2>&1 && \
bash /data1/mamingrui/LLM-DynDrive-pangu/scripts/dynamic_steer_pangu_dist_conf_all.sh > $PROJECT_DIR/outlog/dynamic_steer_pangu_dist_conf_all.log 2>&1
