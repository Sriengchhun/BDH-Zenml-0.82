## Activate ZenML environment in ubuntu
cd /home/chhun/Zenml/zenml/examples/quickstart
source /home/chhun/Zenml/zenml/examples/quickstart/ZenML/bin/activate

python run.py --test_size 0.4 --select_model Decision_Tree


zenml integration upgrade bentoml -y