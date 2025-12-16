# setup the python environment
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt


# ensure data is present
if [ -d ./MachineLearningCVE ]; then
    echo "data here [MachineLearningCVE]"
else
    echo "running data setup script"
    ./createMasterData.sh
fi
