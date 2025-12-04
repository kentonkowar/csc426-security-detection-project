# to create a master data csv for all data types accross the whole week.
pwd
unzip ./MachineLearningCSV.zip

echo "creating masterData.csv file"
touch ./MachineLearningCVE/masterData.csv
echo "adding column names to masterData.csv file"
head -n 1 ./MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv >> ./MachineLearningCVE/masterData.csv
for file in ./MachineLearningCVE/*.pcap_ISCX.csv; do 
    echo "adding $file to masterData.csv"
    tail -n +2 $file >> ./MachineLearningCVE/masterData.csv; 
done
echo "completed making masterData.csv"