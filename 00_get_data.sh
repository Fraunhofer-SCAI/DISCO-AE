#!/bin/bash

dataset=$1

echo "Download data and maps for dataset: " $dataset

mkdir -p data
cd data

if [[ "$dataset" == "gallop" ]]
then
    folder=Data_elephant
    wget https://owncloud.scai.fraunhofer.de/index.php/s/5oiYYZPYf7ipmCt/download/Data_elephant.tar.gz
    echo "tar -xzvf $folder.tar.gz && rm $folder.tar.gz"
    tar -xzvf $folder.tar.gz && rm $folder.tar.gz

    folder=Data_horse
    wget https://owncloud.scai.fraunhofer.de/index.php/s/nfPjcsJCAntDdjk/download/Data_horse.tar.gz
    echo "tar -xzvf $folder.tar.gz && rm $folder.tar.gz"
    tar -xzvf $folder.tar.gz && rm $folder.tar.gz

    folder=Data_camel
    wget https://owncloud.scai.fraunhofer.de/index.php/s/x6Mi5Q3nzKRrjzK/download/Data_camel.tar.gz
    echo "tar -xzvf $folder.tar.gz && rm $folder.tar.gz"
    tar -xzvf $folder.tar.gz && rm $folder.tar.gz

    echo "load the p2p maps"
    wget https://owncloud.scai.fraunhofer.de/index.php/s/52mzPiQeJQ5kZpB/download/GALLOP_samp_.tar.gz
    maps=GALLOP_samp_
    tar -xzvf $maps.tar.gz && rm $maps.tar.gz

elif [[ "$dataset" == "TRUCK" ]]
then
    folder=TRUCK_data_DISCO_AE
    wget https://owncloud.scai.fraunhofer.de/index.php/s/9Ke2fXNcepPwqas/download/TRUCK_data_DISCO_AE.tar
    echo "tar -xzvf $folder.tar && rm $folder.tar"
    tar -xzvf $folder.tar && rm $folder.tar

elif [[ "$dataset" == "FAUST" ]]
then
    folder=Data_FAUST
    wget https://owncloud.scai.fraunhofer.de/index.php/s/DTHAEB9BfyZsar6/download/Data_FAUST.tar.gz
    echo "tar -xzvf $folder.tar.gz && rm $folder.tar.gz"
    tar -xzvf $folder.tar.gz && rm $folder.tar.gz

    echo "load the p2p maps"
    wget https://owncloud.scai.fraunhofer.de/index.php/s/B93fZJAdxzqJ2yL/download/FAUST_.tar.gz
    maps=FAUST_
    tar -xzvf $maps.tar.gz && rm $maps.tar.gz

else
    echo "Not a valid dataset."
fi

cd ..

echo "downloaded the mesh data, true p2p maps, and unsupervised p2p maps, and putting it in data directory"

