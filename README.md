# NRVPrediction
This code in the current repository is developed as part of a project to forecast net regulation volume in the Norwegian power system. The code includes several different models, as well as an example dataset to demonstrate the use of the models.
The project is developed and tested using Python version 3.8. Other versions might work as well, but it has not been tested. 
## Train and evaluate a model:


1. Clone the repository and navigate into the folder:

    `git clone https://github.com/jonasoo98/NRVPrediction`

     `cd NRVPrediction/`

2. Create and activate a virtual environment. You can use the following snippets or use another environment of your choice.

    `python -m venv venv`

    `source venv/bin/activate`

3. Install dependencies:

    `pip install -r requirements.txt`

4. Train and evaluate a model:

     `python src/forecast.py --lookback_window 3 --target_column 'NO3_mFRR' --epochs 100 --horizon 24  --data_path <path_to_data_file> --model encoder_decoder`

5. For further options, run:
   
    `python forecast.py --help`
