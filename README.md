# NRVPrediction

1. Clone repository and navigate into folder:

    `git clone https://github.com/jonasoo98/NRVPrediction`

     `cd NRVPrediction/`

2. Create and activate a virtual environment. You can use the following snippets or use another environment of your choice.

    `python -m venv venv`

    `source venv/bin/activate`

3. Install dependencies:

    `pip install -r requirements.txt`

4. Make a prediction:

     `python forecast.py --lookback_window 3 --target_column 'NO3_mFRR' --epochs 100 --horizon 24  --data_path <path_to_data_file> --model encoder_decoder`

5. For further options, run:
   
    `python forecast.py --help`
