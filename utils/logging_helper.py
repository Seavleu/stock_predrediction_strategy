import logging 

# # ✅ Create a logger for baseline_lstm
# lstm_logger = logging.getLogger("baseline_lstm")
# lstm_logger.setLevel(logging.INFO)
# lstm_handler = logging.FileHandler("logs/baseline_lstm.log")
# lstm_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# lstm_handler.setFormatter(lstm_formatter)
# lstm_logger.addHandler(lstm_handler)

# # ✅ Create a separate logger for baseline_lstm_2
# lstm2_logger = logging.getLogger("baseline_lstm_2")
# lstm2_logger.setLevel(logging.INFO)
# lstm2_handler = logging.FileHandler("logs/baseline_lstm_2.log")
# lstm2_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# lstm2_handler.setFormatter(lstm2_formatter)
# lstm2_logger.addHandler(lstm2_handler)

# def log_message(logger_name: str, message: str):
#     """Logs messages to the specified logger and prints them."""
#     if logger_name == "baseline_lstm":
#         lstm_logger.info(message)
#     elif logger_name == "baseline_lstm_2":
#         lstm2_logger.info(message)
#     print(message)  


log_files = {
    "baseline_lstm": "logs/baseline_lstm.log",
    "baseline_lstm_2": "logs/baseline_lstm_2.log"
}

def log_message(log_file, message):
    """Logs messages to a specified file and console."""
    if log_file not in log_files:
        raise ValueError(f"Invalid log file name: {log_file}")

    logging.basicConfig(
        filename=log_files[log_file],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )

    logging.info(message)
    print(message) 
