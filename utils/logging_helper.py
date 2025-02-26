import logging

logging.basicConfig(
    filename="logs/baseline_lstm.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_message(message):
    """Logs messages to console and file."""
    logging.info(message)
    print(message) 
