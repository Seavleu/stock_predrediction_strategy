import logging

def setup_logger():
    logger = logging.getLogger('StockPredictionLogger')
    handler = logging.FileHandler('trading_log.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
