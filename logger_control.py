import os
import logging
from logging.handlers import RotatingFileHandler

logPath = os.path.abspath(os.path.join(os.path.dirname("__file__"), "log"))

pwd = os.getcwd()

if not os.path.exists(logPath):
    os.mkdir(logPath)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

log_error = logging.getLogger('error')
log_error.setLevel(logging.WARNING)
handler_error = RotatingFileHandler(os.path.join(logPath, 'error.log'), maxBytes=50 * 1024 * 1024, backupCount=5)
handler_error.setFormatter(formatter)
handler_error.encoding = 'utf-8'
log_error.addHandler(handler_error)

log_warning = logging.getLogger('warning')
log_warning.setLevel(logging.WARNING)
handler_warning = RotatingFileHandler(os.path.join(logPath, 'warning.log'), maxBytes=50 * 1024 * 1024, backupCount=5)
handler_warning.setFormatter(formatter)
handler_warning.encoding = 'utf-8'
log_warning.addHandler(handler_warning)

log_info = logging.getLogger('text_vec')
log_info.setLevel(logging.INFO)
handler_info = RotatingFileHandler(os.path.join(logPath, 'log_info.log'))
handler_info.setFormatter(formatter)
handler_info.encoding = 'utf-8'
log_info.addHandler(handler_info)
