import logging
from logging.config import fileConfig


def init_ini_log() -> None:
    fileConfig("log.ini")


init_ini_log()
logger = logging.getLogger()
