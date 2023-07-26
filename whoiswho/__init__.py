import logging

logger = logging.getLogger("whoiswho")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

c_handler = logging.StreamHandler()
c_handler.setFormatter(formatter)
c_handler.setLevel(logging.INFO)

f_handler = logging.FileHandler('file.log') #relative path
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)


