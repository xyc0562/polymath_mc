import logging
import os
import time
import uuid
from datetime import datetime, timezone
from logging import handlers
from src.exceptions import ParamException

import pydash
from ruamel import yaml

from src import const


def environ():
    return os.environ.get('ENV', const.ENV_DEV)


def is_prod():
    return environ() == const.ENV_PROD


def is_dev():
    return environ() == const.ENV_DEV


def db_settings():
    with open('./config/db.yml', 'r') as fp:
        settings = yaml.load(fp, Loader=yaml.Loader)
    return settings[environ()]


def web3_settings():
    with open('./config/web3.yml', 'r') as fp:
        settings = yaml.load(fp, Loader=yaml.Loader)
    return settings

def web3_setting(acc_name):
    return web3_settings()[acc_name]

def account_settings():
    with open('./config/accounts.yml', 'r') as fp:
        settings = yaml.load(fp, Loader=yaml.Loader)
    return settings
def platform_settings():
    with open('./config/platform_config.yml', 'r') as fp:
        settings = yaml.load(fp, Loader=yaml.Loader)
    return settings

def decode_yaml(yaml_path):
    with open(yaml_path, 'r') as fp:
        settings = yaml.load(fp, Loader=yaml.Loader)
    return settings


def lg_settings():
    with open('./config/lg.yml', 'r') as fp:
        settings = yaml.load(fp, Loader=yaml.Loader)
    return settings[environ()]


def account_setting(acc_name, with_hidden_config=False):
    ret = account_settings()[acc_name]
    if with_hidden_config:
        hidden_file_name = pydash.get(ret, 'hidden_file_name', None)
        hidden_setting_name = pydash.get(ret, 'hidden_setting_name', None)
        if hidden_file_name is not None and hidden_setting_name is not None:
            hidden_setting = hidden_file_setting(hidden_file_name, hidden_setting_name)
            pydash.merge(ret, hidden_setting)
    return ret


def platform_setting(acc_name):
    return platform_settings()[acc_name]


def create_datbase_url(username, password, host, port, dbname):
    return 'postgresql+psycopg2://%s:%s@%s:%s/%s' %(username, password, host, port, dbname)


def create_default_database_url():
    ds = db_settings()
    return create_datbase_url(ds['username'], ds['password'], ds['host'], ds['port'], ds['dbname'])


def adapter_setting(acc_name, pairs):
    acc = account_setting(acc_name, True)
    return {
        **acc,
        'acc_name': acc_name,
        'pairs': pairs
    }


def strategy_settings(strtg_name):
    with open('./config/strategies/%s.yml' % strtg_name, 'r') as fp:
        YAML = yaml.YAML(typ='rt')
        settings = YAML.load(fp)
    return settings


def get_settings(file_name, cfg_name):
    with open('./config/%s.yml' % file_name, 'r') as fp:
        settings = yaml.load(fp, Loader=yaml.Loader)
    return settings[cfg_name]


def api_keys(cat):
    with open('./config/api_keys.yml', 'r') as fp:
        settings = yaml.load(fp, Loader=yaml.Loader)
    return settings[cat]


def strategy_setting(strtg_name, cfg_name):
    return strategy_settings(strtg_name)[cfg_name]


def hidden_file_settings(file_name):
    with open('./config/hidden_files/%s.yml' % file_name, 'r') as fp:
        YAML = yaml.YAML(typ='rt')
        settings = YAML.load(fp)
    return settings


def hidden_file_setting(hidden_file_name, setting_name):
    return hidden_file_settings(hidden_file_name)[setting_name]


def generate_uuid(length=None):
    if length is None:
        return str(uuid.uuid4())
    else:
        return str(uuid.uuid4())[:length]


def time_millisecond():
    return (int)(time.time() * 1000)


def time_second() -> int:
    """
    :return: current unix time in second
    """
    return (int)(time.time())


def time_second_float():
    return time.time()


def ts_to_local_time(ts):
    return datetime.fromtimestamp(ts)


def time_string(tm: datetime or None=None):
    """
    Deprecated
    WARNING..datetime do not have a timezone, but this function will append localtime timezone into the string
    """
    #'2018-09-01T07:25:06:575043 +08'
    return (tm or datetime.now()).strftime('%Y-%m-%dT%H:%M:%S:%f ') + time.localtime().tm_zone


def from_utc_datetime_to_second(dt: datetime) -> int:
    # to unix timestamp
    if dt.tzinfo is not None and dt.tzinfo != timezone.utc:
        raise ParamException('datetime timezone is not utc: %s' % dt)
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def from_utc_datetime_to_local_string(dt: datetime):
    # '2018-09-01T07:25:06:575043'
    if dt.tzinfo is not None and dt.tzinfo != timezone.utc:
        raise ParamException('datetime timezone is not utc: %s' % dt)
    dt = dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
    return dt.strftime('%Y-%m-%dT%H:%M:%S:%f %Z')


def from_iso_time_string(s=None):
    # '2019-03-29T03:00:00+00:00'
    return datetime.fromisoformat(s)


def from_millisecond_to_datetime(millisecond) -> datetime:
    return datetime.fromtimestamp(float(millisecond) / 1000.0)


def from_millisecond_to_utc_datetime(millisecond) -> datetime:
    dt = datetime.utcfromtimestamp(float(millisecond) / 1000.0)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def from_second_to_utc_datetime(second) -> datetime:
    dt = datetime.utcfromtimestamp(float(second))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_kalman_indicator_name(adapter_acc_name, pair, cfg):
    return "%s_%s_%s_%s_%s" %(adapter_acc_name, pair, cfg['interval'], cfg['rolling_count'], cfg['alignment'])


def utc_now():
    return datetime.utcnow()


class FileHandlers:

    def __init__(self):
        # map from file name to handler
        self.handler_dict = {}

    def get_handler(self, file_name, lvl=logging.INFO):
        # in pydash, the dot refers to the nested object
        key = str(str(lvl) + '_' + str(file_name)).replace('.', '')
        fh = pydash.get(self.handler_dict, key)
        if fh is None:
            nfh = handlers.RotatingFileHandler(self._log_file_path(file_name), maxBytes=2000000, backupCount= 5, encoding='utf-8') # 2MB
            nfh.setLevel(lvl)
            formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
            nfh.setFormatter(formatter)
            self.handler_dict[key] = nfh
            return nfh
        else:
            return fh

    def _log_file_path(self, file_name):
        current_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(current_path, '../../log')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return os.path.join(dir_path, file_name)


class LoggerManager:

    def __init__(self):
        self.loggers = {}

    def get_logger(self, name='main', file_name=None, enable_file_handler=False):
        key = str(name) + '_' + str(file_name)
        lg = pydash.get(self.loggers, key)
        if lg is None:
            lg = self._create_logger(name, file_name, enable_file_handler)
            self.loggers[key] = lg
        return lg

    def _create_logger(self, name='main', file_name=None, enable_file_handler=False):
        logger = logging.getLogger(name)
        lvl = os.environ.get('LOG_LEVEL', logging.INFO)
        logger.setLevel(lvl)
        lsh = logging.StreamHandler()
        lsh.setLevel(logging.NOTSET)
        formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        lsh.setFormatter(formatter)
        logger.addHandler(lsh)

        if enable_file_handler:
            if file_name is None:
                efh = FILE_HANDLER.get_handler('error.log', logging.ERROR)
                ifh = FILE_HANDLER.get_handler('info.log', logging.INFO)
            else:
                efh = FILE_HANDLER.get_handler(file_name + '_error.log', logging.ERROR)
                ifh = FILE_HANDLER.get_handler(file_name + '_info.log', logging.INFO)

            logger.addHandler(efh)
            logger.addHandler(ifh)
        logger.propagate = False
        return logger


FILE_HANDLER = FileHandlers()
LOGGER_MANAGER = LoggerManager()


def get_logger(name='main', file_name=None, enable_file_handler=False):
    return LOGGER_MANAGER.get_logger(name, file_name, enable_file_handler)


def read_prompt(name) -> str:
    with open('./config/ai_agent_prompts/' + name) as fp:
        return fp.read()


# shared logger
APP_LOGGER = get_logger(enable_file_handler=False)
