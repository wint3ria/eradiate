import logging
import re

from tqdm.auto import tqdm


def _add_logging_level(level_name, level_num, method_name=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> _add_logging_level('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5
    """
    # See also: https://stackoverflow.com/a/35804945/3645374
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        raise AttributeError(f"{level_name} already defined in logging module")
    if hasattr(logging, method_name):
        raise AttributeError(f"{method_name} already defined in logging module")
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError(f"{method_name} already defined in logger class")

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)


# Regex for log parsing
mts_log_parser = re.compile(
    r"^(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*"
    r"(?P<level>\S*)\s*"
    r"(?P<thread>\S*)\s*"
    r"\[(?P<class>\S*)\]\s*"
    r"(?P<message>.*)$"
)

# Mitsuba Logger instance
mts_logger = None

# Module-local logger
logger = logging.getLogger(__name__)


def _get_logger():
    """
    Get Mitsuba Logger instance
    """
    from mitsuba.core import Thread

    return Thread.thread().logger()


def install_logging(force=False):
    """
    Install logging
    """
    global mts_logger

    if mts_logger is not None and not force:
        return

    from mitsuba.core import Appender, LogLevel

    try:
        _add_logging_level("TRACE", logging.DEBUG - 5, "trace")
    except AttributeError:
        pass

    LOGLEVEL_DISPATCH = {
        LogLevel.Trace: "trace",
        LogLevel.Debug: "debug",
        LogLevel.Info: "info",
        LogLevel.Warn: "warning",
        LogLevel.Error: "error",
    }

    class EradiateAppender(Appender):
        def __init__(self):
            super().__init__()
            self.reset()

        def reset(self):
            self.bar = None
            self.progress = None

        def append(self, level, text):
            m = mts_log_parser.match(text)
            msg = (
                f"{m.group('thread')} [{m.group('class')}] {m.group('message')}"
                if m is not None
                else text
            )
            getattr(logger, LOGLEVEL_DISPATCH[level])(msg)

        def log_progress(self, progress, name, formatted, eta, ptr=None):
            if self.bar is None:
                self.bar = tqdm(
                    desc="Mitsuba",
                    initial=0.0,
                    total=1.0,
                    unit_scale=1.0,
                    leave=True,
                    bar_format="{l_bar}{bar}| {elapsed}, ETA={remaining}",
                )
                self.progress = 0.0

            self.bar.update(progress - self.progress)
            self.progress = progress

            if progress >= 1.0:
                self.bar.close()
                self.reset()

    if mts_logger is None:
        mts_logger = _get_logger()
        mts_logger.clear_appenders()
        mts_logger.add_appender(EradiateAppender())
        mts_logger.set_log_level(LogLevel.Info)
