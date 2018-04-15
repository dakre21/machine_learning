import coloredlogs, logging

# Create logger object for application
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')

